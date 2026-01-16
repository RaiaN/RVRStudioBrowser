#!/usr/bin/env python3
"""
generate_mpi.py
Multi-Plane Image (MPI) Generation Pipeline

A practical single-image "true 3D" preprocessing pipeline that:
- Detects guaranteed hidden (disoccluded) regions under small virtual camera motion
- Segments the image into depth layers (foreground / midground / background)
- Generates Multi-Plane Image (MPI) data structures
- Prepares inpainting masks for each layer (no generative model required)

Philosophy:
    We are discovering WHERE content MUST be missing, not guessing WHAT it is.
    Geometry first, hallucination later.

Requirements:
    pip install numpy opencv-python scipy matplotlib

Usage:
    # Single image with depth map
    python generate_mpi.py image.jpg depth.png --output ./mpi_output/
    
    # Batch process directory
    python generate_mpi.py ./images/ ./depths/ --output ./mpi_output/ --batch
    
    # Custom camera shift
    python generate_mpi.py image.jpg depth.png --shift 10 --output ./mpi_output/
    
    # Custom number of layers
    python generate_mpi.py image.jpg depth.png --layers 5 --output ./mpi_output/
    
    # Generate debug visualizations
    python generate_mpi.py image.jpg depth.png --output ./mpi_output/ --debug

Author: RVR Studio Team
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2

# Optional scipy for improved processing
try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, sobel
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using OpenCV fallback.")

# Optional matplotlib for debug visualizations
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Debug visualizations disabled.")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MPILayer:
    """
    Represents a single layer in the Multi-Plane Image.
    
    Each layer contains:
    - RGB color data for pixels belonging to this depth range
    - Alpha mask indicating which pixels belong to this layer
    - Representative depth value for the layer plane
    - Inpainting mask showing regions that need content generation
    """
    rgb: np.ndarray              # (H, W, 3) uint8 - Color data
    alpha: np.ndarray            # (H, W) uint8 - Layer membership mask [0, 255]
    depth: float                 # Normalized depth value [0, 1] for this plane
    depth_range: Tuple[float, float]  # (min, max) depth bounds
    inpaint_mask: np.ndarray     # (H, W) uint8 - Regions needing inpainting [0, 255]
    layer_index: int             # Layer index (0 = background, N-1 = foreground)
    name: str                    # Human-readable name (e.g., "background", "foreground")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary (without image data)."""
        return {
            "layer_index": self.layer_index,
            "name": self.name,
            "depth": self.depth,
            "depth_range": list(self.depth_range),
            "pixel_count": int(np.sum(self.alpha > 0)),
            "inpaint_pixel_count": int(np.sum(self.inpaint_mask > 0))
        }


@dataclass
class MPIResult:
    """
    Complete MPI output containing all layers and metadata.
    """
    layers: List[MPILayer]
    occlusion_mask: np.ndarray   # (H, W) uint8 - Global disocclusion mask
    depth_edges: np.ndarray      # (H, W) uint8 - Depth discontinuity edges
    source_image: np.ndarray     # (H, W, 3) uint8 - Original RGB
    source_depth: np.ndarray     # (H, W) float32 - Normalized depth [0, 1]
    width: int
    height: int
    camera_shift: float          # Virtual camera shift used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "camera_shift": self.camera_shift,
            "num_layers": len(self.layers),
            "layers": [layer.to_dict() for layer in self.layers],
            "total_disoccluded_pixels": int(np.sum(self.occlusion_mask > 0))
        }


# ============================================================================
# STEP 1: LOAD INPUTS
# ============================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Load RGB image and ensure proper format.
    
    Args:
        image_path: Path to image file
        
    Returns:
        (H, W, 3) uint8 numpy array in RGB format
    """
    print(f"  Loading image: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # OpenCV loads as BGR, convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"    Dimensions: {img_rgb.shape[1]}x{img_rgb.shape[0]} (W×H)")
    return img_rgb


def load_depth(depth_path: str, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Load depth map and normalize to [0, 1] range.
    
    Depth convention:
    - 0.0 = far (background)
    - 1.0 = near (foreground)
    
    This matches the "closer = brighter" convention used by most depth estimators.
    
    Args:
        depth_path: Path to depth map (grayscale image)
        target_shape: (height, width) to resize to if needed
        
    Returns:
        (H, W) float32 numpy array normalized to [0, 1]
    """
    print(f"  Loading depth: {depth_path}")
    
    # Load as grayscale
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth: {depth_path}")
    
    # Resize if needed
    if depth.shape[:2] != target_shape:
        print(f"    Resizing depth from {depth.shape[:2]} to {target_shape}")
        depth = cv2.resize(depth, (target_shape[1], target_shape[0]), 
                          interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1]
    depth_float = depth.astype(np.float32) / 255.0
    
    print(f"    Depth range: [{depth_float.min():.3f}, {depth_float.max():.3f}]")
    return depth_float


# ============================================================================
# STEP 2: SIMULATE VIRTUAL CAMERA MOTION
# ============================================================================

def project_to_new_view(
    rgb: np.ndarray,
    depth: np.ndarray,
    shift_x: float,
    shift_y: float = 0.0,
    depth_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate virtual camera motion via depth-based pixel displacement.
    
    GEOMETRIC REASONING:
    Each pixel is treated as a 3D point P = (x, y, depth(x,y)).
    Under lateral camera motion, pixels shift by an amount proportional
    to their inverse depth (parallax effect):
    - Near pixels (high depth value) shift more
    - Far pixels (low depth value) shift less
    
    This is a simplified model that assumes:
    - Orthographic projection (no perspective distortion)
    - Small camera motion (no significant occlusion changes from depth variation)
    
    Args:
        rgb: (H, W, 3) Source image
        depth: (H, W) Normalized depth [0, 1]
        shift_x: Horizontal camera shift in pixels (positive = move right)
        shift_y: Vertical camera shift in pixels
        depth_scale: Multiplier for depth influence on parallax
        
    Returns:
        projected_rgb: (H, W, 3) Reprojected image with holes
        projected_depth: (H, W) Reprojected depth map
        valid_mask: (H, W) Boolean mask of pixels that received data
    """
    print(f"  Projecting with shift: ({shift_x}, {shift_y}) pixels")
    
    H, W = depth.shape
    
    # Initialize output buffers
    # Use -1 for depth buffer to indicate "no data yet"
    projected_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    projected_depth = np.full((H, W), -1.0, dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    
    # Calculate per-pixel displacement based on depth
    # Near pixels (depth ~1) shift by full amount
    # Far pixels (depth ~0) shift minimally
    displacement_x = shift_x * depth * depth_scale
    displacement_y = shift_y * depth * depth_scale
    
    # New coordinates after displacement
    new_x = x_coords + displacement_x
    new_y = y_coords + displacement_y
    
    # Round to integer coordinates for simple splatting
    # (More sophisticated: bilinear splatting, but for occlusion detection this suffices)
    new_x_int = np.round(new_x).astype(np.int32)
    new_y_int = np.round(new_y).astype(np.int32)
    
    # Process each pixel using z-buffer logic
    # We iterate in order of increasing depth (far to near)
    # so nearer pixels overwrite farther ones
    depth_order = np.argsort(depth.ravel())
    
    for idx in depth_order:
        src_y, src_x = divmod(idx, W)
        dst_x = new_x_int[src_y, src_x]
        dst_y = new_y_int[src_y, src_x]
        
        # Bounds check
        if 0 <= dst_x < W and 0 <= dst_y < H:
            pixel_depth = depth[src_y, src_x]
            
            # Z-buffer test: only write if this pixel is closer (higher depth)
            # or if no data exists yet
            if projected_depth[dst_y, dst_x] < pixel_depth:
                projected_rgb[dst_y, dst_x] = rgb[src_y, src_x]
                projected_depth[dst_y, dst_x] = pixel_depth
                valid_mask[dst_y, dst_x] = True
    
    # Replace -1 placeholder with 0 in depth
    projected_depth[projected_depth < 0] = 0
    
    return projected_rgb, projected_depth, valid_mask


# ============================================================================
# STEP 3 & 4: Z-BUFFER VISIBILITY & DISOCCLUSION DETECTION
# ============================================================================

def detect_disocclusions(
    rgb: np.ndarray,
    depth: np.ndarray,
    shift_x: float,
    shift_y: float = 0.0,
    depth_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect disoccluded (hidden) regions revealed by virtual camera motion.
    
    GEOMETRIC REASONING:
    When a camera moves laterally, foreground objects appear to move
    more than background objects (parallax). This reveals regions that
    were hidden behind foreground objects in the original view.
    
    By simulating this motion and tracking which target pixels never
    receive source data, we identify guaranteed disoccluded regions:
    - These are pixels that MUST have hidden content
    - No ML model can reliably know what's there
    - This information is geometric ground truth
    
    Args:
        rgb: (H, W, 3) Source image
        depth: (H, W) Normalized depth
        shift_x: Camera shift magnitude
        shift_y: Vertical shift (usually 0)
        depth_scale: Depth influence multiplier
        
    Returns:
        occlusion_mask: (H, W) uint8 binary mask (255 = disoccluded)
        projected_rgb: (H, W, 3) Reprojected view for debugging
        valid_mask: (H, W) Boolean mask of valid projections
    """
    print("  Detecting disocclusions via virtual camera motion...")
    
    # Project to new view
    projected_rgb, projected_depth, valid_mask = project_to_new_view(
        rgb, depth, shift_x, shift_y, depth_scale
    )
    
    # Disoccluded = pixels that received no data
    occlusion_mask = (~valid_mask).astype(np.uint8) * 255
    
    # Clean up the mask using morphological operations
    # Small isolated holes are likely interpolation artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes in the mask
    occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
    
    num_disoccluded = np.sum(occlusion_mask > 0)
    total_pixels = occlusion_mask.size
    pct = 100.0 * num_disoccluded / total_pixels
    print(f"    Disoccluded pixels: {num_disoccluded:,} ({pct:.2f}%)")
    
    return occlusion_mask, projected_rgb, valid_mask


# ============================================================================
# STEP 5: COMPUTE DEPTH EDGES
# ============================================================================

def compute_depth_edges(
    depth: np.ndarray,
    threshold: float = 0.1,
    blur_sigma: float = 1.0
) -> np.ndarray:
    """
    Detect depth discontinuities (edges) in the depth map.
    
    GEOMETRIC REASONING:
    Depth discontinuities indicate object boundaries where:
    - Foreground objects end and background begins
    - Occlusion relationships exist
    - MPI layers should potentially separate
    
    These edges guide layer segmentation and help identify
    regions prone to disocclusion artifacts.
    
    Args:
        depth: (H, W) Normalized depth map
        threshold: Gradient magnitude threshold (in depth units)
        blur_sigma: Gaussian blur sigma before edge detection
        
    Returns:
        edges: (H, W) uint8 edge map (255 = strong depth edge)
    """
    print("  Computing depth discontinuity edges...")
    
    # Smooth depth to reduce noise
    if SCIPY_AVAILABLE:
        depth_smooth = gaussian_filter(depth, sigma=blur_sigma)
        # Compute gradients using Sobel
        grad_x = sobel(depth_smooth, axis=1)
        grad_y = sobel(depth_smooth, axis=0)
    else:
        # OpenCV fallback
        depth_smooth = cv2.GaussianBlur(depth, (0, 0), blur_sigma)
        grad_x = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)
    
    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize and threshold
    grad_normalized = grad_mag / (grad_mag.max() + 1e-8)
    edges = (grad_normalized > threshold).astype(np.uint8) * 255
    
    # Optional: thin edges using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    print(f"    Edge pixels: {np.sum(edges > 0):,}")
    return edges


# ============================================================================
# STEP 6: DEPTH-BASED LAYER SEGMENTATION
# ============================================================================

def segment_depth_layers(
    rgb: np.ndarray,
    depth: np.ndarray,
    num_layers: int = 3,
    overlap_margin: float = 0.02
) -> List[Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]]:
    """
    Segment image into depth layers using quantization.
    
    GEOMETRIC REASONING:
    MPI represents a scene as a stack of RGBA planes at discrete depths.
    By quantizing the continuous depth map into layers, we create:
    - Background layer (far, low depth values)
    - Midground layers (intermediate)
    - Foreground layer (near, high depth values)
    
    A small overlap margin ensures layer transitions are smooth
    and reduces hard edge artifacts.
    
    Args:
        rgb: (H, W, 3) Source image
        depth: (H, W) Normalized depth [0, 1]
        num_layers: Number of depth layers to create
        overlap_margin: Depth overlap between adjacent layers
        
    Returns:
        List of (rgb_layer, alpha_mask, depth_value, depth_range) tuples
        Sorted from background (far) to foreground (near)
    """
    print(f"  Segmenting into {num_layers} depth layers...")
    
    H, W = depth.shape
    layers = []
    
    # Calculate layer boundaries
    # Depth 0 = far, Depth 1 = near
    depth_step = 1.0 / num_layers
    
    layer_names = {
        0: "background",
        num_layers - 1: "foreground"
    }
    
    for i in range(num_layers):
        # Layer depth range
        depth_min = i * depth_step - overlap_margin
        depth_max = (i + 1) * depth_step + overlap_margin
        
        # Clamp to valid range
        depth_min = max(0.0, depth_min)
        depth_max = min(1.0, depth_max)
        
        # Create mask for this layer
        mask = (depth >= depth_min) & (depth < depth_max)
        
        # Handle edge case for final layer (include depth == 1.0)
        if i == num_layers - 1:
            mask = mask | (depth >= depth_max - overlap_margin)
        
        # Extract RGB for this layer
        rgb_layer = np.zeros_like(rgb)
        rgb_layer[mask] = rgb[mask]
        
        # Create alpha mask
        alpha = mask.astype(np.uint8) * 255
        
        # Representative depth for this plane
        layer_depth = (depth_min + depth_max) / 2
        
        # Determine layer name
        if i in layer_names:
            name = layer_names[i]
        else:
            name = f"midground_{i}"
        
        layers.append((rgb_layer, alpha, layer_depth, (depth_min, depth_max), i, name))
        
        pixel_count = np.sum(mask)
        pct = 100.0 * pixel_count / (H * W)
        print(f"    Layer {i} ({name}): depth [{depth_min:.3f}, {depth_max:.3f}], "
              f"pixels: {pixel_count:,} ({pct:.1f}%)")
    
    return layers


# ============================================================================
# STEP 7: GENERATE INPAINTING MASKS
# ============================================================================

def generate_inpainting_masks(
    layers: List[Tuple[np.ndarray, np.ndarray, float, Tuple[float, float], int, str]],
    occlusion_mask: np.ndarray,
    depth_edges: np.ndarray,
    dilate_edges: int = 3
) -> List[np.ndarray]:
    """
    Generate per-layer inpainting masks by intersecting disocclusion with layers.
    
    GEOMETRIC REASONING:
    Inpainting is needed where:
    1. Content was hidden (disoccluded) by the virtual camera motion
    2. The hidden region belongs to this specific depth layer
    
    We also expand masks slightly around depth edges, as these
    regions are most likely to exhibit artifacts in 3D viewing.
    
    Args:
        layers: List of layer tuples from segment_depth_layers()
        occlusion_mask: (H, W) Global disocclusion mask
        depth_edges: (H, W) Depth discontinuity edges
        dilate_edges: Pixels to dilate around edges
        
    Returns:
        List of (H, W) uint8 inpainting masks, one per layer
    """
    print("  Generating per-layer inpainting masks...")
    
    inpaint_masks = []
    
    # Optionally include depth edges in inpainting regions
    # These areas often need refinement for clean parallax
    if dilate_edges > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (dilate_edges * 2 + 1, dilate_edges * 2 + 1))
        expanded_edges = cv2.dilate(depth_edges, kernel)
    else:
        expanded_edges = depth_edges
    
    for rgb_layer, alpha, depth_val, depth_range, idx, name in layers:
        # Layer mask (where this layer has content)
        layer_mask = alpha > 0
        
        # Inpainting needed = disoccluded AND belongs to this layer
        # Also include edge regions that might need cleanup
        inpaint = np.zeros_like(occlusion_mask)
        
        # Primary: disoccluded regions within this layer's depth range
        # For background layers, disocclusions are particularly important
        inpaint[layer_mask & (occlusion_mask > 0)] = 255
        
        # Secondary: edge regions (optional, often helps with layer blending)
        # Only apply to pixels already in this layer
        # inpaint[layer_mask & (expanded_edges > 0)] = 255
        
        # Extend inpainting into layer boundaries
        # This helps fill gaps at layer transitions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inpaint = cv2.dilate(inpaint, kernel)
        
        # But constrain to reasonable vicinity of the layer
        # Don't let inpainting spread too far
        layer_vicinity = cv2.dilate(alpha, kernel)
        inpaint = inpaint & layer_vicinity
        
        inpaint_masks.append(inpaint)
        
        inpaint_count = np.sum(inpaint > 0)
        print(f"    Layer {idx} ({name}): {inpaint_count:,} pixels need inpainting")
    
    return inpaint_masks


# ============================================================================
# STEP 8: BUILD MPI DATA STRUCTURE
# ============================================================================

def build_mpi(
    rgb: np.ndarray,
    depth: np.ndarray,
    shift_x: float = 5.0,
    shift_y: float = 0.0,
    num_layers: int = 3,
    depth_scale: float = 1.0
) -> MPIResult:
    """
    Main pipeline: build complete MPI from RGB + depth.
    
    This function orchestrates the full pipeline:
    1. Detect disoccluded regions via virtual camera motion
    2. Compute depth edges for layer guidance
    3. Segment into depth layers
    4. Generate inpainting masks per layer
    5. Package into MPI data structure
    
    Args:
        rgb: (H, W, 3) Source image
        depth: (H, W) Normalized depth [0, 1]
        shift_x: Virtual camera shift in pixels
        shift_y: Vertical shift (usually 0)
        num_layers: Number of MPI planes
        depth_scale: Depth influence on parallax
        
    Returns:
        MPIResult containing all layers and metadata
    """
    print("\n" + "=" * 60)
    print("Building Multi-Plane Image (MPI)")
    print("=" * 60)
    
    H, W = depth.shape
    
    # Step 3-4: Detect disocclusions
    occlusion_mask, projected_rgb, valid_mask = detect_disocclusions(
        rgb, depth, shift_x, shift_y, depth_scale
    )
    
    # Step 5: Compute depth edges
    depth_edges = compute_depth_edges(depth)
    
    # Step 6: Segment into layers
    layer_data = segment_depth_layers(rgb, depth, num_layers)
    
    # Step 7: Generate inpainting masks
    inpaint_masks = generate_inpainting_masks(
        layer_data, occlusion_mask, depth_edges
    )
    
    # Step 8: Build MPI layers
    mpi_layers = []
    for i, ((rgb_layer, alpha, depth_val, depth_range, idx, name), inpaint_mask) \
            in enumerate(zip(layer_data, inpaint_masks)):
        
        layer = MPILayer(
            rgb=rgb_layer,
            alpha=alpha,
            depth=depth_val,
            depth_range=depth_range,
            inpaint_mask=inpaint_mask,
            layer_index=idx,
            name=name
        )
        mpi_layers.append(layer)
    
    result = MPIResult(
        layers=mpi_layers,
        occlusion_mask=occlusion_mask,
        depth_edges=depth_edges,
        source_image=rgb,
        source_depth=depth,
        width=W,
        height=H,
        camera_shift=shift_x
    )
    
    print(f"\n  MPI built successfully: {len(mpi_layers)} layers")
    return result


# ============================================================================
# STEP 9: DEBUG VISUALIZATIONS
# ============================================================================

def visualize_mpi(mpi: MPIResult, output_dir: str):
    """
    Generate comprehensive debug visualizations.
    
    Creates a multi-panel figure showing:
    - Original RGB image
    - Depth map
    - Disocclusion mask
    - Depth edges
    - Each MPI layer (RGB + alpha + inpaint mask)
    
    Args:
        mpi: MPIResult to visualize
        output_dir: Directory to save visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not available, skipping visualizations")
        return
    
    print("\n  Generating debug visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========== Overview Figure ==========
    n_layers = len(mpi.layers)
    fig = plt.figure(figsize=(16, 4 + 4 * ((n_layers + 2) // 3)))
    
    # Create grid: first row for source data, subsequent rows for layers
    n_rows = 2 + (n_layers + 2) // 3
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    # Row 1: Source image, depth, occlusion
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(mpi.source_image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mpi.source_depth, cmap='plasma')
    ax2.set_title("Depth Map")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mpi.occlusion_mask, cmap='Reds')
    ax3.set_title(f"Disocclusion Mask (shift={mpi.camera_shift}px)")
    ax3.axis('off')
    
    # Row 2: Depth edges, combined layers preview
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(mpi.depth_edges, cmap='gray')
    ax4.set_title("Depth Edges")
    ax4.axis('off')
    
    # Layer alpha composite
    ax5 = fig.add_subplot(gs[1, 1])
    layer_vis = np.zeros((mpi.height, mpi.width, 3), dtype=np.float32)
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))[:, :3]
    for i, layer in enumerate(mpi.layers):
        mask = layer.alpha > 0
        layer_vis[mask] = colors[i]
    ax5.imshow(layer_vis)
    ax5.set_title(f"Layer Segmentation ({n_layers} layers)")
    ax5.axis('off')
    
    # Combined inpainting regions
    ax6 = fig.add_subplot(gs[1, 2])
    combined_inpaint = np.zeros((mpi.height, mpi.width), dtype=np.uint8)
    for layer in mpi.layers:
        combined_inpaint = np.maximum(combined_inpaint, layer.inpaint_mask)
    ax6.imshow(combined_inpaint, cmap='hot')
    ax6.set_title("Combined Inpainting Regions")
    ax6.axis('off')
    
    # Subsequent rows: individual layers
    for i, layer in enumerate(mpi.layers):
        row = 2 + i // 3
        col = i % 3
        
        if row < n_rows:
            ax = fig.add_subplot(gs[row, col])
            
            # Create RGBA visualization
            rgba = np.zeros((mpi.height, mpi.width, 4), dtype=np.uint8)
            rgba[:, :, :3] = layer.rgb
            rgba[:, :, 3] = layer.alpha
            
            # Checkerboard background for transparency
            checker = np.zeros((mpi.height, mpi.width, 3), dtype=np.uint8)
            checker_size = 16
            for cy in range(0, mpi.height, checker_size):
                for cx in range(0, mpi.width, checker_size):
                    if ((cy // checker_size) + (cx // checker_size)) % 2 == 0:
                        checker[cy:cy+checker_size, cx:cx+checker_size] = [200, 200, 200]
                    else:
                        checker[cy:cy+checker_size, cx:cx+checker_size] = [240, 240, 240]
            
            # Composite
            alpha_float = layer.alpha.astype(np.float32) / 255.0
            composite = (layer.rgb.astype(np.float32) * alpha_float[:, :, np.newaxis] + 
                        checker.astype(np.float32) * (1 - alpha_float[:, :, np.newaxis]))
            
            # Overlay inpainting regions in red
            inpaint_overlay = composite.copy()
            inpaint_mask = layer.inpaint_mask > 0
            inpaint_overlay[inpaint_mask, 0] = 255  # Red channel
            inpaint_overlay[inpaint_mask, 1] = np.minimum(inpaint_overlay[inpaint_mask, 1] * 0.3, 255)
            inpaint_overlay[inpaint_mask, 2] = np.minimum(inpaint_overlay[inpaint_mask, 2] * 0.3, 255)
            
            ax.imshow(inpaint_overlay.astype(np.uint8))
            ax.set_title(f"Layer {i}: {layer.name}\n"
                        f"depth={layer.depth:.2f}, inpaint={np.sum(layer.inpaint_mask > 0):,}px")
            ax.axis('off')
    
    plt.suptitle("MPI Debug Visualization", fontsize=14, fontweight='bold')
    
    # Save overview
    overview_path = output_path / "mpi_overview.png"
    plt.savefig(str(overview_path), dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"    Saved: {overview_path}")
    
    # ========== Individual Layer Figures ==========
    for i, layer in enumerate(mpi.layers):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB
        axes[0].imshow(layer.rgb)
        axes[0].set_title(f"Layer {i} RGB ({layer.name})")
        axes[0].axis('off')
        
        # Alpha
        axes[1].imshow(layer.alpha, cmap='gray')
        axes[1].set_title(f"Alpha Mask")
        axes[1].axis('off')
        
        # Inpainting
        axes[2].imshow(layer.inpaint_mask, cmap='hot')
        axes[2].set_title(f"Inpainting Mask ({np.sum(layer.inpaint_mask > 0):,}px)")
        axes[2].axis('off')
        
        plt.suptitle(f"MPI Layer {i}: {layer.name} (depth={layer.depth:.3f})", 
                     fontsize=12, fontweight='bold')
        
        layer_path = output_path / f"layer_{i:02d}_detail.png"
        plt.savefig(str(layer_path), dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"    Saved: {layer_path}")


# ============================================================================
# FILE I/O
# ============================================================================

def save_mpi(mpi: MPIResult, output_dir: str, base_name: str = "mpi"):
    """
    Save MPI layers and metadata to disk.
    
    Output structure:
        output_dir/
            {base_name}_metadata.json   - Layer info and dimensions
            layer_00_rgb.png            - Background RGB
            layer_00_alpha.png          - Background alpha
            layer_00_inpaint.png        - Background inpainting mask
            layer_01_rgb.png            - Midground RGB
            ...
            occlusion_mask.png          - Global disocclusion mask
            depth_edges.png             - Depth discontinuities
    
    Args:
        mpi: MPIResult to save
        output_dir: Output directory
        base_name: Prefix for output files
    """
    print(f"\n  Saving MPI to: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata JSON
    metadata = mpi.to_dict()
    metadata["files"] = {
        "layers": [],
        "occlusion_mask": "occlusion_mask.png",
        "depth_edges": "depth_edges.png"
    }
    
    # Save each layer
    for layer in mpi.layers:
        layer_prefix = f"layer_{layer.layer_index:02d}"
        
        # RGB (convert to BGR for OpenCV)
        rgb_path = output_path / f"{layer_prefix}_rgb.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(layer.rgb, cv2.COLOR_RGB2BGR))
        
        # Alpha
        alpha_path = output_path / f"{layer_prefix}_alpha.png"
        cv2.imwrite(str(alpha_path), layer.alpha)
        
        # Inpainting mask
        inpaint_path = output_path / f"{layer_prefix}_inpaint.png"
        cv2.imwrite(str(inpaint_path), layer.inpaint_mask)
        
        # RGBA combined (for easy viewing)
        rgba = np.dstack([
            cv2.cvtColor(layer.rgb, cv2.COLOR_RGB2BGR),
            layer.alpha
        ])
        rgba_path = output_path / f"{layer_prefix}_rgba.png"
        cv2.imwrite(str(rgba_path), rgba)
        
        metadata["files"]["layers"].append({
            "index": layer.layer_index,
            "name": layer.name,
            "rgb": f"{layer_prefix}_rgb.png",
            "alpha": f"{layer_prefix}_alpha.png",
            "inpaint": f"{layer_prefix}_inpaint.png",
            "rgba": f"{layer_prefix}_rgba.png"
        })
        
        print(f"    Layer {layer.layer_index}: saved 4 files")
    
    # Save global masks
    occlusion_path = output_path / "occlusion_mask.png"
    cv2.imwrite(str(occlusion_path), mpi.occlusion_mask)
    
    edges_path = output_path / "depth_edges.png"
    cv2.imwrite(str(edges_path), mpi.depth_edges)
    
    # Save metadata
    meta_path = output_path / f"{base_name}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    Metadata: {meta_path}")
    print(f"  [OK] Saved {len(mpi.layers) * 4 + 2} files + metadata")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_single(
    image_path: str,
    depth_path: str,
    output_dir: str,
    shift_x: float = 5.0,
    num_layers: int = 3,
    debug: bool = False
) -> MPIResult:
    """
    Process a single image+depth pair.
    
    Args:
        image_path: Path to RGB image
        depth_path: Path to depth map
        output_dir: Output directory
        shift_x: Virtual camera shift
        num_layers: Number of MPI layers
        debug: Generate debug visualizations
        
    Returns:
        MPIResult
    """
    print("\n" + "=" * 60)
    print(f"Processing: {Path(image_path).name}")
    print("=" * 60)
    
    # Step 1: Load inputs
    print("\nStep 1: Loading inputs")
    rgb = load_image(image_path)
    depth = load_depth(depth_path, rgb.shape[:2])
    
    # Steps 2-8: Build MPI
    print("\nSteps 2-8: Building MPI")
    mpi = build_mpi(rgb, depth, shift_x=shift_x, num_layers=num_layers)
    
    # Save outputs
    base_name = Path(image_path).stem
    save_mpi(mpi, output_dir, base_name)
    
    # Step 9: Debug visualizations
    if debug:
        print("\nStep 9: Debug visualizations")
        visualize_mpi(mpi, output_dir)
    
    print("\n" + "=" * 60)
    print("[OK] Processing complete!")
    print("=" * 60)
    
    return mpi


def process_batch(
    images_dir: str,
    depths_dir: str,
    output_dir: str,
    shift_x: float = 5.0,
    num_layers: int = 3,
    debug: bool = False
):
    """
    Batch process a directory of images.
    
    Expects:
        images_dir/image.jpg
        depths_dir/image_depth.png  OR  depths_dir/image.png
    
    Args:
        images_dir: Directory containing RGB images
        depths_dir: Directory containing depth maps
        output_dir: Output root directory
        shift_x: Virtual camera shift
        num_layers: Number of MPI layers
        debug: Generate debug visualizations
    """
    images_path = Path(images_dir)
    depths_path = Path(depths_dir)
    output_path = Path(output_dir)
    
    # Find image files (exclude depth maps and masks)
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    exclude_suffixes = ['_depth', '_mask', '_alpha', '_inpaint', '_rgba']
    
    image_files = []
    for f in images_path.iterdir():
        if f.suffix.lower() not in extensions:
            continue
        # Skip files that look like depth maps or masks
        stem_lower = f.stem.lower()
        if any(stem_lower.endswith(suffix) for suffix in exclude_suffixes):
            continue
        image_files.append(f)
    
    print(f"\nFound {len(image_files)} images to process")
    print("=" * 60)
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_file.name}")
        
        # Find corresponding depth map
        depth_file = None
        for suffix in ['_depth.png', '_depth.jpg', '.png', '.jpg']:
            candidate = depths_path / f"{img_file.stem}{suffix}"
            if candidate.exists():
                depth_file = candidate
                break
        
        if depth_file is None:
            print(f"  [ERROR] No depth map found for {img_file.name}")
            continue
        
        # Create output subdirectory
        img_output_dir = output_path / img_file.stem
        
        try:
            process_single(
                str(img_file),
                str(depth_file),
                str(img_output_dir),
                shift_x=shift_x,
                num_layers=num_layers,
                debug=debug
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
    
    print(f"\n{'=' * 60}")
    print(f"[OK] Batch processing complete!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Multi-Plane Images (MPI) from RGB + depth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with depth map
  python generate_mpi.py image.jpg depth.png --output ./mpi_output/
  
  # Batch process directories
  python generate_mpi.py ./images/ ./depths/ --output ./mpi_output/ --batch
  
  # Custom camera shift (larger = more disocclusion)
  python generate_mpi.py image.jpg depth.png --shift 10 --output ./mpi_output/
  
  # More layers for smoother parallax
  python generate_mpi.py image.jpg depth.png --layers 5 --output ./mpi_output/
  
  # Generate debug visualizations
  python generate_mpi.py image.jpg depth.png --output ./mpi_output/ --debug

Pipeline Overview:
  1. Load RGB image and depth map
  2. Simulate virtual camera motion (lateral shift)
  3. Z-buffer visibility resolution
  4. Detect disoccluded (hidden) regions
  5. Compute depth discontinuity edges
  6. Segment into depth layers (foreground/midground/background)
  7. Generate inpainting masks per layer
  8. Save MPI data structure (PNG + JSON)
  9. Optional: debug visualizations

Philosophy:
  We discover WHERE content MUST be missing (geometry).
  We don't guess WHAT is missing (that comes later with inpainting).
"""
    )
    
    parser.add_argument(
        "input",
        help="Input image path (or directory with --batch)"
    )
    
    parser.add_argument(
        "depth",
        help="Depth map path (or directory with --batch)"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
    )
    
    parser.add_argument(
        "--shift", "-s",
        type=float,
        default=5.0,
        help="Virtual camera shift in pixels (default: 5)"
    )
    
    parser.add_argument(
        "--layers", "-l",
        type=int,
        default=3,
        help="Number of depth layers (default: 3)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch process directories"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Generate debug visualizations"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Multi-Plane Image (MPI) Generator")
    print("RVR Studio - Geometry First, Hallucination Later")
    print("=" * 60)
    print(f"Camera shift: {args.shift} pixels")
    print(f"Layers: {args.layers}")
    print(f"Debug: {args.debug}")
    
    if args.batch:
        process_batch(
            args.input,
            args.depth,
            args.output,
            shift_x=args.shift,
            num_layers=args.layers,
            debug=args.debug
        )
    else:
        process_single(
            args.input,
            args.depth,
            args.output,
            shift_x=args.shift,
            num_layers=args.layers,
            debug=args.debug
        )


if __name__ == "__main__":
    main()
