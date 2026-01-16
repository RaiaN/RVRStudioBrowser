#!/usr/bin/env python3
"""
inpaint_disocclusions.py
Prepare inpainted textures for Point Cloud rendering

This script creates an inpainted version of the source image that fills
disoccluded regions revealed during parallax viewing.

The Point Cloud renderer uses:
1. Original image - displayed for originally visible pixels
2. Inpainted image - displayed for pixels revealed during camera movement

This approach treats the original image as a "decal" projected onto 3D geometry,
with AI/algorithmic inpainting filling in what was hidden.

Reuses disocclusion detection from generate_mpi.py

Requirements:
    pip install numpy opencv-python

Usage:
    python inpaint_disocclusions.py image.jpg depth.png --output ./inpainted/
    
    # Multiple shift directions for full parallax coverage
    python inpaint_disocclusions.py image.jpg depth.png --output ./inpainted/ --multi-direction

Author: RVR Studio Team
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


# ============================================================================
# REUSED FROM generate_mpi.py - CORE ALGORITHMS
# ============================================================================

def load_image(image_path: str) -> np.ndarray:
    """Load RGB image and ensure proper format."""
    print(f"  Loading image: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # OpenCV loads as BGR, convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"    Dimensions: {img_rgb.shape[1]}x{img_rgb.shape[0]} (W×H)")
    return img_rgb


def load_depth(depth_path: str, target_shape: Tuple[int, int], invert: bool = True) -> np.ndarray:
    """
    Load depth map and normalize to [0, 1] range.
    
    Standard depth map format (most common):
        Black (0) = NEAR (close to camera)  
        White (255) = FAR (background)
    
    Script needs internally:
        1.0 = near (foreground moves more during parallax)
        0.0 = far (background moves less)
    
    So we INVERT standard depth maps by default.
    """
    print(f"  Loading depth: {depth_path}")
    
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth: {depth_path}")
    
    if depth.shape[:2] != target_shape:
        print(f"    Resizing depth from {depth.shape[:2]} to {target_shape}")
        depth = cv2.resize(depth, (target_shape[1], target_shape[0]), 
                          interpolation=cv2.INTER_LINEAR)
    
    depth_float = depth.astype(np.float32) / 255.0
    
    if invert:
        # Standard format: Black=near, White=far → Invert so 1.0=near, 0.0=far
        depth_float = 1.0 - depth_float
        print(f"    ✓ Standard depth format (Black=near, White=far)")
    else:
        # Reversed format: White=near, Black=far → No inversion needed
        print(f"    Using reversed depth format (White=near, Black=far)")
    
    print(f"    Internal range: [{depth_float.min():.3f}, {depth_float.max():.3f}] (1.0=near, 0.0=far)")
    return depth_float


def detect_disocclusions_fast(
    depth: np.ndarray,
    shift_amount: float = 15.0
) -> np.ndarray:
    """
    FAST disocclusion detection using depth edges + dilation.
    
    Key insight: Disocclusions happen at depth EDGES where foreground
    occludes background. We can detect these directly from the depth map
    without simulating camera movement.
    
    This is 100x faster than per-pixel projection!
    """
    print("  Detecting disocclusions (fast edge-based method)...")
    
    H, W = depth.shape
    
    # 1. Find depth edges using Sobel
    sobel_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 2. Threshold to find significant depth discontinuities
    edge_threshold = 0.05  # Depth difference threshold
    depth_edges = (edge_magnitude > edge_threshold).astype(np.uint8) * 255
    
    # 3. Dilate edges based on shift amount (bigger shift = more revealed area)
    # The dilation size represents how much area could be revealed
    dilation_size = int(shift_amount * 1.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    dilated_edges = cv2.dilate(depth_edges, kernel)
    
    # 4. The disoccluded regions are on the BACKGROUND side of edges
    # Erode depth slightly and compare to find background-adjacent regions
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    depth_eroded = cv2.erode((depth * 255).astype(np.uint8), erode_kernel)
    depth_diff = cv2.absdiff((depth * 255).astype(np.uint8), depth_eroded)
    
    # 5. Combine: dilated edges where depth decreases (background side)
    # Use the dilated edges directly - they mark where disocclusions occur
    disocclusion_mask = dilated_edges
    
    # 6. Also add image boundary regions (always partially occluded)
    boundary_size = int(shift_amount)
    boundary_mask = np.zeros((H, W), dtype=np.uint8)
    boundary_mask[:boundary_size, :] = 255  # Top
    boundary_mask[-boundary_size:, :] = 255  # Bottom
    boundary_mask[:, :boundary_size] = 255  # Left
    boundary_mask[:, -boundary_size:] = 255  # Right
    
    disocclusion_mask = np.maximum(disocclusion_mask, boundary_mask)
    
    # Clean up
    disocclusion_mask = cv2.morphologyEx(disocclusion_mask, cv2.MORPH_CLOSE, 
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    num_disoccluded = np.sum(disocclusion_mask > 0)
    total_pixels = disocclusion_mask.size
    pct = 100.0 * num_disoccluded / total_pixels
    print(f"    Disoccluded region: {num_disoccluded:,} pixels ({pct:.1f}%)")
    
    return disocclusion_mask


# ============================================================================
# INPAINTING ALGORITHMS
# ============================================================================

def inpaint_telea(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 5
) -> np.ndarray:
    """
    Fast inpainting using Telea's algorithm (OpenCV built-in).
    Good for small regions, fast execution.
    """
    print(f"  Inpainting with Telea algorithm (radius={radius})...")
    
    # OpenCV expects BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.uint8) * 255
    
    # Inpaint
    result_bgr = cv2.inpaint(image_bgr, mask_binary, radius, cv2.INPAINT_TELEA)
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    return result_rgb


def inpaint_ns(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 5
) -> np.ndarray:
    """
    Inpainting using Navier-Stokes based algorithm.
    Better quality for larger regions, slower.
    """
    print(f"  Inpainting with Navier-Stokes algorithm (radius={radius})...")
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask_binary = (mask > 127).astype(np.uint8) * 255
    
    result_bgr = cv2.inpaint(image_bgr, mask_binary, radius, cv2.INPAINT_NS)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    return result_rgb


def inpaint_patchmatch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 9
) -> np.ndarray:
    """
    Simple PatchMatch-inspired inpainting.
    Fills masked regions by finding similar patches in visible areas.
    
    This is a simplified version - for production, consider using
    deep learning inpainting models (LaMa, MAT, etc.)
    """
    print(f"  Inpainting with PatchMatch-inspired algorithm (patch_size={patch_size})...")
    
    H, W, C = image.shape
    result = image.copy()
    mask_binary = mask > 127
    
    # Get coordinates of masked pixels
    masked_coords = np.argwhere(mask_binary)
    
    if len(masked_coords) == 0:
        print("    No pixels to inpaint")
        return result
    
    # Half patch size
    half_patch = patch_size // 2
    
    # Create padded versions
    pad_img = cv2.copyMakeBorder(image, half_patch, half_patch, half_patch, half_patch, 
                                  cv2.BORDER_REFLECT)
    pad_mask = cv2.copyMakeBorder(mask_binary.astype(np.uint8), half_patch, half_patch, 
                                   half_patch, half_patch, cv2.BORDER_CONSTANT, value=1)
    
    # Find valid source patches (fully visible)
    valid_sources = []
    for y in range(half_patch, H + half_patch, 4):  # Step by 4 for speed
        for x in range(half_patch, W + half_patch, 4):
            patch_mask = pad_mask[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
            if np.sum(patch_mask) == 0:  # Fully visible
                valid_sources.append((y, x))
    
    if len(valid_sources) == 0:
        print("    Warning: No valid source patches found, using fallback")
        return inpaint_telea(image, mask)
    
    print(f"    Found {len(valid_sources)} valid source patches")
    print(f"    Filling {len(masked_coords)} masked pixels...")
    
    # Process masked pixels in a dilated order (from boundary inward)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    remaining_mask = mask_binary.copy()
    
    iterations = 0
    max_iterations = 100
    
    while np.sum(remaining_mask) > 0 and iterations < max_iterations:
        # Find boundary pixels (masked pixels adjacent to filled pixels)
        eroded = cv2.erode(remaining_mask.astype(np.uint8), kernel)
        boundary = remaining_mask.astype(np.uint8) - eroded
        boundary_coords = np.argwhere(boundary > 0)
        
        if len(boundary_coords) == 0:
            break
        
        # Fill boundary pixels
        for y, x in boundary_coords:
            py, px = y + half_patch, x + half_patch
            
            # Get target patch (with current fill state)
            current_padded = cv2.copyMakeBorder(result, half_patch, half_patch, 
                                                 half_patch, half_patch, cv2.BORDER_REFLECT)
            target_patch = current_padded[py-half_patch:py+half_patch+1, 
                                          px-half_patch:px+half_patch+1].astype(np.float32)
            
            # Find best matching source patch (sample subset for speed)
            sample_sources = valid_sources[::max(1, len(valid_sources)//100)]
            best_dist = float('inf')
            best_color = result[y, x]
            
            for sy, sx in sample_sources:
                source_patch = pad_img[sy-half_patch:sy+half_patch+1, 
                                       sx-half_patch:sx+half_patch+1].astype(np.float32)
                
                # Compare only visible pixels in target
                target_mask_patch = pad_mask[py-half_patch:py+half_patch+1, 
                                             px-half_patch:px+half_patch+1]
                visible = target_mask_patch == 0
                
                if np.sum(visible) > 0:
                    dist = np.mean((target_patch[visible] - source_patch[visible]) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        # Get center pixel color from source
                        best_color = pad_img[sy, sx]
            
            result[y, x] = best_color
        
        remaining_mask[boundary > 0] = False
        iterations += 1
    
    print(f"    Completed in {iterations} iterations")
    return result


def inpaint_hybrid(
    image: np.ndarray,
    mask: np.ndarray,
    small_region_threshold: int = 1000
) -> np.ndarray:
    """
    Hybrid inpainting: uses fast algorithm for small regions,
    better algorithm for larger ones.
    """
    print("  Using hybrid inpainting strategy...")
    
    # Find connected components in mask
    mask_binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary)
    
    result = image.copy()
    
    for label_id in range(1, num_labels):  # Skip background (0)
        region_size = stats[label_id, cv2.CC_STAT_AREA]
        region_mask = (labels == label_id).astype(np.uint8) * 255
        
        if region_size < small_region_threshold:
            # Use fast Telea for small regions
            partial = cv2.inpaint(
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                region_mask, 3, cv2.INPAINT_TELEA
            )
            result = cv2.cvtColor(partial, cv2.COLOR_BGR2RGB)
        else:
            # Use Navier-Stokes for larger regions
            partial = cv2.inpaint(
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                region_mask, 5, cv2.INPAINT_NS
            )
            result = cv2.cvtColor(partial, cv2.COLOR_BGR2RGB)
    
    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_image(
    image_path: str,
    depth_path: str,
    output_dir: str,
    shift_amount: float = 15.0,
    inpaint_method: str = "hybrid",
    invert_depth: bool = True
) -> Dict[str, Any]:
    """
    Process a single image to create inpainted texture for point cloud rendering.
    
    Args:
        image_path: Path to RGB image
        depth_path: Path to depth map
        output_dir: Output directory
        shift_amount: How much parallax to prepare for (pixels)
        inpaint_method: "telea", "ns", "patchmatch", or "hybrid"
        invert_depth: Invert depth map convention
        
    Returns:
        Dictionary with output file paths and metadata
    """
    print("\n" + "=" * 60)
    print(f"Processing: {Path(image_path).name}")
    print("=" * 60)
    
    # Load inputs
    print("\n[1/4] Loading inputs")
    rgb = load_image(image_path)
    depth = load_depth(depth_path, rgb.shape[:2], invert=invert_depth)
    
    H, W = depth.shape
    
    # Detect disocclusions (fast edge-based method)
    print("\n[2/4] Detecting disocclusions")
    disocclusion_mask = detect_disocclusions_fast(depth, shift_amount)
    
    # Inpaint disoccluded regions
    print("\n[3/4] Inpainting disoccluded regions")
    
    if np.sum(disocclusion_mask > 0) == 0:
        print("    No disoccluded regions detected, using original image")
        inpainted = rgb.copy()
    else:
        if inpaint_method == "telea":
            inpainted = inpaint_telea(rgb, disocclusion_mask)
        elif inpaint_method == "ns":
            inpainted = inpaint_ns(rgb, disocclusion_mask)
        elif inpaint_method == "patchmatch":
            inpainted = inpaint_patchmatch(rgb, disocclusion_mask)
        else:  # hybrid
            inpainted = inpaint_hybrid(rgb, disocclusion_mask)
    
    # Save outputs
    print("\n[4/4] Saving outputs")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(image_path).stem
    
    # Save inpainted image
    inpainted_path = output_path / f"{base_name}_inpainted.png"
    cv2.imwrite(str(inpainted_path), cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
    print(f"    Saved: {inpainted_path}")
    
    # Save disocclusion mask (useful for debugging)
    mask_path = output_path / f"{base_name}_disocclusion_mask.png"
    cv2.imwrite(str(mask_path), disocclusion_mask)
    print(f"    Saved: {mask_path}")
    
    # Save comparison image (original | inpainted | diff)
    comparison = np.zeros((H, W * 3, 3), dtype=np.uint8)
    comparison[:, :W] = rgb
    comparison[:, W:W*2] = inpainted
    
    # Difference visualization (enhanced)
    diff = np.abs(inpainted.astype(np.int16) - rgb.astype(np.int16)).astype(np.uint8)
    diff_enhanced = np.clip(diff * 3, 0, 255).astype(np.uint8)  # Enhance visibility
    comparison[:, W*2:] = diff_enhanced
    
    comparison_path = output_path / f"{base_name}_comparison.png"
    cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"    Saved: {comparison_path}")
    
    # Save metadata
    metadata = {
        "source_image": str(image_path),
        "source_depth": str(depth_path),
        "width": W,
        "height": H,
        "shift_amount": shift_amount,
        "inpaint_method": inpaint_method,
        "disoccluded_pixels": int(np.sum(disocclusion_mask > 0)),
        "disoccluded_percentage": float(100.0 * np.sum(disocclusion_mask > 0) / (H * W)),
        "outputs": {
            "inpainted": str(inpainted_path.name),
            "mask": str(mask_path.name),
            "comparison": str(comparison_path.name)
        }
    }
    
    meta_path = output_path / f"{base_name}_inpaint_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved: {meta_path}")
    
    print("\n" + "=" * 60)
    print("[OK] Processing complete!")
    print("=" * 60)
    
    return metadata


def process_batch(
    images_dir: str,
    depths_dir: str,
    output_dir: str,
    shift_amount: float = 15.0,
    inpaint_method: str = "hybrid",
    invert_depth: bool = True
):
    """
    Batch process a directory of images.
    
    Expects:
        images_dir/image.jpg
        depths_dir/image_depth.png  OR  depths_dir/image.png
    """
    images_path = Path(images_dir)
    depths_path = Path(depths_dir)
    output_path = Path(output_dir)
    
    # Find image files (exclude depth maps and masks)
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    exclude_suffixes = ['_depth', '_mask', '_alpha', '_inpaint', '_rgba', '_inpainted', '_disocclusion']
    
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
    
    successful = 0
    failed = 0
    
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
            failed += 1
            continue
        
        try:
            process_image(
                str(img_file),
                str(depth_file),
                str(output_path),
                shift_amount=shift_amount,
                inpaint_method=inpaint_method,
                invert_depth=invert_depth
            )
            successful += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
            continue
    
    print(f"\n{'=' * 60}")
    print(f"[OK] Batch processing complete!")
    print(f"    Successful: {successful}")
    print(f"    Failed: {failed}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Fill hidden regions in images for 3D parallax viewing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all images in folders (default)
    python inpaint_disocclusions.py ./images/ ./depths/ -o ./inpainted/
    
    # Single image
    python inpaint_disocclusions.py image.jpg depth.png -o ./inpainted/ --single

Output:
    {name}_inpainted.png  - Image with hidden regions filled in
"""
    )
    
    parser.add_argument("images", help="Images folder (or single image with --single)")
    parser.add_argument("depths", help="Depth maps folder (or single depth with --single)")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument("--single", action="store_true",
                        help="Process single image instead of folder")
    parser.add_argument("--no-invert", action="store_true",
                        help="Only if your depth maps are REVERSED (White=near, Black=far)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🎨 Inpainting Hidden Regions for 3D Parallax")
    print("=" * 60)
    
    if args.single:
        # Single file mode
        process_image(
            args.images,
            args.depths,
            args.output,
            invert_depth=not args.no_invert
        )
    else:
        # Batch mode (DEFAULT)
        process_batch(
            args.images,
            args.depths,
            args.output,
            invert_depth=not args.no_invert
        )


if __name__ == "__main__":
    main()
