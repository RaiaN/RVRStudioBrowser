#!/usr/bin/env python3
"""
generate_segmentation.py
Generate segmentation masks for 3D photo rendering.

MODES:
  - foreground: Binary mask (WHITE=subject, BLACK=background) for Hybrid mode
  - multilayer: Multiple depth-ordered layers for MPI rendering

Models:
  - SAM (Segment Anything Model) - Meta's foundation model
  - HQ-SAM - Higher quality masks, better edges (2024/2025 SOTA)
  - depth: Use depth map to separate foreground/background (fast, no ML)

Requirements:
    pip install torch torchvision transformers pillow numpy opencv-python

Usage:
    # Simple foreground/background mask (for Hybrid mode) - RECOMMENDED
    python generate_segmentation.py photo.jpg mask.png --mode foreground
    
    # Use depth map for FG/BG separation (fast, no ML model needed)
    python generate_segmentation.py photo.jpg mask.png --mode foreground --method depth --depth depth.png
    
    # Batch process directory
    python generate_segmentation.py ./images/ ./masks/ --mode foreground
    
    # Multi-layer segmentation (for MPI)
    python generate_segmentation.py ./images/ ./masks/ --mode multilayer

Author: RVR Studio Team
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Check for dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Install with: pip install torch")

try:
    from transformers import (
        SamModel, 
        SamProcessor,
        pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv not available. Install with: pip install opencv-python")


# =============================================================================
# FOREGROUND/BACKGROUND DETECTION (for Hybrid mode)
# =============================================================================

def generate_foreground_mask_from_depth(
    image_path: str,
    depth_path: str,
    threshold: float = 0.5,
    blur_size: int = 5,
    morph_size: int = 7
) -> np.ndarray:
    """
    Generate foreground mask from depth map.
    Uses depth threshold: closer objects (darker in standard depth) = foreground.
    
    For depth maps where BLACK = near, WHITE = far:
    - Foreground = darker pixels (closer to camera)
    - Background = lighter pixels (further from camera)
    
    Args:
        image_path: Path to source image (for dimensions)
        depth_path: Path to depth map
        threshold: Depth threshold (0-1), pixels darker than this are foreground
        blur_size: Gaussian blur kernel size for smoothing
        morph_size: Morphological operation kernel size
    
    Returns:
        Binary mask (255 = foreground, 0 = background)
    """
    print(f"  Generating foreground mask from depth map...")
    print(f"    Threshold: {threshold:.2f} (pixels darker = foreground)")
    
    # Load depth
    depth = np.array(Image.open(depth_path).convert('L'), dtype=np.float32) / 255.0
    
    # In standard depth maps: BLACK (0) = near, WHITE (1) = far
    # So foreground = values < threshold
    # But we want WHITE = foreground in output mask
    foreground_mask = (depth < threshold).astype(np.uint8) * 255
    
    # Smooth and clean up the mask
    if CV2_AVAILABLE and blur_size > 0:
        # Gaussian blur to smooth edges
        foreground_mask = cv2.GaussianBlur(foreground_mask, (blur_size, blur_size), 0)
        
        # Re-threshold after blur
        _, foreground_mask = cv2.threshold(foreground_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        if morph_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
            # Close small holes
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
            # Remove small noise
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
    fg_percent = 100.0 * np.sum(foreground_mask > 0) / foreground_mask.size
    print(f"    Foreground coverage: {fg_percent:.1f}%")
    
    return foreground_mask


def generate_foreground_mask_from_sam(
    image_path: str,
    depth_path: Optional[str] = None,
    model_size: str = "base",
    method: str = "sam"
) -> np.ndarray:
    """
    Generate foreground mask using SAM.
    Strategy: Find the most salient/central object as foreground.
    
    Args:
        image_path: Path to source image
        depth_path: Optional depth map for refinement
        model_size: SAM model size
        method: "sam" or "hq-sam"
    
    Returns:
        Binary mask (255 = foreground, 0 = background)
    """
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("Install requirements: pip install torch transformers")
    
    # Model mapping
    if method == "hq-sam":
        model_name = "syscv/hq-sam-vit-base"
    else:
        models = {
            "base": "facebook/sam-vit-base",
            "large": "facebook/sam-vit-large",
            "huge": "facebook/sam-vit-huge",
        }
        model_name = models.get(model_size, models["base"])
    
    print(f"  Loading {method.upper()}: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    try:
        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(device)
    except Exception as e:
        if method == "hq-sam":
            print(f"  HQ-SAM not available, falling back to SAM")
            return generate_foreground_mask_from_sam(image_path, depth_path, model_size, "sam")
        raise
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Load optional depth for refinement
    depth = None
    if depth_path:
        depth = np.array(Image.open(depth_path).convert('L'), dtype=np.float32) / 255.0
        print("  Using depth map for foreground detection")
    
    # Strategy: Sample center region more densely (subjects usually in center)
    print("  Finding main subject...")
    
    # Sample points - focus on center
    center_x, center_y = width // 2, height // 2
    sample_points = [
        # Center points (most likely to hit subject)
        (center_x, center_y),
        (center_x - width//6, center_y),
        (center_x + width//6, center_y),
        (center_x, center_y - height//6),
        (center_x, center_y + height//6),
        # Rule of thirds points
        (width//3, height//3),
        (2*width//3, height//3),
        (width//3, 2*height//3),
        (2*width//3, 2*height//3),
    ]
    
    # If we have depth, also sample from the nearest (darkest) regions
    if depth is not None:
        # Find closest points (smallest depth values)
        flat_depth = depth.flatten()
        closest_indices = np.argsort(flat_depth)[:20]  # 20 closest points
        for idx in closest_indices[::4]:  # Sample every 4th
            y, x = divmod(idx, width)
            sample_points.append((x, y))
    
    best_mask = None
    best_score = -1
    
    for px, py in sample_points:
        try:
            inputs = processor(
                image,
                input_points=[[[px, py]]],
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            scores = outputs.iou_scores.cpu().numpy().flatten()
            best_idx = int(scores.argmax())
            
            # Extract mask
            if hasattr(masks, 'numpy'):
                masks_np = masks.numpy() if not isinstance(masks, np.ndarray) else masks
            else:
                masks_np = np.array(masks)
            
            if masks_np.ndim == 4:
                mask = masks_np[0, best_idx]
            elif masks_np.ndim == 3:
                mask = masks_np[best_idx]
            else:
                mask = masks_np
            
            # Score this mask
            # Good foreground: reasonable size, centered, close (if depth available)
            area_ratio = mask.sum() / (height * width)
            
            # Skip too small or too large
            if area_ratio < 0.01 or area_ratio > 0.9:
                continue
            
            # Compute mask center
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            mask_cx = xs.mean()
            mask_cy = ys.mean()
            
            # Distance from image center (normalized)
            dist_from_center = np.sqrt(
                ((mask_cx - center_x) / width)**2 + 
                ((mask_cy - center_y) / height)**2
            )
            
            # Depth score (if available) - prefer closer objects
            depth_score = 0
            if depth is not None:
                mask_depth = depth[mask].mean()
                depth_score = 1.0 - mask_depth  # Lower depth = higher score
            
            # Combined score
            iou = float(scores[best_idx]) if best_idx < len(scores) else 0.5
            score = (
                iou * 0.3 +                      # SAM confidence
                (1.0 - dist_from_center) * 0.3 + # Centered-ness
                depth_score * 0.3 +              # Closeness
                min(area_ratio * 2, 0.3)         # Reasonable size (not too small)
            )
            
            if score > best_score:
                best_score = score
                best_mask = mask
                
        except Exception as e:
            continue
    
    if best_mask is None:
        print("  Warning: No good foreground found, using center fallback")
        # Fallback: use center region
        best_mask = np.zeros((height, width), dtype=bool)
        cx, cy = width // 2, height // 2
        r = min(width, height) // 3
        y, x = np.ogrid[:height, :width]
        best_mask = ((x - cx)**2 + (y - cy)**2 <= r**2)
    
    # Convert to uint8
    foreground_mask = best_mask.astype(np.uint8) * 255
    
    # Clean up with morphology if available
    if CV2_AVAILABLE:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        # Slight blur for smoother edges
        foreground_mask = cv2.GaussianBlur(foreground_mask, (5, 5), 0)
        _, foreground_mask = cv2.threshold(foreground_mask, 127, 255, cv2.THRESH_BINARY)
    
    fg_percent = 100.0 * np.sum(foreground_mask > 0) / foreground_mask.size
    print(f"  Foreground coverage: {fg_percent:.1f}%")
    
    return foreground_mask


def segment_with_sam(
    image_path: str, 
    depth_path: Optional[str] = None,
    model_size: str = "base",
    grid_points: int = 8
) -> dict:
    """
    Segment using SAM (Segment Anything Model) via HuggingFace.
    
    Args:
        image_path: Path to source image
        depth_path: Optional depth map for layer ordering
        model_size: "base", "large", or "huge"
        grid_points: Grid density for automatic prompts (higher = more masks)
    """
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("Install requirements: pip install torch transformers")
    
    # Model mapping
    models = {
        "base": "facebook/sam-vit-base",
        "large": "facebook/sam-vit-large",
        "huge": "facebook/sam-vit-huge",
    }
    model_name = models.get(model_size, models["base"])
    
    print(f"Loading SAM: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    image_np = np.array(image)
    
    # Load optional depth
    depth = None
    if depth_path:
        depth = np.array(Image.open(depth_path).convert('L'), dtype=np.float32) / 255.0
        print("Using depth for layer ordering")
    
    print(f"Generating masks with {grid_points}x{grid_points} grid...")
    
    # Generate masks from grid of points
    all_masks = []
    step_x = width // grid_points
    step_y = height // grid_points
    
    for y in range(step_y // 2, height, step_y):
        for x in range(step_x // 2, width, step_x):
            try:
                inputs = processor(
                    image, 
                    input_points=[[[x, y]]],
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )[0]
                
                # Get best mask
                scores = outputs.iou_scores.cpu().numpy().flatten()
                best_idx = int(scores.argmax())
                
                # Handle different mask tensor shapes
                if hasattr(masks, 'numpy'):
                    masks_np = masks.numpy() if not isinstance(masks, np.ndarray) else masks
                else:
                    masks_np = np.array(masks)
                
                if masks_np.ndim == 4:  # (batch, num_masks, H, W)
                    mask = masks_np[0, best_idx]
                elif masks_np.ndim == 3:  # (num_masks, H, W)
                    mask = masks_np[best_idx]
                else:
                    mask = masks_np
                
                if mask.sum() > 100:  # Min size
                    score = float(scores[best_idx]) if best_idx < len(scores) else 0.5
                    all_masks.append({
                        'mask': mask.astype(bool),
                        'score': score,
                        'area': int(mask.sum())
                    })
            except Exception as e:
                # Skip this point if there's an error
                continue
    
    print(f"Found {len(all_masks)} candidate masks")
    
    # Merge overlapping masks
    masks = _deduplicate_masks(all_masks)
    print(f"After deduplication: {len(masks)} unique masks")
    
    # Calculate depth for each mask
    for m in masks:
        m['depth'] = _estimate_depth(m['mask'], depth, height, width)
    
    # Sort by depth (background first)
    masks = sorted(masks, key=lambda x: x['depth'])
    
    return _build_result(image_path, masks, width, height, "sam")


def segment_with_hqsam(
    image_path: str,
    depth_path: Optional[str] = None,
    grid_points: int = 8
) -> dict:
    """
    Segment using HQ-SAM (High-Quality SAM) via HuggingFace.
    Better mask quality, especially for fine details and edges.
    
    HQ-SAM paper: https://arxiv.org/abs/2306.01567
    """
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("Install requirements: pip install torch transformers")
    
    model_name = "syscv/hq-sam-vit-base"  # or syscv/hq-sam-vit-large, syscv/hq-sam-vit-huge
    
    print(f"Loading HQ-SAM: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"HQ-SAM not available ({e}), falling back to SAM")
        return segment_with_sam(image_path, depth_path, "base", grid_points)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Load optional depth
    depth = None
    if depth_path:
        depth = np.array(Image.open(depth_path).convert('L'), dtype=np.float32) / 255.0
        print("Using depth for layer ordering")
    
    print(f"Generating high-quality masks...")
    
    # Generate masks
    all_masks = []
    step_x = width // grid_points
    step_y = height // grid_points
    
    for y in range(step_y // 2, height, step_y):
        for x in range(step_x // 2, width, step_x):
            try:
                inputs = processor(
                    image,
                    input_points=[[[x, y]]],
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )[0]
                
                scores = outputs.iou_scores.cpu().numpy().flatten()
                best_idx = int(scores.argmax())
                
                # Handle different mask tensor shapes
                if hasattr(masks, 'numpy'):
                    masks_np = masks.numpy() if not isinstance(masks, np.ndarray) else masks
                else:
                    masks_np = np.array(masks)
                
                if masks_np.ndim == 4:
                    mask = masks_np[0, best_idx]
                elif masks_np.ndim == 3:
                    mask = masks_np[best_idx]
                else:
                    mask = masks_np
                
                if mask.sum() > 100:
                    score = float(scores[best_idx]) if best_idx < len(scores) else 0.5
                    all_masks.append({
                        'mask': mask.astype(bool),
                        'score': score,
                        'area': int(mask.sum())
                    })
            except Exception as e:
                continue
    
    print(f"Found {len(all_masks)} candidate masks")
    
    masks = _deduplicate_masks(all_masks)
    print(f"After deduplication: {len(masks)} unique masks")
    
    for m in masks:
        m['depth'] = _estimate_depth(m['mask'], depth, height, width)
    
    masks = sorted(masks, key=lambda x: x['depth'])
    
    return _build_result(image_path, masks, width, height, "hq-sam")


def _deduplicate_masks(masks: list, iou_threshold: float = 0.5) -> list:
    """Remove duplicate/overlapping masks, keep larger ones."""
    if not masks:
        return []
    
    # Sort by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    kept = []
    for m in masks:
        # Check overlap with already kept masks
        dominated = False
        for k in kept:
            intersection = np.logical_and(m['mask'], k['mask']).sum()
            union = np.logical_or(m['mask'], k['mask']).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                dominated = True
                break
        
        if not dominated:
            kept.append(m)
    
    return kept


def _estimate_depth(mask: np.ndarray, depth: Optional[np.ndarray], height: int, width: int) -> float:
    """Estimate depth for a mask (using depth map or heuristics)."""
    if depth is not None:
        return float(depth[mask].mean()) if mask.any() else 0.5
    
    # Heuristic: larger masks + higher position = background
    y_coords = np.where(mask)[0]
    if len(y_coords) == 0:
        return 0.5
    
    avg_y = y_coords.mean() / height
    area_ratio = mask.sum() / (height * width)
    
    # Lower depth = background, higher = foreground
    return (1 - area_ratio * 0.5) * (avg_y * 0.5 + 0.5)


def _build_result(image_path: str, masks: list, width: int, height: int, method: str) -> dict:
    """Build the result dictionary with layer images."""
    # Create layer mask with evenly-spaced values (0, 127, 255 for 3 layers, etc.)
    # This makes it easier for the viewer to distinguish layers
    layer_mask = np.zeros((height, width), dtype=np.uint8)
    num_masks = len(masks)
    for i, m in enumerate(masks):
        # Space values evenly across 0-255 range
        if num_masks > 1:
            layer_value = int(i * 255 / (num_masks - 1))
        else:
            layer_value = 0
        layer_mask[m['mask']] = layer_value
    
    # Create layer images
    image_rgba = np.array(Image.open(image_path).convert('RGBA'))
    layers = []
    
    for i, m in enumerate(masks):
        layer_img = image_rgba.copy()
        layer_img[:, :, 3] = m['mask'].astype(np.uint8) * 255
        
        layers.append({
            'id': i,
            'image': Image.fromarray(layer_img, 'RGBA'),
            'depth': m['depth'],
            'pixel_count': int(m['area']),
        })
    
    return {
        'mask': layer_mask,
        'layers': layers,
        'metadata': {
            'num_layers': len(masks),
            'width': width,
            'height': height,
            'method': method
        }
    }


def save_results(result: dict, output_path: str, split_layers: bool = False):
    """Save segmentation results."""
    out_path = Path(output_path)
    
    if split_layers:
        out_path.mkdir(parents=True, exist_ok=True)
        
        for layer in result['layers']:
            layer_file = out_path / f"layer_{layer['id']:02d}.png"
            layer['image'].save(str(layer_file))
            print(f"  Saved: {layer_file}")
        
        # Save metadata
        meta = result['metadata'].copy()
        meta['layers'] = [
            {'id': l['id'], 'depth': l['depth'], 'pixels': l['pixel_count'], 'file': f"layer_{l['id']:02d}.png"}
            for l in result['layers']
        ]
        
        with open(out_path / "layers.json", 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved: {out_path / 'layers.json'}")
    
    else:
        # Save mask
        mask = result['mask']
        num_layers = max(1, result['metadata']['num_layers'] - 1)
        gray_mask = (mask * (255 // num_layers)).astype(np.uint8)
        
        if out_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            Image.fromarray(gray_mask, 'L').save(str(out_path))
            print(f"  Saved: {out_path}")
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(gray_mask, 'L').save(str(out_path / 'mask.png'))
            print(f"  Saved: {out_path / 'mask.png'}")


def get_image_files(input_path: str) -> list:
    """Get list of image files from path (file or directory)."""
    path = Path(input_path)
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    elif path.is_dir():
        files = []
        for ext in extensions:
            files.extend(path.glob(f'*{ext}'))
            files.extend(path.glob(f'*{ext.upper()}'))
        return sorted(set(files))
    return []


def process_single_foreground(
    image_path: Path,
    output_path: Path,
    method: str,
    model_size: str,
    depth_path: Optional[str],
    threshold: float
):
    """Process a single image in FOREGROUND mode (binary mask)."""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"Mode: FOREGROUND (binary mask)")
    print(f"Method: {method}")
    print(f"{'='*60}")
    
    # Generate foreground mask
    if method == "depth":
        if not depth_path:
            raise ValueError("Depth-based method requires depth map")
        mask = generate_foreground_mask_from_depth(
            str(image_path), 
            depth_path,
            threshold=threshold
        )
    else:
        # SAM or HQ-SAM
        mask = generate_foreground_mask_from_sam(
            str(image_path),
            depth_path,
            model_size,
            method
        )
    
    # Save mask
    output_path = Path(output_path)
    if output_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        output_path = output_path / f"{image_path.stem}_mask.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, 'L').save(str(output_path))
    print(f"  ✓ Saved: {output_path}")
    
    return mask


def process_single_multilayer(
    image_path: Path,
    output_path: Path,
    method: str,
    model_size: str,
    grid: int,
    depth_path: Optional[str],
    split_layers: bool
):
    """Process a single image in MULTILAYER mode."""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"Mode: MULTILAYER")
    print(f"{'='*60}")
    
    # Run segmentation
    if method == "hq-sam":
        result = segment_with_hqsam(str(image_path), depth_path, grid)
    elif method == "sam":
        result = segment_with_sam(str(image_path), depth_path, model_size, grid)
    else:
        # Depth-based multilayer
        raise ValueError("Multilayer mode requires SAM or HQ-SAM method")
    
    print(f"Found {result['metadata']['num_layers']} layers")
    
    # Save results
    save_results(result, str(output_path), split_layers)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks for 3D photo rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FOREGROUND MODE (for Hybrid rendering) - RECOMMENDED
  python generate_segmentation.py photo.jpg mask.png --mode foreground
  
  # Use depth map for FG/BG (fast, no ML model)
  python generate_segmentation.py photo.jpg mask.png --mode foreground --method depth --depth depth.png
  
  # Batch process with depth maps
  python generate_segmentation.py ./images/ ./masks/ --mode foreground --method depth --depth ./depths/
  
  # Use SAM for foreground detection (slower but more accurate)
  python generate_segmentation.py photo.jpg mask.png --mode foreground --method sam
  
  # MULTILAYER MODE (for MPI rendering)
  python generate_segmentation.py ./images/ ./masks/ --mode multilayer

Modes:
  foreground  Binary mask (WHITE=subject, BLACK=background) for Hybrid mode
  multilayer  Multiple depth-ordered layers for MPI rendering

Methods:
  depth    Use depth map threshold (fast, no ML) - requires --depth
  sam      SAM (Segment Anything Model) - Meta's foundation model
  hq-sam   HQ-SAM - Higher quality masks, better edges
"""
    )
    
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output mask/directory")
    
    parser.add_argument(
        "--mode",
        choices=["foreground", "multilayer"],
        default="foreground",
        help="Mask type: 'foreground' for Hybrid mode, 'multilayer' for MPI (default: foreground)"
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["depth", "sam", "hq-sam"],
        default="depth",
        help="Detection method: 'depth' (fast), 'sam', 'hq-sam' (default: depth)"
    )
    
    parser.add_argument(
        "--model-size", "-s",
        choices=["base", "large", "huge"],
        default="base",
        help="SAM model size (default: base)"
    )
    
    parser.add_argument(
        "--grid", "-g",
        type=int,
        default=8,
        help="Grid density for SAM prompts (default: 8)"
    )
    
    parser.add_argument(
        "--depth", "-d",
        help="Depth map file or directory (required for --method depth)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Depth threshold for foreground (0-1, lower=more foreground, default: 0.5)"
    )
    
    parser.add_argument(
        "--split-layers",
        action="store_true",
        help="Save separate layer images (multilayer mode only)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.method == "depth" and not args.depth:
        parser.error("--method depth requires --depth <depth_map>")
    
    if args.mode == "foreground" and args.split_layers:
        print("Warning: --split-layers ignored in foreground mode")
    
    # Get input files
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_files = get_image_files(args.input)
    
    if not image_files:
        print(f"❌ No images found in: {args.input}")
        return
    
    print(f"\n{'='*60}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Method: {args.method}")
    if args.method in ['sam', 'hq-sam']:
        print(f"Model: {args.model_size}")
    print(f"Found {len(image_files)} image(s)")
    print(f"{'='*60}")
    
    def find_depth_file(img_file: Path) -> Optional[str]:
        """Find matching depth file for an image."""
        if not args.depth:
            return None
        depth_path = Path(args.depth)
        if depth_path.is_file():
            return str(depth_path)
        if depth_path.is_dir():
            # Look for matching depth file
            for pattern in [f"{img_file.stem}_depth", img_file.stem]:
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate = depth_path / f"{pattern}{ext}"
                    if candidate.exists():
                        return str(candidate)
        return None
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end="")
        
        # Find depth file
        depth_file = find_depth_file(img_file)
        
        # Determine output path
        if len(image_files) == 1 and output_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img_output = output_path
        elif args.mode == "multilayer" and args.split_layers:
            img_output = output_path / img_file.stem
        else:
            img_output = output_path / f"{img_file.stem}_mask.png"
        
        try:
            if args.mode == "foreground":
                process_single_foreground(
                    img_file,
                    img_output,
                    args.method,
                    args.model_size,
                    depth_file,
                    args.threshold
                )
            else:  # multilayer
                process_single_multilayer(
                    img_file,
                    img_output,
                    args.method,
                    args.model_size,
                    args.grid,
                    depth_file,
                    args.split_layers
                )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Done! Processed {len(image_files)} image(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
