#!/usr/bin/env python3
"""
generate_segmentation.py
Generate segmentation masks for multi-layer 3D photo rendering.

Models:
  - SAM (Segment Anything Model) - Meta's foundation model
  - HQ-SAM - Higher quality masks, better edges (2024/2025 SOTA)

Requirements:
    pip install torch torchvision transformers pillow numpy

Usage:
    # Single image
    python generate_segmentation.py photo.jpg mask.png
    
    # Directory of images (batch processing)
    python generate_segmentation.py ./images/ ./masks/
    
    # HQ-SAM for better quality
    python generate_segmentation.py ./images/ ./masks/ --method hq-sam
    
    # Split each image into layer images
    python generate_segmentation.py ./images/ ./output/ --split-layers

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


def process_single(
    image_path: Path,
    output_path: Path,
    method: str,
    model_size: str,
    grid: int,
    depth_path: Optional[str],
    split_layers: bool
):
    """Process a single image."""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")
    
    # Run segmentation
    if method == "hq-sam":
        result = segment_with_hqsam(str(image_path), depth_path, grid)
    else:
        result = segment_with_sam(str(image_path), depth_path, model_size, grid)
    
    print(f"Found {result['metadata']['num_layers']} layers")
    
    # Save results
    save_results(result, str(output_path), split_layers)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks using SAM or HQ-SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python generate_segmentation.py photo.jpg mask.png
  
  # Directory of images (batch processing)
  python generate_segmentation.py ./images/ ./masks/
  
  # HQ-SAM for better quality
  python generate_segmentation.py ./images/ ./masks/ --method hq-sam
  
  # Larger model
  python generate_segmentation.py ./images/ ./masks/ --model-size large
  
  # Split into layer images per image
  python generate_segmentation.py ./images/ ./output/ --split-layers

Methods:
  sam      SAM (Segment Anything Model) - Meta's foundation model
  hq-sam   HQ-SAM - Higher quality masks, better edges (2024 SOTA)
"""
    )
    
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output mask/directory")
    
    parser.add_argument(
        "--method", "-m",
        choices=["sam", "hq-sam"],
        default="sam",
        help="Segmentation method (default: sam)"
    )
    
    parser.add_argument(
        "--model-size", "-s",
        choices=["base", "large", "huge"],
        default="base",
        help="Model size (default: base)"
    )
    
    parser.add_argument(
        "--grid", "-g",
        type=int,
        default=8,
        help="Grid density for prompts (default: 8)"
    )
    
    parser.add_argument(
        "--depth", "-d",
        help="Optional depth map or directory for layer ordering"
    )
    
    parser.add_argument(
        "--split-layers",
        action="store_true",
        help="Save separate layer images"
    )
    
    args = parser.parse_args()
    
    # Get input files
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_files = get_image_files(args.input)
    
    if not image_files:
        print(f"❌ No images found in: {args.input}")
        return
    
    print(f"Method: {args.method}")
    print(f"Model: {args.model_size}")
    print(f"Found {len(image_files)} image(s)")
    
    # Single file mode
    if len(image_files) == 1 and input_path.is_file():
        process_single(
            image_files[0],
            output_path,
            args.method,
            args.model_size,
            args.grid,
            args.depth,
            args.split_layers
        )
    
    # Batch mode (directory)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]", end="")
            
            # Determine output path for this image
            if args.split_layers:
                # Create subdirectory for each image's layers
                img_output = output_path / img_file.stem
            else:
                # Just the mask file
                img_output = output_path / f"{img_file.stem}_mask.png"
            
            # Check for corresponding depth map
            depth_file = None
            if args.depth:
                depth_path = Path(args.depth)
                if depth_path.is_dir():
                    # Look for matching depth file
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate = depth_path / f"{img_file.stem}_depth{ext}"
                        if candidate.exists():
                            depth_file = str(candidate)
                            break
                        candidate = depth_path / f"{img_file.stem}{ext}"
                        if candidate.exists():
                            depth_file = str(candidate)
                            break
                elif depth_path.is_file():
                    depth_file = str(depth_path)
            
            try:
                process_single(
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
                continue
    
    print(f"\n{'='*60}")
    print(f"✓ Done! Processed {len(image_files)} image(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
