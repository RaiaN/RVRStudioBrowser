#!/usr/bin/env python3
"""
generate_depth_map.py
Generate depth maps from images using various methods.

This script provides multiple approaches to generate depth maps:
1. Apple Depth Pro - Best quality, sharp edges, metric depth (RECOMMENDED)
2. MiDaS (DPT) - Deep learning based, good quality
3. Depth Anything - Fast and accurate
4. Simple edge-based estimation - Fast but lower quality
5. Gradient generators - For testing/manual editing

Requirements:
    pip install torch torchvision opencv-python numpy pillow

For Depth Pro:
    pip install git+https://github.com/apple/ml-depth-pro.git

Usage:
    python generate_depth_map.py input.jpg output_depth.png --method depth-pro
    python generate_depth_map.py input.jpg output_depth.png --method midas
    python generate_depth_map.py input.jpg output_depth.png --method depth-anything
    python generate_depth_map.py --batch ./images/ ./depths/ --method depth-pro

Author: RVR Studio Team
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Optional imports for advanced methods
TORCH_AVAILABLE = False

try:
    import torch
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch/cv2 not available. Only basic methods will work.")


def generate_depth_pro(image_path: str, output_path: str) -> None:
    """
    Generate depth map using Apple's Depth Pro model via Hugging Face.
    Best quality with sharp edges and metric depth.
    Auto-downloads the model on first use.
    
    Args:
        image_path: Path to input image
        output_path: Path to save depth map
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install with: pip install torch torchvision")
    
    print("Loading Apple Depth Pro model (via Hugging Face)...")
    print("(Model will be downloaded automatically on first use ~1.8GB)")
    
    try:
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
    except ImportError:
        print("\n" + "="*60)
        print("transformers library not installed or outdated!")
        print("="*60)
        print("\nInstall/upgrade with:")
        print("  pip install --upgrade transformers")
        print("\nDepth Pro requires transformers >= 4.48.0 (Feb 2025+)")
        print("="*60 + "\n")
        raise RuntimeError("transformers not available or outdated")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor from Hugging Face
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
    processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    
    print(f"Processing: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=original_size[::-1],  # (height, width)
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
    # Convert to numpy
    depth_np = prediction.cpu().numpy()
    
    # Normalize to 0-255 (invert so closer = brighter)
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
    depth_normalized = 1.0 - depth_normalized  # Invert: close=white, far=black
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Save
    depth_image = Image.fromarray(depth_uint8, mode='L')
    depth_image.save(output_path)
    
    print(f"✓ Saved depth map to: {output_path}")


def generate_depth_anything(image_path: str, output_path: str, model_size: str = "base") -> None:
    """
    Generate depth map using Depth Anything model.
    Fast and accurate, good for batch processing.
    
    Args:
        image_path: Path to input image
        output_path: Path to save depth map
        model_size: "small", "base", or "large"
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install with: pip install torch torchvision")
    
    print(f"Loading Depth Anything model ({model_size})...")
    
    # Load from HuggingFace
    from transformers import pipeline
    
    pipe = pipeline(
        task="depth-estimation",
        model=f"LiheYoung/depth-anything-{model_size}-hf",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print(f"Processing: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Run inference
    result = pipe(image)
    depth = result["depth"]
    
    # Convert to numpy array
    depth_np = np.array(depth)
    
    # Normalize to 0-255 (invert so closer = brighter)
    depth_normalized = depth_np.astype(np.float32)
    depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min() + 1e-8)
    depth_normalized = 1.0 - depth_normalized  # Invert
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Save
    depth_image = Image.fromarray(depth_uint8, mode='L')
    depth_image.save(output_path)
    
    print(f"✓ Saved depth map to: {output_path}")


def generate_depth_midas(image_path: str, output_path: str, model_type: str = "DPT_Large") -> None:
    """
    Generate depth map using MiDaS (Depth Prediction Transformer).
    
    Args:
        image_path: Path to input image
        output_path: Path to save depth map
        model_type: MiDaS model variant ("DPT_Large", "DPT_Hybrid", "MiDaS_small")
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for MiDaS. Install with: pip install torch torchvision")
    
    print(f"Loading MiDaS model: {model_type}")
    
    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    # Load and process image
    print(f"Processing: {image_path}")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(img).to(device)
    
    # Inference
    with torch.no_grad():
        prediction = midas(input_batch)
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy and normalize
    depth = prediction.cpu().numpy()
    
    # Normalize to 0-255 (invert so closer = brighter)
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    depth_normalized = 1.0 - depth_normalized  # Invert
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Save
    depth_image = Image.fromarray(depth_uint8, mode='L')
    depth_image.save(output_path)
    
    print(f"✓ Saved depth map to: {output_path}")


def generate_depth_edges(image_path: str, output_path: str) -> None:
    """
    Generate a simple depth map based on edge detection and blur.
    This is a fast but rough approximation.
    
    Args:
        image_path: Path to input image
        output_path: Path to save depth map
    """
    print(f"Processing with edge method: {image_path}")
    
    # Load image
    img = Image.open(image_path).convert('L')
    
    # Edge detection
    edges = img.filter(ImageFilter.FIND_EDGES)
    
    # Blur to create depth-like gradient
    blurred = img.filter(ImageFilter.GaussianBlur(radius=20))
    
    # Combine edges and blur
    edges_np = np.array(edges, dtype=np.float32)
    blurred_np = np.array(blurred, dtype=np.float32)
    
    # Invert edges (edges usually indicate closer objects)
    edges_inverted = 255 - edges_np
    
    # Combine with weights
    combined = (blurred_np * 0.7 + edges_inverted * 0.3)
    
    # Normalize
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    combined = (combined * 255).astype(np.uint8)
    
    # Apply additional blur for smoother depth
    depth_image = Image.fromarray(combined, mode='L')
    depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=10))
    
    depth_image.save(output_path)
    print(f"✓ Saved depth map to: {output_path}")


def generate_depth_gradient(image_path: str, output_path: str, direction: str = "vertical") -> None:
    """
    Generate a simple gradient depth map.
    Useful as a base for manual editing.
    
    Args:
        image_path: Path to input image (used for dimensions)
        output_path: Path to save depth map
        direction: "vertical", "horizontal", or "radial"
    """
    print(f"Generating gradient depth map: {direction}")
    
    # Get dimensions from input image
    img = Image.open(image_path)
    width, height = img.size
    
    if direction == "vertical":
        # Top = close (white), Bottom = far (black)
        gradient = np.linspace(255, 50, height, dtype=np.uint8)
        depth = np.tile(gradient.reshape(-1, 1), (1, width))
        
    elif direction == "horizontal":
        # Left = far, Right = close
        gradient = np.linspace(50, 255, width, dtype=np.uint8)
        depth = np.tile(gradient, (height, 1))
        
    elif direction == "radial":
        # Center = close, edges = far
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        depth = (1 - distance / max_dist) * 205 + 50
        depth = depth.astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    depth_image = Image.fromarray(depth, mode='L')
    depth_image.save(output_path)
    print(f"✓ Saved gradient depth map to: {output_path}")


def batch_process(input_dir: str, output_dir: str, method: str = "depth-pro", model_arg: str = None) -> None:
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save depth maps
        method: Depth estimation method
        model_arg: Model-specific argument
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    
    print(f"Found {len(images)} images to process with method: {method}")
    print("="*50)
    
    for i, img_file in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img_file.name}")
        output_file = output_path / f"{img_file.stem}_depth.png"
        
        try:
            if method == "depth-pro":
                generate_depth_pro(str(img_file), str(output_file))
            elif method == "depth-anything":
                generate_depth_anything(str(img_file), str(output_file), model_arg or "base")
            elif method == "midas":
                generate_depth_midas(str(img_file), str(output_file), model_arg or "DPT_Large")
            elif method == "edges":
                generate_depth_edges(str(img_file), str(output_file))
            elif method in ["vertical", "horizontal", "radial"]:
                generate_depth_gradient(str(img_file), str(output_file), method)
            else:
                print(f"Unknown method: {method}")
                return
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*50)
    print(f"✓ Batch processing complete! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps from images for RVR Studio 3D Photo Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Best quality (Apple Depth Pro - recommended)
  python generate_depth_map.py photo.jpg photo_depth.png --method depth-pro
  
  # Fast and accurate (Depth Anything)
  python generate_depth_map.py photo.jpg photo_depth.png --method depth-anything
  
  # Classic MiDaS
  python generate_depth_map.py photo.jpg photo_depth.png --method midas
  
  # Quick edge-based (no ML)
  python generate_depth_map.py photo.jpg photo_depth.png --method edges
  
  # Batch process folder
  python generate_depth_map.py --batch ./photos/ ./depths/ --method depth-pro
  
  # Generate test gradient
  python generate_depth_map.py photo.jpg depth.png --method radial

Model Quality Comparison:
  depth-pro      ★★★★★  Sharp edges, metric depth, best quality
  depth-anything ★★★★☆  Fast, accurate, good all-rounder
  midas          ★★★☆☆  Established, reliable
  edges          ★★☆☆☆  Fast, no ML required
  gradients      ★☆☆☆☆  For testing only
"""
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="Input image path or directory (for batch mode)"
    )
    
    parser.add_argument(
        "output",
        nargs="?",
        help="Output depth map path or directory (for batch mode)"
    )
    
    parser.add_argument(
        "--method",
        choices=["depth-pro", "depth-anything", "midas", "edges", "vertical", "horizontal", "radial"],
        default="depth-pro",
        help="Depth estimation method (default: depth-pro)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in input directory"
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="Model variant: MiDaS (DPT_Large/DPT_Hybrid/MiDaS_small), Depth Anything (small/base/large)"
    )
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        return
    
    if args.batch:
        batch_process(args.input, args.output, args.method, args.model)
    else:
        if not args.output:
            print("Error: Output path required")
            return
            
        if args.method == "depth-pro":
            generate_depth_pro(args.input, args.output)
        elif args.method == "depth-anything":
            generate_depth_anything(args.input, args.output, args.model or "base")
        elif args.method == "midas":
            generate_depth_midas(args.input, args.output, args.model or "DPT_Large")
        elif args.method == "edges":
            generate_depth_edges(args.input, args.output)
        else:
            generate_depth_gradient(args.input, args.output, args.method)


if __name__ == "__main__":
    main()
