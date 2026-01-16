# BLADE / SOTA Humanoid Mesh Generation Pipeline

Generate interactive 3D humanoid meshes from single RGB images using state-of-the-art models.

## Overview

This pipeline assumes **every image contains a humanoid subject** and uses SOTA methods to generate textured 3D meshes:

```
RGB Image → Human Detection → Mesh Recovery (HMR2/SMPL) → Texture Projection → glTF Export
```

## SOTA Methods Supported

| Method | Description | Best For |
|--------|-------------|----------|
| **hmr2** | HMR2 (4D-Humans) - SMPL mesh from single image | Best human pose/shape recovery |
| **transformers** | HuggingFace pipeline (pose + depth) | Good fallback, easy setup |
| **depth** | Depth-based mesh with human detection | General purpose, works without ML models |
| **poisson** | Point cloud + Poisson reconstruction | Smooth, watertight surfaces |

## Quick Start

### 1. Single Image → Humanoid Mesh

```bash
cd tools

# Using HMR2 (4D-Humans) - Best for humans
python generate_mesh.py ../assets/photo.jpg ../assets/meshes/photo.glb --method hmr2

# Using depth-based method (no HMR2 required)
python generate_mesh.py ../assets/photo.jpg ../assets/meshes/photo.glb --method depth

# With custom depth map
python generate_mesh.py ../assets/photo.jpg ../assets/meshes/photo.glb \
    --method depth \
    --depth ../assets/photo_depth.png

# Using HuggingFace transformers
python generate_mesh.py ../assets/photo.jpg ../assets/meshes/photo.glb --method transformers
```

### 2. Batch Processing (Multiple Images)

```bash
# Batch with HMR2 (all images assumed to contain humans)
python generate_mesh.py --batch ../assets/ ../assets/meshes/ --method hmr2

# Batch with depth-to-mesh
python generate_mesh.py --batch ../assets/ ../assets/meshes/ \
    --method depth \
    --depth ./depths/

# Parallel batch processing (4 workers)
python generate_mesh.py --batch ../assets/ ../assets/meshes/ \
    --method depth \
    --depth ./depths/ \
    --parallel 4
```

### 3. Complete Pipeline (Depth + Mesh)

```bash
# Step 1: Generate depth maps
python generate_depth_map.py --batch ../assets/ ./depths/ --method depth-pro

# Step 2: Generate humanoid meshes
python generate_mesh.py --batch ../assets/ ../assets/meshes/ \
    --method depth \
    --depth ./depths/ \
    --resolution 256 \
    --parallel 4
```

## Installation

### Core Dependencies

```bash
pip install -r requirements.txt
```

### For HMR2 (4D-Humans) - Best Results (Optional)

**Note:** HMR2 requires complex dependencies. If installation fails, the pipeline 
automatically falls back to depth-based mesh generation which works well for most cases.

```bash
# Step 1: Install chumpy first (may require workaround)
pip install chumpy

# If chumpy fails, try:
pip install git+https://github.com/mattloper/chumpy.git

# Step 2: Install HMR2
pip install git+https://github.com/shubham-goel/4D-Humans.git

# Step 3: Download SMPL models (required for HMR2)
# Get from https://smpl.is.tue.mpg.de/
# Set environment variable:
# Windows: set SMPL_MODEL_DIR=C:\path\to\smpl\models
# Linux/Mac: export SMPL_MODEL_DIR=/path/to/smpl/models
```

**If HMR2 installation fails:** Don't worry! The `--method hmr2` will automatically 
fallback to depth-based mesh generation, which produces good results.

### For HuggingFace Transformers

```bash
pip install transformers accelerate
```

## Methods Explained

### HMR2 (Human Mesh Recovery 2.0) - `--method hmr2`

State-of-the-art human mesh recovery from 4D-Humans project:
- Outputs SMPL mesh with pose and shape parameters
- Best accuracy for human body reconstruction
- Requires SMPL model files

```bash
python generate_mesh.py photo.jpg mesh.glb --method hmr2
```

### Depth-Based with Human Focus - `--method depth`

Enhanced depth-to-mesh with automatic human detection:
- Detects human in image using DETR
- Focuses depth on human subject
- Auto-generates depth if not provided
- Good fallback when HMR2 unavailable

```bash
python generate_mesh.py photo.jpg mesh.glb --method depth --depth photo_depth.png
```

### Transformers Pipeline - `--method transformers`

HuggingFace-based pipeline:
- Uses Depth Anything V2 for depth estimation
- Human detection via DETR
- Easy setup, no SMPL required

```bash
python generate_mesh.py photo.jpg mesh.glb --method transformers
```

### Poisson Reconstruction - `--method poisson`

Point cloud to mesh via Poisson surface reconstruction:
- Produces smooth, watertight meshes
- Good for organic shapes
- Requires Open3D

```bash
python generate_mesh.py photo.jpg mesh.glb --method poisson --depth photo_depth.png
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--method` | depth | Mesh generation method |
| `--depth` | auto | Depth map path (auto-generated if not provided) |
| `--resolution` | 256 | Mesh grid resolution |
| `--depth-scale` | 0.3 | Z displacement scale |
| `--simplify` | 0.5 | Mesh simplification ratio |
| `--parallel` | 1 | Batch workers (1 = sequential) |
| `--no-human-focus` | false | Disable human detection |

## Batch Processing Details

### Input/Output Structure

```
Input:                      Output:
./images/                   ./meshes/
  ├── photo1.jpg              ├── photo1.glb
  ├── photo2.jpg              ├── photo1.json
  ├── photo3.png              ├── photo2.glb
  └── ...                     ├── photo2.json
                              └── ...
./depths/ (optional)
  ├── photo1_depth.png
  ├── photo2_depth.png
  └── ...
```

### Parallel Processing

For large batches, use parallel workers:

```bash
# 4 parallel workers (good for depth method)
python generate_mesh.py --batch ./images/ ./meshes/ --method depth --parallel 4

# Note: HMR2 doesn't parallelize well due to GPU memory
python generate_mesh.py --batch ./images/ ./meshes/ --method hmr2  # Sequential
```

### Auto Depth Generation

If no depth map is provided, the pipeline automatically generates one using Depth Anything V2:

```bash
# No --depth flag = auto-generate
python generate_mesh.py photo.jpg mesh.glb --method depth
```

## Human Detection

All methods (except `--no-human-focus`) automatically:

1. **Detect humans** using DETR object detection
2. **Crop to human** with padding
3. **Focus depth** on human region
4. **Suppress background** depth

This ensures the mesh focuses on the humanoid subject.

## Output Files

For each processed image:

| File | Description |
|------|-------------|
| `image.glb` | Binary glTF mesh (Three.js compatible) |
| `image.json` | Metadata (vertices, faces, method, etc.) |

## Viewer Integration

Generated meshes work automatically with RVR Studio Browser:

1. Place `.glb` files in `assets/meshes/`
2. Enable "🎭 3D Mesh Mode (BLADE)" in viewer
3. Use orbit controls: drag to rotate, scroll to zoom

## Troubleshooting

### "HMR2 not available"

```bash
pip install git+https://github.com/shubham-goel/4D-Humans.git
```

### "SMPL faces not found"

Download SMPL models from https://smpl.is.tue.mpg.de/ and set:
```bash
export SMPL_MODEL_DIR=/path/to/smpl/models
```

### "No human detected"

- Ensure image clearly shows a person
- Try `--no-human-focus` to process without detection
- Check image orientation

### GPU Memory Issues

- Use `--parallel 1` for HMR2
- Reduce `--resolution` (try 128 instead of 256)
- Process in smaller batches

## Model Comparison

| Method | Quality | Speed | GPU Memory | Setup Complexity |
|--------|---------|-------|------------|------------------|
| hmr2 | ★★★★★ | Slow | High | Complex (SMPL) |
| transformers | ★★★★☆ | Medium | Medium | Easy |
| depth | ★★★☆☆ | Fast | Low | Easy |
| poisson | ★★★★☆ | Medium | Medium | Easy |

## Future Enhancements

- [ ] SiTH integration (texture hallucination)
- [ ] PSHuman integration (photorealistic)
- [ ] SMPLer-X for whole-body (hands, face)
- [ ] BLADE direct integration
- [ ] Real-time mesh streaming
