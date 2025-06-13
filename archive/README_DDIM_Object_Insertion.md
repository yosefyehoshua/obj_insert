# DDIM Object Insertion with Attention Switching

This repository implements a custom object insertion mechanism using DDIM inversion in the Stable Diffusion XL pipeline, with focus on manipulating the attention mechanism during the forward pass to blend foreground and background images along a defined mask.

## Overview

The pipeline implements the attention manipulation strategy described in `rules.py`:
- Invert both foreground and background images using DDIM inversion
- Save all intermediate noise latents (x_t) for both fg and bg
- Generate a binary mask for boundary regions between fg and bg
- During forward denoising, replace keys and queries at boundary regions with those from noise
- Preserve original keys/queries in non-masked regions to maintain fg/bg fidelity

## Key Features

- **Deterministic DDIM Inversion**: Uses eta=0 for reproducible results
- **Attention Switching**: Manipulates Q/K/V tensors in self-attention layers
- **Boundary-Aware Blending**: Only affects transition regions between objects
- **Soft Edge Blending**: Gaussian blur for smooth transitions
- **Multi-Layer Support**: Configurable target layers (up, mid, down blocks)

## Installation

1. Install dependencies:
```bash
pip install -r requirements_complete.txt
```

2. Ensure you have CUDA available for GPU acceleration.

## Usage

### Basic Demo

Run the complete demo with your own images:

```bash
python demo_ddim_object_insertion.py \
    --fg_image path/to/foreground.jpg \
    --bg_image path/to/background.jpg \
    --output_dir output \
    --prompt "a seamless composite image with natural lighting"
```

### Advanced Usage

```bash
python demo_ddim_object_insertion.py \
    --fg_image path/to/foreground.jpg \
    --bg_image path/to/background.jpg \
    --output_dir output \
    --prompt "a beautiful composite scene" \
    --fg_prompt "detailed object description" \
    --bg_prompt "background scene description" \
    --num_steps 50 \
    --guidance_scale 7.5 \
    --blend_strength 0.8 \
    --mask_method edge_detection
```

### Programmatic Usage

```python
from src.ddim_object_insertion_example import DDIMObjectInserter

# Initialize the inserter
inserter = DDIMObjectInserter(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    device="cuda"
)

# Perform object insertion
result = inserter.insert_object(
    fg_image_path="path/to/foreground.jpg",
    bg_image_path="path/to/background.jpg",
    output_prompt="a seamless composite image",
    blend_strength=0.8,
    num_inference_steps=50
)
```

## Architecture

### Core Components

1. **DDIMObjectInsertionAttnProcessor** (`src/attention_switching_processor.py`)
   - Custom attention processor that handles Q/K/V manipulation
   - Implements boundary mask blending
   - Supports timestep-aware noise latent access

2. **DDIMObjectInserter** (`src/ddim_object_insertion_example.py`)
   - Main orchestrator class
   - Handles DDIM inversion, mask creation, and generation
   - Provides high-level API for object insertion

3. **DDIM Inversion** (`src/ddim_inversion_obj_insert.py`)
   - Deterministic inversion implementation for SDXL
   - Supports dual text encoders and pooled embeddings
   - Captures intermediate noise latents

### Attention Switching Mechanism

The attention switching works by:

1. **Noise Latent Storage**: Store intermediate latents from DDIM inversion
2. **Feature Generation**: Generate Q/K/V from noise latents at each timestep
3. **Boundary Mask Application**: Blend original and noise features based on mask
4. **Soft Blending**: Apply Gaussian blur for smooth transitions

```python
# Core blending formula
blended_q = original_q * (1 - mask * blend_strength) + noise_q * (mask * blend_strength)
```

## Parameters

### Key Parameters

- **blend_strength** (0.0-1.0): Controls how much noise features are blended
- **target_layers**: Which UNet layers to apply switching to ["up", "mid", "down"]
- **mask_method**: Boundary mask generation method ("edge_detection", "manual")
- **num_inference_steps**: Number of denoising steps
- **guidance_scale**: Classifier-free guidance strength

### Mask Generation

- **Edge Detection**: Uses Canny edge detection + dilation
- **Manual**: Creates simple geometric masks for testing
- **Custom**: Implement your own mask generation logic

## File Structure

```
NewtonRaphsonInversion/
├── src/
│   ├── attention_switching_processor.py    # Attention manipulation
│   ├── ddim_object_insertion_example.py    # Main inserter class
│   ├── ddim_inversion_obj_insert.py        # DDIM inversion
│   ├── config.py                           # Configuration
│   ├── euler_scheduler.py                  # Custom scheduler
│   └── sdxl_inversion_pipeline.py          # SDXL pipeline
├── demo_ddim_object_insertion.py           # Main demo script
├── main_complete.py                        # Integrated demo
├── requirements_complete.txt               # Dependencies
└── README_DDIM_Object_Insertion.md        # This file
```

## Examples

### Basic Object Insertion

```python
# Simple object insertion
inserter = DDIMObjectInserter()
result = inserter.insert_object(
    fg_image_path="cat.jpg",
    bg_image_path="garden.jpg",
    output_prompt="a cat in a beautiful garden"
)
```

### Advanced Control

```python
# Advanced control with custom parameters
inserter = DDIMObjectInserter(device="cuda", dtype=torch.float16)

# Custom boundary mask
boundary_mask = inserter.create_boundary_mask(
    fg_image, bg_image, 
    method="edge_detection",
    dilation_size=15,
    blur_sigma=3.0
)

# Setup attention switching
inserter.setup_attention_switching(
    boundary_mask=boundary_mask,
    blend_strength=0.9,
    target_layers=["up", "mid"]
)

# Generate with custom settings
result = inserter.generate_with_object_insertion(
    prompt="photorealistic composite scene",
    guidance_scale=8.0,
    num_inference_steps=100
)
```

## Technical Details

### DDIM Inversion Process

1. Encode images to latent space using VAE
2. Perform deterministic inversion (eta=0) through noise schedule
3. Store intermediate latents at each timestep
4. Use dual text encoders for SDXL compatibility

### Attention Mechanism

The attention switching operates on self-attention layers by:
- Extracting Q/K/V from stored noise latents
- Applying spatial masks to control blending regions
- Preserving cross-attention layers unchanged

### Memory Optimization

- Uses xformers for memory-efficient attention
- Supports gradient checkpointing
- Configurable precision (fp16/fp32)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or image resolution
   - Use fp16 precision
   - Enable gradient checkpointing

2. **Poor Blending Quality**
   - Adjust blend_strength parameter
   - Modify boundary mask dilation/blur
   - Try different target_layers

3. **Slow Performance**
   - Reduce num_inference_steps
   - Use smaller image sizes
   - Enable xformers optimization

### Performance Tips

- Use CUDA for GPU acceleration
- Enable xformers for memory efficiency
- Use fp16 precision when possible
- Optimize boundary mask resolution

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ddim_object_insertion,
  title={DDIM Object Insertion with Attention Switching},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on DDIM (Denoising Diffusion Implicit Models)
- Uses Stable Diffusion XL architecture
- Inspired by attention manipulation techniques in diffusion models 