#!/usr/bin/env python3
"""
DDIM Object Insertion Demo

This script demonstrates the complete DDIM-based object insertion pipeline
with attention switching as described in rules.py.

Usage:
    python demo_ddim_object_insertion.py --fg_image path/to/foreground.jpg --bg_image path/to/background.jpg
"""

import torch
import numpy as np
from PIL import Image
import cv2
import argparse
import os
from pathlib import Path
from matplotlib import pyplot as plt

from src.ddim_object_insertion_example import DDIMObjectInserter
from src.config import RunConfig


def create_boundary_mask_simple(fg_image, bg_image, device="cuda"):
    """
    Create a simple boundary mask using edge detection.
    """
    # Convert PIL images to numpy arrays
    fg_array = np.array(fg_image)
    bg_array = np.array(bg_image)
    
    # Convert to grayscale
    fg_gray = cv2.cvtColor(fg_array, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(bg_array, cv2.COLOR_RGB2GRAY)
    
    # Detect edges
    fg_edges = cv2.Canny(fg_gray, 50, 150)
    bg_edges = cv2.Canny(bg_gray, 50, 150)
    
    # Combine edges
    combined_edges = cv2.bitwise_or(fg_edges, bg_edges)
    
    # Dilate to create boundary regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    boundary_mask = cv2.dilate(combined_edges, kernel, iterations=1)
    
    # Apply Gaussian blur for soft edges
    boundary_mask = cv2.GaussianBlur(boundary_mask, (0, 0), 2.0)
    
    # Normalize to [0, 1]
    boundary_mask = boundary_mask.astype(np.float32) / 255.0
    
    return torch.from_numpy(boundary_mask).to(device)


def load_and_preprocess_image(image_path: str, size: tuple = (1024, 1024)) -> Image.Image:
    """Load and preprocess an image."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size, Image.LANCZOS)
    return image


def save_results(images: dict, output_dir: str):
    """Save all result images to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, image in images.items():
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:  # CHW format
                image = image.permute(1, 2, 0)
            image = (image.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
            image = Image.fromarray(image)
        
        output_path = os.path.join(output_dir, f"{name}.png")
        image.save(output_path)
        print(f"Saved {name} to {output_path}")


def visualize_results(fg_image, bg_image, boundary_mask, result_image, output_dir):
    """Create a visualization showing the complete pipeline."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(fg_image)
    axes[0, 0].set_title("Foreground Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(bg_image)
    axes[0, 1].set_title("Background Image")
    axes[0, 1].axis('off')
    
    # Boundary mask
    if isinstance(boundary_mask, torch.Tensor):
        mask_np = boundary_mask.cpu().numpy()
    else:
        mask_np = boundary_mask
    axes[0, 2].imshow(mask_np, cmap='gray')
    axes[0, 2].set_title("Boundary Mask")
    axes[0, 2].axis('off')
    
    # Result
    axes[1, 1].imshow(result_image)
    axes[1, 1].set_title("DDIM Object Insertion Result")
    axes[1, 1].axis('off')
    
    # Hide unused subplots
    axes[1, 0].axis('off')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pipeline_visualization.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pipeline visualization to {output_dir}/pipeline_visualization.png")


def main():
    parser = argparse.ArgumentParser(description="DDIM Object Insertion Demo")
    parser.add_argument("--fg_image", type=str, required=True, help="Path to foreground image")
    parser.add_argument("--bg_image", type=str, required=True, help="Path to background image")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--prompt", type=str, default="a seamless composite image with natural lighting", 
                       help="Text prompt for generation")
    parser.add_argument("--fg_prompt", type=str, default="", help="Foreground inversion prompt")
    parser.add_argument("--bg_prompt", type=str, default="", help="Background inversion prompt")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--blend_strength", type=float, default=0.8, help="Attention blending strength")
    parser.add_argument("--mask_method", type=str, default="edge_detection", 
                       choices=["edge_detection", "manual"], help="Boundary mask generation method")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", 
                       help="SDXL model ID")
    
    args = parser.parse_args()
    
    # Check if images exist
    if not os.path.exists(args.fg_image):
        raise FileNotFoundError(f"Foreground image not found: {args.fg_image}")
    if not os.path.exists(args.bg_image):
        raise FileNotFoundError(f"Background image not found: {args.bg_image}")
    
    print("=" * 60)
    print("DDIM Object Insertion Demo")
    print("=" * 60)
    print(f"Foreground image: {args.fg_image}")
    print(f"Background image: {args.bg_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Model: {args.model_id}")
    print("=" * 60)
    
    # Load images
    print("Loading and preprocessing images...")
    fg_image = load_and_preprocess_image(args.fg_image)
    bg_image = load_and_preprocess_image(args.bg_image)
    
    # Initialize the object inserter
    print("Initializing DDIM Object Inserter...")
    inserter = DDIMObjectInserter(
        model_id=args.model_id,
        device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32
    )
    
    # Create boundary mask
    print("Creating boundary mask...")
    boundary_mask = inserter.create_boundary_mask(
        fg_image, bg_image, 
        method=args.mask_method,
        dilation_size=10,
        blur_sigma=2.0
    )
    
    # Perform DDIM inversion
    print("Performing DDIM inversion...")
    fg_noise_latents, bg_noise_latents = inserter.invert_images(
        fg_image, bg_image,
        fg_prompt=args.fg_prompt,
        bg_prompt=args.bg_prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale
    )
    
    # Setup attention switching
    print("Setting up attention switching...")
    inserter.setup_attention_switching(
        boundary_mask=boundary_mask,
        blend_strength=args.blend_strength,
        target_layers=["up", "mid"]
    )
    
    # Generate result with object insertion
    print("Generating result with object insertion...")
    result_image = inserter.generate_with_object_insertion(
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        height=1024,
        width=1024
    )
    
    # Save results
    print("Saving results...")
    results = {
        "foreground": fg_image,
        "background": bg_image,
        "boundary_mask": boundary_mask,
        "result": result_image
    }
    save_results(results, args.output_dir)
    
    # Create visualization
    print("Creating visualization...")
    visualize_results(fg_image, bg_image, boundary_mask, result_image, args.output_dir)
    
    print("=" * 60)
    print("DDIM Object Insertion completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main() 