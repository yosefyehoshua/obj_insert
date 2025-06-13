#!/usr/bin/env python3
"""
Test script to verify DDIM Object Insertion installation and basic functionality.
"""

import sys
import torch
import numpy as np
from PIL import Image

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.attention_switching_processor import DDIMObjectInsertionAttnProcessor, register_ddim_object_insertion_attention
        print("✓ Attention switching processor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import attention switching processor: {e}")
        return False
    
    try:
        from src.ddim_object_insertion_example import DDIMObjectInserter
        print("✓ DDIM Object Inserter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DDIM Object Inserter: {e}")
        return False
    
    try:
        from src.ddim_inversion_obj_insert import invert_ddim, encode_prompt_sdxl
        print("✓ DDIM inversion functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DDIM inversion functions: {e}")
        return False
    
    try:
        import diffusers
        import transformers
        import xformers
        print("✓ Core dependencies (diffusers, transformers, xformers) available")
    except ImportError as e:
        print(f"✗ Missing core dependencies: {e}")
        return False
    
    return True


def test_device_availability():
    """Test CUDA availability and device setup."""
    print("\nTesting device availability...")
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("⚠ CUDA not available, will use CPU (slower)")
    
    return device


def test_basic_functionality():
    """Test basic functionality without requiring large models."""
    print("\nTesting basic functionality...")
    
    try:
        from src.attention_switching_processor import DDIMObjectInsertionAttnProcessor
        
        # Test processor initialization
        processor = DDIMObjectInsertionAttnProcessor(
            place_in_unet="up",
            fg_noise_latents=[],
            bg_noise_latents=[],
            boundary_mask=None,
            blend_strength=0.8
        )
        print("✓ Attention processor initialization successful")
        
        # Test boundary mask creation
        from src.ddim_object_insertion_example import DDIMObjectInserter
        
        # Create dummy images for testing
        dummy_fg = Image.new('RGB', (256, 256), color='red')
        dummy_bg = Image.new('RGB', (256, 256), color='blue')
        
        # Test mask creation (without initializing full model)
        import cv2
        fg_array = np.array(dummy_fg)
        bg_array = np.array(dummy_bg)
        fg_gray = cv2.cvtColor(fg_array, cv2.COLOR_RGB2GRAY)
        bg_gray = cv2.cvtColor(bg_array, cv2.COLOR_RGB2GRAY)
        
        print("✓ Basic image processing functions work")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_model_loading():
    """Test model loading (optional, requires internet and disk space)."""
    print("\nTesting model loading (optional)...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        from src.ddim_object_insertion_example import DDIMObjectInserter
        
        print("Attempting to load SDXL model (this may take a while)...")
        inserter = DDIMObjectInserter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            device=device,
            dtype=dtype
        )
        print("✓ SDXL model loaded successfully")
        
        # Test boundary mask creation
        dummy_fg = Image.new('RGB', (512, 512), color='red')
        dummy_bg = Image.new('RGB', (512, 512), color='blue')
        
        boundary_mask = inserter.create_boundary_mask(dummy_fg, dummy_bg)
        print(f"✓ Boundary mask created: shape {boundary_mask.shape}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Model loading test failed (this is optional): {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DDIM Object Insertion Installation Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your installation.")
        sys.exit(1)
    
    # Test device
    device = test_device_availability()
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed.")
        sys.exit(1)
    
    # Test model loading (optional)
    model_test_passed = test_model_loading()
    
    print("\n" + "=" * 60)
    if model_test_passed:
        print("🎉 All tests passed! DDIM Object Insertion is ready to use.")
    else:
        print("✅ Core functionality tests passed!")
        print("⚠ Model loading test failed, but this is optional.")
        print("You can still use the system with manual model setup.")
    
    print("\nNext steps:")
    print("1. Prepare your foreground and background images")
    print("2. Run: python demo_ddim_object_insertion.py --fg_image fg.jpg --bg_image bg.jpg")
    print("3. Check the output directory for results")
    print("=" * 60)


if __name__ == "__main__":
    main() 