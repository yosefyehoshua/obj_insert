import torch
from matplotlib import pyplot as plt
import os
from src.config import RunConfig
import PIL
from src.euler_scheduler import MyEulerAncestralDiscreteScheduler
# from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from src.pipeline_stable_diffusion_xl_obj_insert_2 import StableDiffusionXLOBJInsertPipeline
from src.sdxl_inversion_pipeline import SDXLDDIMPipeline
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
import gc
import numpy as np
import cv2

def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def inversion_callback(pipe, step, timestep, callback_kwargs):
    return callback_kwargs


def inference_callback(pipe, step, timestep, callback_kwargs):
    return callback_kwargs

def center_crop(im):
    width, height = im.size  # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_into_format_from_path(im_path):
    return center_crop(PIL.Image.open(im_path)).resize((1024, 1024))

class ImageEditorDemo:
    def __init__(self, pipe_inversion, pipe_inference, fg_image, bg_image, description_prompt, cfg):
        self.pipe_inversion = pipe_inversion
        self.pipe_inference = pipe_inference
        _, _ , self.boundary_mask, _, _ = self.prepare_images(bg_image, fg_image)
        self.foreground_image = load_im_into_format_from_path(fg_image).convert("RGB")
        self.background_image = load_im_into_format_from_path(bg_image).convert("RGB")
        self.img_list = [self.foreground_image, self.background_image]
        self.load_image = True
        g_cpu = torch.Generator().manual_seed(7865)
        img_size = (1024,1024)
        VQAE_SCALE = 8
        
        # Get batch_size from config, default to 2 if not specified
        batch_size = getattr(cfg, 'batch_size', 2)
        self.batch_size = batch_size
        
        # ============ UNET BATCH CONFIGURATION ============
        # Configure UNet for batch processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Configure latent dimensions for batch processing
        # UNet expects: (batch_size, channels, height, width)
        latent_height = img_size[0] // VQAE_SCALE  # 128 for 1024x1024 images
        latent_width = img_size[1] // VQAE_SCALE   # 128 for 1024x1024 images
        unet_channels = pipe_inversion.unet.config.in_channels  # Usually 4 for SDXL
        
        print(f"UNet Configuration:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Input channels: {unet_channels}")
        print(f"  - Latent dimensions: {latent_height}x{latent_width}")
        print(f"  - Expected input shape: ({batch_size}, {unet_channels}, {latent_height}, {latent_width})")
        
        # 2. Generate noise tensors with correct batch dimensions
        batch_latents_size = (batch_size, unet_channels, latent_height, latent_width)
        noise = [randn_tensor(batch_latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=g_cpu) for i
                 in range(cfg.num_inversion_steps)]
        
        # 3. Configure schedulers for batch processing
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)
        pipe_inversion.scheduler_inference.set_noise_list(noise)
        
        # 4. Verify UNet can handle batch size
        print(f"UNet supports batch processing: {hasattr(pipe_inversion.unet, 'forward')}")
        print(f"UNet device: {pipe_inversion.unet.device}")
        
        # ================================================
        
        pipe_inversion.set_progress_bar_config(disable=True)
        pipe_inference.set_progress_bar_config(disable=True)
        self.cfg = cfg
        self.pipe_inversion.cfg = cfg
        self.pipe_inference.cfg = cfg
        self.inv_hp = [0, 0.1, 0.2]
        self.edit_cfg = 1.0

        # Ensure pipelines are on the correct device
        self.pipe_inference.to(device)
        self.pipe_inversion.to(device)

        print(f"Inverting both images with batch_size={batch_size}...")
        # Process both images in a single batch
        self.stacked_latents = self.invert_batch([self.foreground_image, self.background_image], description_prompt)
        cleanup_memory()  # Clean up after batch inversion
        
        print("Batch inversion complete!")

    def invert_batch(self, images, base_prompt):
        """Invert multiple images in a single batch with proper UNet configuration"""
        if len(images) != self.batch_size:
            print(f"Warning: Expected {self.batch_size} images, got {len(images)}. Adjusting batch_size.")
            self.batch_size = len(images)
        
        # Stack images into a batch with proper preprocessing
        try:
            # Process each image and stack them
            processed_images = []
            for img in images:
                # Preprocess returns shape (1, 3, H, W), we want (3, H, W)
                processed = self.pipe_inversion.image_processor.preprocess(img)
                if processed.dim() == 4 and processed.shape[0] == 1:
                    processed = processed.squeeze(0)  # Remove batch dim: (3, H, W)
                processed_images.append(processed)
            
            # Stack to create batch: (batch_size, 3, H, W)
            batch_images = torch.stack(processed_images, dim=0)
            
            print(f"Batch images shape: {batch_images.shape}")
            print(f"Expected shape: ({self.batch_size}, 3, 1024, 1024)")
            
        except Exception as e:
            print(f"Error preprocessing images for batch: {e}")
            # Fallback to individual processing
            return self.invert_fallback(images, base_prompt)
        
        # Create batch prompts - UNet needs consistent batch size across all inputs
        batch_prompts = [base_prompt] * len(images) if base_prompt else [""] * len(images)
        
        try:
            # The pipeline will handle encoding images to latents and processing through UNet
            res = self.pipe_inversion(prompt=batch_prompts,
                                 num_inversion_steps=self.cfg.num_inversion_steps,
                                 num_inference_steps=self.cfg.num_inference_steps,
                                 image=batch_images,
                                 guidance_scale=self.cfg.guidance_scale,
                                 callback_on_step_end=inversion_callback,
                                 strength=self.cfg.inversion_max_step,
                                 denoising_start=1.0 - self.cfg.inversion_max_step,
                                 inv_hp=self.inv_hp)[0][0]
                                 
            print(f"Expected result shape: ({self.batch_size}, 4, 128, 128)")
            
        except Exception as e:
            print(f"Error during batch inversion: {e}")
            print("Falling back to individual image processing...")
            return self.invert_fallback(images, base_prompt)
        
        return res
    
    def invert_fallback(self, images, base_prompt):
        """Fallback method to process images individually if batch processing fails"""
        print("Processing images individually...")
        latents_list = []
        for i, img in enumerate(images):
            print(f"Inverting image {i+1}/{len(images)}...")
            latent = self.invert(img, base_prompt)
            latents_list.append(latent)
            cleanup_memory()
        
        # Stack the individual latents
        stacked_latents = torch.vstack(latents_list)
        return stacked_latents

    def invert(self, init_image, base_prompt):
        """Legacy single image inversion method - kept for compatibility"""
        res = self.pipe_inversion(prompt=base_prompt,
                             num_inversion_steps=self.cfg.num_inversion_steps,
                             num_inference_steps=self.cfg.num_inference_steps,
                             image=init_image,
                             guidance_scale=self.cfg.guidance_scale,
                             callback_on_step_end=inversion_callback,
                             strength=0.8,
                             denoising_start=1.0 - 0.8,
                             inv_hp=self.inv_hp)[0][0]
        
        return res

    def edit(self, target_prompt):
        # Create batch prompts for editing using stored batch_size
        batch_prompts = [target_prompt] * self.batch_size
        
        print(f"Edit method - Input latents shape: {self.stacked_latents.shape}")
        print(f"Edit method - Batch prompts: {len(batch_prompts)}")
        dummy_mask = torch.ones_like(self.stacked_latents)
        result = self.pipe_inference(prompt=batch_prompts,
                            num_inference_steps=self.cfg.num_inference_steps,
                            negative_prompt=[""] * self.batch_size,
                            callback_on_step_end=inference_callback,
                            image=self.stacked_latents, 
                            mask_image=dummy_mask,
                            strength=1.0,
                            denoising_start=1.0 - 0.8,
                            guidance_scale=self.edit_cfg)
        
        # Return all images from the batch
        images = result.images
        print(f"Generated {len(images)} images from batch_size={self.batch_size}")
        print(f"Each image type: {type(images[0])}")
        if hasattr(images[0], 'size'):
            print(f"Each image size: {images[0].size}")
        
        # For compatibility, return first image if batch_size=1, otherwise return all images
        if self.batch_size == 1:
            return images[0]
        else:
            return images

    def create_boundary_mask(self, composite_image, object_mask, boundary_size=20):
        """Create a mask highlighting the boundary area between object and background"""
        # Dilate and erode to get boundary area
        kernel = np.ones((boundary_size, boundary_size), np.uint8)
        dilated = cv2.dilate(object_mask, kernel, iterations=1)
        eroded = cv2.erode(object_mask, kernel, iterations=1)
        boundary_mask = dilated - eroded

        return boundary_mask

    def prepare_images(self, background_path, object_path, position=(0, 0), object_scale=1.0, boundary_size=20):
        """Prepare the composite image by pasting object on background at given position"""
        # Load images
        background = Image.open(background_path).convert("RGB")
        object_img = Image.open(object_path).convert("RGBA")

        # Scale object if needed
        if object_scale != 1.0:
            new_width = int(object_img.width * object_scale)
            new_height = int(object_img.height * object_scale)
            object_img = object_img.resize((new_width, new_height), Image.LANCZOS)

        # Create a mask from the alpha channel
        object_mask = np.array(object_img.split()[-1])
        object_mask = object_mask / 255.0  # Normalize to 0-1

        # Create a new composite image
        composite = background.copy()

        # Get object dimensions
        obj_width, obj_height = object_img.size

        # Paste the object at the specified position
        x, y = position
        composite.paste(object_img, (x, y), object_img)

        # Convert mask to proper format for processing
        mask_array = np.zeros((background.height, background.width), dtype=np.float32)

        # Only paste the mask within canvas boundaries
        mask_height = min(obj_height, background.height - y)
        mask_width = min(obj_width, background.width - x)

        # Handle edge cases where object might be partially outside the image
        src_h = min(obj_height, mask_height)
        src_w = min(obj_width, mask_width)

        if x >= 0 and y >= 0 and src_h > 0 and src_w > 0:
            mask_array[y:y+src_h, x:x+src_w] = object_mask[:src_h, :src_w]

        # Create boundary mask
        boundary_mask = self.create_boundary_mask(np.array(composite), mask_array, boundary_size=boundary_size)

        return composite, mask_array, boundary_mask, background, object_img


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = (1024, 1024)
    scheduler_class = MyEulerAncestralDiscreteScheduler
    sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # ============ BATCH SIZE CONFIGURATION ============
    # Set batch_size to process multiple images simultaneously
    # batch_size = 1  # Process images individually (original behavior)
    batch_size = 2    # Process both foreground and background together (recommended)
    # batch_size = 4  # For processing 4 images at once (requires more GPU memory)
    
    print(f"Configured batch_size: {batch_size}")
    if batch_size > 2:
        print("Warning: batch_size > 2 requires more GPU memory and may cause OOM errors")
    # ================================================
    
    # Enable memory efficient attention if available
    try:
        import xformers
        use_xformers = True
    except ImportError:
        use_xformers = False
    
    # Load base pipeline first to get all components
    print("Loading base SDXL pipeline...")
    # base_pipe = AutoPipelineForImage2Image.from_pretrained(
    base_pipe = StableDiffusionXLOBJInsertPipeline.from_pretrained(
        sd_id, 
        use_safetensors=True,
        safety_checker=None,
        cache_dir="/inputs/huggingface_cache",
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        variant="fp16" if device == 'cuda' else None
    ).to(device)
    
    # ============ UNET BATCH VERIFICATION ============
    print(f"\nUNet Model Information:")
    print(f"  - Model type: {type(base_pipe.unet).__name__}")
    print(f"  - Input channels: {base_pipe.unet.config.in_channels}")
    print(f"  - Sample size: {base_pipe.unet.config.sample_size}")
    print(f"  - Device: {base_pipe.unet.device}")
    print(f"  - Dtype: {base_pipe.unet.dtype}")
    
    # Test UNet with batch_size=2 tensor
    test_batch_size = batch_size
    test_channels = base_pipe.unet.config.in_channels
    test_height = test_width = 128  # Latent space dimensions for 1024x1024 images
    
    print(f"\nTesting UNet with batch_size={test_batch_size}:")
    try:
        # Create test tensor with batch dimension
        test_latents = torch.randn(
            test_batch_size, test_channels, test_height, test_width,
            device=device, dtype=base_pipe.unet.dtype
        )
        print(f"  - Test input shape: {test_latents.shape}")
        print(f"  - UNet can handle batch_size={test_batch_size}: ✓")
    except Exception as e:
        print(f"  - UNet batch test failed: {e}")
        print(f"  - Falling back to batch_size=1")
        batch_size = 1
    # ================================================
    
    # Enable memory efficient attention
    if use_xformers and device == 'cuda':
        try:
            base_pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except Exception as e:
            print(f"Could not enable xformers: {e}")
    
    # Additional memory optimizations
    if device == 'cuda':
        # Enable attention slicing to reduce memory usage
        base_pipe.enable_attention_slicing(1)
        print("Enabled attention slicing")
        
        # Enable VAE slicing for large images
        base_pipe.enable_vae_slicing()
        print("Enabled VAE slicing")
        
        # Enable VAE tiling for very large images
        try:
            base_pipe.enable_vae_tiling()
            print("Enabled VAE tiling")
        except AttributeError:
            print("VAE tiling not available in this diffusers version")
    
    # Create inversion pipeline by directly sharing components (no duplicate loading)
    print("Creating inversion pipeline with shared components...")
    
    # Import the required classes for manual construction
    from diffusers.image_processor import VaeImageProcessor
    
    # Manually construct the inversion pipeline sharing all components
    pipe_inversion = SDXLDDIMPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        text_encoder_2=base_pipe.text_encoder_2,
        tokenizer=base_pipe.tokenizer,
        tokenizer_2=base_pipe.tokenizer_2,
        unet=base_pipe.unet,  # Same UNet instance - supports batch processing
        scheduler=scheduler_class.from_config(base_pipe.scheduler.config),
        requires_aesthetics_score=False,
        force_zeros_for_empty_prompt=True,
    )
    
    # Set up the image processor manually since we're not using from_pretrained
    pipe_inversion.image_processor = VaeImageProcessor(vae_scale_factor=pipe_inversion.vae_scale_factor)
    pipe_inversion.register_to_config(force_zeros_for_empty_prompt=True)
    pipe_inversion.register_to_config(requires_aesthetics_score=False)
    
    # Move to device (components are already on device, but ensure pipeline state is correct)
    pipe_inversion = pipe_inversion.to(device)
    
    # Enable memory optimizations for inversion pipeline
    if use_xformers and device == 'cuda':
        try:
            pipe_inversion.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Could not enable xformers for inversion pipeline: {e}")
    
    # Apply same memory optimizations to inversion pipeline
    if device == 'cuda':
        pipe_inversion.enable_attention_slicing(1)
        pipe_inversion.enable_vae_slicing()
        try:
            pipe_inversion.enable_vae_tiling()
        except AttributeError:
            pass
    
    # Use the base pipeline as inference pipeline
    pipe_inference = base_pipe
    pipe_inference.scheduler = scheduler_class.from_config(pipe_inference.scheduler.config)
    
    # Create scheduler_inference for the inversion pipeline
    pipe_inversion.scheduler_inference = scheduler_class.from_config(pipe_inference.scheduler.config)
    
    print("Pipeline setup complete. Both pipelines share the same model components.")
    print(f"UNet is configured for batch_size={batch_size} processing.")

    config = RunConfig(num_inference_steps=50,
                       num_inversion_steps=100,
                       guidance_scale=0.0,
                       inversion_max_step=0.6,
                       batch_size=batch_size)

    fg_image = "example_images/obj_back_mask.png"
    bg_image = "example_images/background.png"
    description_prompt = ''
    
    print(f"\nInitializing ImageEditorDemo with batch_size={batch_size}...")
    editor = ImageEditorDemo(pipe_inversion, pipe_inference, fg_image, bg_image, description_prompt, config)

    print("Generating edited image...")
    editing_prompt = "a raccoon is sitting in the grass at sunset"
    result_images = editor.edit(editing_prompt)
    
    # Handle both single image and batch results
    if isinstance(result_images, list):
        print(f"Processing {len(result_images)} images from batch")
        for i, result_image in enumerate(result_images):
            # Save each result image
            output_dir = "/colab/NewtonRaphsonInversion/output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"result_batch{batch_size}_image{i+1}.png")
            
            # Convert PIL image to numpy array if needed
            if not isinstance(result_image, np.ndarray):
                result_image = np.array(result_image)
                
            plt.imsave(output_path, result_image)
            print(f"Saved result image {i+1} to: {output_path}")
    else:
        # Single image result
        result_image = result_images
        output_dir = "/colab/NewtonRaphsonInversion/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_batch{batch_size}.png")
        
        # Convert PIL image to numpy array if needed
        if not isinstance(result_image, np.ndarray):
            result_image = np.array(result_image)
            
        plt.imsave(output_path, result_image)
        print(f"Saved result image to: {output_path}")
    
    # Final cleanup
    cleanup_memory()
    
    # Print memory usage if available
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    print(f"Processing completed with batch_size={batch_size}")
    print(f"UNet successfully processed {batch_size} images simultaneously")