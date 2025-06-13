import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional
from diffusers import StableDiffusionXLPipeline
from diffusers.schedulers import DDIMScheduler

from .ddim_inversion_obj_insert import invert_ddim, ddim_reconstruction
from .attention_switching_processor import register_ddim_object_insertion_attention, DDIMObjectInsertionAttnProcessor


class DDIMObjectInserter:
    """
    Complete implementation of DDIM-based object insertion with attention manipulation.
    
    This class implements the full pipeline described in rules.py:
    1. Invert both foreground and background images using DDIM inversion
    2. Save all intermediate noise latents (x_t) for both fg and bg  
    3. Generate a binary mask for boundary regions
    4. During forward denoising, replace keys and queries at boundary regions with noise
    5. Preserve original keys/queries in non-masked regions
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = dtype
        
        # Load SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        
        # Use DDIM scheduler for deterministic inversion
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Storage for inverted latents
        self.fg_noise_latents: List[torch.Tensor] = []
        self.bg_noise_latents: List[torch.Tensor] = []
        self.boundary_mask: Optional[torch.Tensor] = None
        
    def create_boundary_mask(
        self,
        fg_image: Image.Image,
        bg_image: Image.Image,
        method: str = "edge_detection",
        dilation_size: int = 10,
        blur_sigma: float = 2.0
    ) -> torch.Tensor:
        """
        Generate a binary mask for boundary regions between fg and bg.
        
        Args:
            fg_image: Foreground image
            bg_image: Background image  
            method: Method for mask generation ("edge_detection", "alpha_matting", "manual")
            dilation_size: Size of dilation kernel for expanding boundary regions
            blur_sigma: Gaussian blur sigma for soft edges
            
        Returns:
            Binary mask tensor [H, W] where 1 = boundary region, 0 = preserve original
        """
        if method == "edge_detection":
            # Convert images to grayscale
            fg_gray = cv2.cvtColor(np.array(fg_image), cv2.COLOR_RGB2GRAY)
            bg_gray = cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            fg_edges = cv2.Canny(fg_gray, 50, 150)
            bg_edges = cv2.Canny(bg_gray, 50, 150)
            
            # Combine edges
            combined_edges = cv2.bitwise_or(fg_edges, bg_edges)
            
            # Dilate to create boundary regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            boundary_mask = cv2.dilate(combined_edges, kernel, iterations=1)
            
        elif method == "alpha_matting":
            # Placeholder for alpha matting implementation
            # You would implement alpha matting here or use a library like pymatting
            raise NotImplementedError("Alpha matting not implemented. Use edge_detection method.")
            
        elif method == "manual":
            # Create a simple center region mask as example
            h, w = fg_image.size[1], fg_image.size[0]
            boundary_mask = np.zeros((h, w), dtype=np.uint8)
            center_h, center_w = h // 2, w // 2
            boundary_mask[center_h-50:center_h+50, center_w-50:center_w+50] = 255
            
        else:
            raise ValueError(f"Unknown mask method: {method}")
        
        # Apply Gaussian blur for soft edges
        if blur_sigma > 0:
            boundary_mask = cv2.GaussianBlur(boundary_mask, (0, 0), blur_sigma)
        
        # Normalize to [0, 1]
        boundary_mask = boundary_mask.astype(np.float32) / 255.0
        
        return torch.from_numpy(boundary_mask).to(self.device)
    
    def invert_images(
        self,
        fg_image: Image.Image,
        bg_image: Image.Image,
        fg_prompt: str = "",
        bg_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform DDIM inversion on both foreground and background images.
        
        Args:
            fg_image: Foreground image
            bg_image: Background image
            fg_prompt: Text prompt for foreground inversion
            bg_prompt: Text prompt for background inversion
            num_inference_steps: Number of DDIM inversion steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Tuple of (fg_noise_latents, bg_noise_latents)
        """
        print("Inverting foreground image...")
        fg_latent, fg_intermediates, fg_noise = invert_ddim(
            x0=fg_image,
            pipe=self.pipe,
            prompt_src=fg_prompt,
            num_diffusion_steps=num_inference_steps,
            cfg_scale_src=guidance_scale,
            eta=0.0,  # Deterministic inversion
            capture_noise_patterns=True
        )
        
        print("Inverting background image...")
        bg_latent, bg_intermediates, bg_noise = invert_ddim(
            x0=bg_image,
            pipe=self.pipe,
            prompt_src=bg_prompt,
            num_diffusion_steps=num_inference_steps,
            cfg_scale_src=guidance_scale,
            eta=0.0,  # Deterministic inversion
            capture_noise_patterns=True
        )
        
        # Store the intermediate latents (these represent the noise at each timestep)
        self.fg_noise_latents = fg_intermediates
        self.bg_noise_latents = bg_intermediates
        
        print(f"Stored {len(self.fg_noise_latents)} foreground noise latents")
        print(f"Stored {len(self.bg_noise_latents)} background noise latents")
        
        return self.fg_noise_latents, self.bg_noise_latents
    
    def setup_attention_switching(
        self,
        boundary_mask: torch.Tensor,
        blend_strength: float = 0.8,
        target_layers: List[str] = ["up", "mid"]
    ):
        """
        Setup attention processors for switching between noise and original features.
        
        Args:
            boundary_mask: Binary mask defining boundary regions
            blend_strength: Strength of attention blending (0.0 to 1.0)
            target_layers: UNet layers to apply attention switching to
        """
        self.boundary_mask = boundary_mask
        
        # Register custom attention processors
        self.attn_processors = register_ddim_object_insertion_attention(
            unet=self.pipe.unet,
            fg_noise_latents=self.fg_noise_latents,
            bg_noise_latents=self.bg_noise_latents,
            boundary_mask=boundary_mask,
            blend_strength=blend_strength,
            target_layers=target_layers
        )
        
        print(f"Registered attention switching for {len(self.attn_processors)} attention layers")
    
    def update_timestep_in_processors(self, timestep_idx: int):
        """Update the current timestep in all attention processors."""
        for processor in self.attn_processors.values():
            if isinstance(processor, DDIMObjectInsertionAttnProcessor):
                processor.update_timestep(timestep_idx)
    
    def generate_with_object_insertion(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        generator: Optional[torch.Generator] = None
    ) -> Image.Image:
        """
        Generate image with object insertion using attention switching.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height, width: Output image dimensions
            generator: Random generator for reproducibility
            
        Returns:
            Generated PIL Image
        """
        if not self.fg_noise_latents or not self.bg_noise_latents:
            raise ValueError("Must call invert_images() first to generate noise latents")
        
        if self.boundary_mask is None:
            raise ValueError("Must call setup_attention_switching() first to set boundary mask")
        
        # Custom generation loop with timestep updates
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        
        # Prepare latents
        shape = (1, self.pipe.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Encode prompts
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            negative_prompt_2=None
        )
        
        # Prepare added conditioning for SDXL
        # Get text encoder projection dimension
        if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim
        else:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        
        # Check if this is the custom object insertion pipeline or standard SDXL pipeline
        try:
            # Try the extended method signature first (custom pipeline)
            add_time_ids, add_neg_time_ids = self.pipe._get_add_time_ids(
                (height, width), (0, 0), (height, width),
                aesthetic_score=6.0,
                negative_aesthetic_score=2.5,
                negative_original_size=(height, width),
                negative_crops_coords_top_left=(0, 0),
                negative_target_size=(height, width),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim
            )
        except TypeError:
            # Fall back to standard SDXL pipeline method signature
            add_time_ids = self.pipe._get_add_time_ids(
                (height, width), (0, 0), (height, width),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim
            )
            add_neg_time_ids = add_time_ids  # Use same for both positive and negative
        
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
        else:
            add_text_embeds = pooled_prompt_embeds
        
        add_time_ids = add_time_ids.to(self.device)
        
        # Denoising loop with attention switching
        for i, t in enumerate(self.pipe.progress_bar(timesteps)):
            # Update timestep in attention processors
            self.update_timestep_in_processors(i)
            
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # Apply CFG
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents to image
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        return image
    
    def insert_object(
        self,
        fg_image_path: str,
        bg_image_path: str,
        output_prompt: str,
        fg_prompt: str = "",
        bg_prompt: str = "",
        mask_method: str = "edge_detection",
        blend_strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Complete object insertion pipeline.
        
        Args:
            fg_image_path: Path to foreground image
            bg_image_path: Path to background image
            output_prompt: Text prompt for final generation
            fg_prompt: Text prompt for foreground inversion
            bg_prompt: Text prompt for background inversion
            mask_method: Method for boundary mask generation
            blend_strength: Attention blending strength
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG scale
            output_path: Optional path to save result
            
        Returns:
            Generated PIL Image with object insertion
        """
        # Load images
        fg_image = Image.open(fg_image_path).convert("RGB").resize((1024, 1024))
        bg_image = Image.open(bg_image_path).convert("RGB").resize((1024, 1024))
        
        print("Step 1: Creating boundary mask...")
        boundary_mask = self.create_boundary_mask(
            fg_image, bg_image, method=mask_method
        )
        
        print("Step 2: Inverting images with DDIM...")
        self.invert_images(
            fg_image, bg_image, fg_prompt, bg_prompt, 
            num_inference_steps, guidance_scale
        )
        
        print("Step 3: Setting up attention switching...")
        self.setup_attention_switching(boundary_mask, blend_strength)
        
        print("Step 4: Generating with object insertion...")
        result_image = self.generate_with_object_insertion(
            prompt=output_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        if output_path:
            result_image.save(output_path)
            print(f"Saved result to {output_path}")
        
        return result_image


# Example usage
if __name__ == "__main__":
    # Initialize the object inserter
    inserter = DDIMObjectInserter(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda"
    )
    
    # Perform object insertion
    result = inserter.insert_object(
        fg_image_path="path/to/foreground.jpg",
        bg_image_path="path/to/background.jpg", 
        output_prompt="a seamless composite image with natural lighting",
        fg_prompt="a detailed object",
        bg_prompt="a natural background scene",
        mask_method="edge_detection",
        blend_strength=0.8,
        num_inference_steps=50,
        guidance_scale=7.5,
        output_path="output/inserted_object.jpg"
    )
    
    print("Object insertion completed!") 