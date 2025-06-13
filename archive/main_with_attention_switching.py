import torch
from matplotlib import pyplot as plt
import sys
import os

# Add the consistory directory to the path to import our custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'consistory'))

from src.config import RunConfig
import PIL
from src.euler_scheduler import MyEulerAncestralDiscreteScheduler
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from src.sdxl_inversion_pipeline import SDXLDDIMPipeline
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

# Import our custom attention switching modules
from attention_switching_processor import register_ddim_object_insertion_attention, DDIMObjectInsertionAttnProcessor
from ddim_inversion_obj_insert import invert_ddim
import cv2
import numpy as np
import torch.nn.functional as F

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
    return center_crop(PIL.Image.open(im_path)).resize((512, 512))

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

class ImageEditorDemoWithAttentionSwitching:
    def __init__(self, pipe_inversion, pipe_inference, input_image, description_prompt, cfg, 
                 background_image=None, use_attention_switching=True):
        self.pipe_inversion = pipe_inversion
        self.pipe_inference = pipe_inference
        self.original_image = load_im_into_format_from_path(input_image).convert("RGB")
        self.background_image = None
        if background_image:
            self.background_image = load_im_into_format_from_path(background_image).convert("RGB")
        
        self.use_attention_switching = use_attention_switching
        self.load_image = True
        
        g_cpu = torch.Generator().manual_seed(7865)
        img_size = (512, 512)
        VQAE_SCALE = 8
        latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
        noise = [randn_tensor(latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=g_cpu) for i
                 in range(cfg.num_inversion_steps)]
        
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)
        pipe_inversion.scheduler_inference.set_noise_list(noise)
        pipe_inversion.set_progress_bar_config(disable=True)
        pipe_inference.set_progress_bar_config(disable=True)
        
        self.cfg = cfg
        self.pipe_inversion.cfg = cfg
        self.pipe_inference.cfg = cfg
        self.inv_hp = [2, 0.1, 0.2]
        self.edit_cfg = 1.2

        self.pipe_inference.to("cuda")
        self.pipe_inversion.to("cuda")

        # Store noise latents for attention switching
        self.fg_noise_latents = []
        self.bg_noise_latents = []
        self.boundary_mask = None
        self.attn_processors = None

        # Perform initial inversion
        self.last_latent = self.invert(self.original_image, description_prompt)
        self.original_latent = self.last_latent
        
        # If we have a background image and want to use attention switching, set it up
        if self.background_image and self.use_attention_switching:
            self.setup_attention_switching(description_prompt)

    def invert_with_ddim(self, image, prompt):
        """
        Use DDIM inversion to get intermediate noise latents.
        """
        try:
            # Convert PIL image to tensor format expected by invert_ddim
            image_tensor = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to("cuda")
            
            # Perform DDIM inversion
            final_latent, intermediate_latents, noise_patterns = invert_ddim(
                x0=image_tensor,
                pipe=self.pipe_inversion,
                prompt_src=prompt,
                num_diffusion_steps=self.cfg.num_inversion_steps,
                cfg_scale_src=self.cfg.guidance_scale,
                eta=0.0,  # Deterministic inversion
                capture_noise_patterns=True
            )
            
            return final_latent, intermediate_latents
            
        except Exception as e:
            print(f"DDIM inversion failed: {e}")
            # Fallback to original inversion method
            return self.invert(image, prompt), []

    def setup_attention_switching(self, description_prompt):
        """
        Setup attention switching mechanism using DDIM inversion.
        """
        print("Setting up attention switching with DDIM inversion...")
        
        # Invert foreground (original) image
        print("Inverting foreground image...")
        fg_latent, self.fg_noise_latents = self.invert_with_ddim(self.original_image, description_prompt)
        
        # Invert background image
        print("Inverting background image...")
        bg_latent, self.bg_noise_latents = self.invert_with_ddim(self.background_image, description_prompt)
        
        # Create boundary mask
        print("Creating boundary mask...")
        self.boundary_mask = create_boundary_mask_simple(
            self.original_image, self.background_image, device="cuda"
        )
        
        # Register attention processors if we have valid noise latents
        if len(self.fg_noise_latents) > 0 and len(self.bg_noise_latents) > 0:
            print(f"Registering attention processors with {len(self.fg_noise_latents)} noise latents...")
            self.attn_processors = register_ddim_object_insertion_attention(
                unet=self.pipe_inference.unet,
                fg_noise_latents=self.fg_noise_latents,
                bg_noise_latents=self.bg_noise_latents,
                boundary_mask=self.boundary_mask,
                blend_strength=0.8,
                target_layers=["up", "mid"]
            )
            print("Attention switching setup complete!")
        else:
            print("Warning: No noise latents available, attention switching disabled")
            self.use_attention_switching = False

    def update_timestep_in_processors(self, timestep_idx):
        """Update the current timestep in all attention processors."""
        if self.attn_processors:
            for processor in self.attn_processors.values():
                if isinstance(processor, DDIMObjectInsertionAttnProcessor):
                    processor.update_timestep(timestep_idx)

    def invert(self, init_image, base_prompt):
        """Original inversion method (fallback)."""
        res = self.pipe_inversion(prompt=base_prompt,
                             num_inversion_steps=self.cfg.num_inversion_steps,
                             num_inference_steps=self.cfg.num_inference_steps,
                             image=init_image,
                             guidance_scale=self.cfg.guidance_scale,
                             callback_on_step_end=inversion_callback,
                             strength=self.cfg.inversion_max_step,
                             denoising_start=1.0 - self.cfg.inversion_max_step,
                             inv_hp=self.inv_hp)[0][0]
        return res

    def edit(self, target_prompt):
        """
        Edit with optional attention switching.
        """
        if self.use_attention_switching and self.attn_processors:
            # Custom generation loop with timestep updates
            print("Generating with attention switching...")
            
            # Set up scheduler
            self.pipe_inference.scheduler.set_timesteps(self.cfg.num_inference_steps)
            timesteps = self.pipe_inference.scheduler.timesteps
            
            # Start from the inverted latent
            latents = self.last_latent.clone()
            
            # Encode prompt
            prompt_embeds = self.pipe_inference._encode_prompt(
                target_prompt, device="cuda", num_images_per_prompt=1,
                do_classifier_free_guidance=self.edit_cfg > 1.0
            )
            
            # Custom denoising loop
            for i, t in enumerate(timesteps):
                # Update timestep in attention processors
                self.update_timestep_in_processors(i)
                
                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2) if self.edit_cfg > 1.0 else latents
                latent_model_input = self.pipe_inference.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise
                noise_pred = self.pipe_inference.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                
                # Apply CFG
                if self.edit_cfg > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.edit_cfg * (noise_pred_text - noise_pred_uncond)
                
                # Scheduler step
                latents = self.pipe_inference.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Decode latents to image
            image = self.pipe_inference.vae.decode(latents / self.pipe_inference.vae.config.scaling_factor, return_dict=False)[0]
            image = self.pipe_inference.image_processor.postprocess(image, output_type="pil")[0]
            
            return image
        else:
            # Use original edit method
            image = self.pipe_inference(prompt=target_prompt,
                                num_inference_steps=self.cfg.num_inference_steps,
                                negative_prompt="",
                                callback_on_step_end=inference_callback,
                                image=self.last_latent,
                                strength=self.cfg.inversion_max_step,
                                denoising_start=1.0 - self.cfg.inversion_max_step,
                                guidance_scale=self.edit_cfg).images[0]
            return image

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = (512, 512)
    scheduler_class = MyEulerAncestralDiscreteScheduler
    
    pipe_inversion = SDXLDDIMPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True,
                                                      safety_checker=None, cache_dir="/inputs/huggingface_cache").to(
        device)
    pipe_inference = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True,
                                                                safety_checker=None,
                                                                cache_dir="/inputs/huggingface_cache").to(device)
    pipe_inference.scheduler = scheduler_class.from_config(pipe_inference.scheduler.config)
    pipe_inversion.scheduler = scheduler_class.from_config(pipe_inversion.scheduler.config)
    pipe_inversion.scheduler_inference = scheduler_class.from_config(pipe_inference.scheduler.config)

    config = RunConfig(num_inference_steps=4,
                       num_inversion_steps=4,
                       guidance_scale=0.0,
                       inversion_max_step=0.6)

    # Example with attention switching
    input_image = "example_images/lion.jpeg"
    background_image = "example_images/forest_background.jpeg"  # Add your background image
    description_prompt = 'a lion is sitting in the grass at sunset'
    
    # Create editor with attention switching enabled
    editor = ImageEditorDemoWithAttentionSwitching(
        pipe_inversion, pipe_inference, input_image, description_prompt, config,
        background_image=background_image,  # Provide background image
        use_attention_switching=True  # Enable attention switching
    )

    # Edit with object insertion
    editing_prompt = "a raccoon is sitting in the grass at sunset"
    result_image = editor.edit(editing_prompt)
    
    # Display result
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(editor.original_image)
    plt.title("Original (Foreground)")
    plt.axis('off')
    
    if editor.background_image:
        plt.subplot(1, 3, 2)
        plt.imshow(editor.background_image)
        plt.title("Background")
        plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_image)
    plt.title("Result with Attention Switching")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Object insertion with attention switching completed!") 