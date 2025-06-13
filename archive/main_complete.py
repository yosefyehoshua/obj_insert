import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

from src.config import RunConfig
from src.euler_scheduler import MyEulerAncestralDiscreteScheduler
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image
from src.sdxl_inversion_pipeline import SDXLDDIMPipeline
from diffusers.utils.torch_utils import randn_tensor
from src.attention_switching_processor import register_ddim_object_insertion_attention, DDIMObjectInsertionAttnProcessor
from src.ddim_object_insertion_example import DDIMObjectInserter


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
    return center_crop(Image.open(im_path)).resize((512, 512))


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


class ImageEditorDemo:
    def __init__(self, pipe_inversion, pipe_inference, input_image, description_prompt, cfg):
        self.pipe_inversion = pipe_inversion
        self.pipe_inference = pipe_inference
        self.original_image = load_im_into_format_from_path(input_image).convert("RGB")
        self.load_image = True
        g_cpu = torch.Generator().manual_seed(7865)
        img_size = (512,512)
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

        self.last_latent = self.invert(self.original_image, description_prompt)
        self.original_latent = self.last_latent

    def invert(self, init_image, base_prompt):
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
        image = self.pipe_inference(prompt=target_prompt,
                            num_inference_steps=self.cfg.num_inference_steps,
                            negative_prompt="",
                            callback_on_step_end=inference_callback,
                            image=self.last_latent,
                            strength=self.cfg.inversion_max_step,
                            denoising_start=1.0 - self.cfg.inversion_max_step,
                            guidance_scale=self.edit_cfg).images[0]
        return image


class ImageEditorDemoWithAttentionSwitching(ImageEditorDemo):
    def __init__(self, pipe_inversion, pipe_inference, input_image, description_prompt, cfg, 
                 background_image=None, use_attention_switching=False):
        super().__init__(pipe_inversion, pipe_inference, input_image, description_prompt, cfg)
        self.background_image = background_image
        self.use_attention_switching = use_attention_switching
        
        # Initialize DDIM Object Inserter if using attention switching
        if use_attention_switching and background_image:
            self.ddim_inserter = DDIMObjectInserter(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                device="cuda",
                dtype=torch.float16
            )
            self.setup_ddim_object_insertion(description_prompt)

    def setup_ddim_object_insertion(self, description_prompt):
        """
        Setup DDIM object insertion with attention switching.
        """
        if not self.background_image:
            print("Warning: No background image provided, skipping DDIM object insertion setup")
            return
            
        print("Setting up DDIM object insertion with attention switching...")
        
        # Load and preprocess background image
        bg_image = load_im_into_format_from_path(self.background_image).convert("RGB")
        
        # Create boundary mask
        print("Creating boundary mask...")
        boundary_mask = create_boundary_mask_simple(
            self.original_image, bg_image, device="cuda"
        )
        
        # Perform DDIM inversion on both images
        print("Inverting foreground and background images...")
        fg_noise_latents, bg_noise_latents = self.ddim_inserter.invert_images(
            self.original_image, bg_image,
            fg_prompt=description_prompt,
            bg_prompt=description_prompt,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale
        )
        
        # Setup attention switching
        print("Setting up attention switching...")
        self.ddim_inserter.setup_attention_switching(
            boundary_mask=boundary_mask,
            blend_strength=0.8,
            target_layers=["up", "mid"]
        )
        
        print("DDIM object insertion setup complete!")

    def edit(self, target_prompt):
        """
        Edit with optional DDIM object insertion and attention switching.
        """
        if self.use_attention_switching and hasattr(self, 'ddim_inserter'):
            print("Generating with DDIM object insertion...")
            image = self.ddim_inserter.generate_with_object_insertion(
                prompt=target_prompt,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.edit_cfg,
                height=512,  # Match original pipeline resolution
                width=512
            )
        else:
            # Use original editing method
            image = super().edit(target_prompt)
        
        return image


def run_comparison_demo():
    """
    Run a comparison demo showing both original and DDIM object insertion methods.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = (512, 512)
    scheduler_class = MyEulerAncestralDiscreteScheduler
    
    # Load pipelines
    pipe_inversion = SDXLDDIMPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True,
                                                      safety_checker=None, cache_dir="/inputs/huggingface_cache").to(device)
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

    # Example images
    input_image = "/colab/NewtonRaphsonInversion/example_images/obj_back_mask.png"
    background_image = "/colab/NewtonRaphsonInversion/example_images/background.png"  # Add your background image
    description_prompt = ''
    editing_prompt = ""

    # # Original method
    # print("Running original method...")
    # editor_original = ImageEditorDemo(pipe_inversion, pipe_inference, input_image, description_prompt, config)
    # result_original = editor_original.edit(editing_prompt)

    # DDIM object insertion method
    print("Running DDIM object insertion method...")
    editor_ddim = ImageEditorDemoWithAttentionSwitching(
        pipe_inversion, pipe_inference, input_image, description_prompt, config,
        background_image=background_image,
        use_attention_switching=True
    )
    result_ddim = editor_ddim.edit(editing_prompt)

    # Display results
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # axes[0].imshow(editor_original.original_image)
    # axes[0].set_title("Original Image")
    # axes[0].axis('off')
    
    # axes[1].imshow(result_original)
    # axes[1].set_title("Original Method")
    # axes[1].axis('off')
    plt.imshow(result_ddim)
    # axes[2].imshow(result_ddim)
    # axes[2].set_title("DDIM Object Insertion")
    # axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("/colab/NewtonRaphsonInversion/example_images/result_ddim.png", dpi=150, bbox_inches='tight')
    # plt.show()
    
    print("Comparison completed! Results saved to comparison_results.png")


if __name__ == '__main__':
    # Check if example images exist
    import os
    
    if os.path.exists("example_images/lion.jpeg") and os.path.exists("example_images/background.png"):
        run_comparison_demo()
    else:
        print("Example images not found. Please ensure you have:")
        print("- example_images/lion.jpeg")
        print("- example_images/background.png")
        print("\nAlternatively, you can use the demo script:")
        print("python demo_ddim_object_insertion.py --fg_image path/to/fg.jpg --bg_image path/to/bg.jpg")
        
        # Run a simple demo with DDIM object insertion
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize DDIM Object Inserter directly
        inserter = DDIMObjectInserter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        print("DDIM Object Inserter initialized successfully!")
        print("Use demo_ddim_object_insertion.py for complete functionality.") 