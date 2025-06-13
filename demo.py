import torch
from matplotlib import pyplot as plt
import gc

from src.config import RunConfig
import PIL
from src.euler_scheduler import MyEulerAncestralDiscreteScheduler
from diffusers import StableDiffusionXLInpaintPipeline
from src.sdxl_inversion_pipeline import SDXLDDIMPipeline
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

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

class ImageEditorDemo:
    def __init__(self, pipe_inversion, pipe_inference, input_images, description_prompts, cfg):
        self.pipe_inversion = pipe_inversion
        self.pipe_inference = pipe_inference
        self.original_images = [load_im_into_format_from_path(img).convert("RGB") for img in input_images]
        self.description_prompts = description_prompts
        self.load_image = True
        g_cpu = torch.Generator().manual_seed(7865)
        img_size = (512,512)
        VQAE_SCALE = 8
        batch_size = len(input_images)
        latents_size = (batch_size, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
        # Generate noise once and share it
        self.shared_noise = randn_tensor(latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=g_cpu)
        noise = [self.shared_noise.clone() for i in range(cfg.num_inversion_steps)]
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)
        pipe_inversion.scheduler_inference.set_noise_list(noise)
        pipe_inversion.set_progress_bar_config(disable=True)
        pipe_inference.set_progress_bar_config(disable=True)
        self.cfg = cfg
        self.pipe_inversion.cfg = cfg
        self.pipe_inference.cfg = cfg
        self.inv_hp = [2, 0.1, 0.2] # [0, 0.1, 0.2] - for regular DDIM inversion
        self.edit_cfg = 1.0

        # Move models to GPU sequentially
        self.pipe_inversion.to("cuda", torch.float16)
        clear_memory()
        self.pipe_inference.to("cuda", torch.float16)
        clear_memory()

        self.last_latents = self.invert(self.original_images, description_prompts)
        self.original_latents = self.last_latents

    def invert(self, init_images, base_prompts):
        res = self.pipe_inversion(prompt=base_prompts,
                             num_inversion_steps=self.cfg.num_inversion_steps,
                             num_inference_steps=self.cfg.num_inference_steps,
                             image=init_images,
                             guidance_scale=self.cfg.guidance_scale,
                             callback_on_step_end=inversion_callback,
                             strength=self.cfg.inversion_max_step,
                             denoising_start=1.0 - self.cfg.inversion_max_step,
                             inv_hp=self.inv_hp)[0][0]
        clear_memory()
        return res

    def edit(self, target_prompts):
        dummy_mask = torch.ones_like(self.last_latents)
        images = self.pipe_inference(prompt=target_prompts,
                            num_inference_steps=self.cfg.num_inference_steps,
                            negative_prompt=[""] * len(target_prompts),
                            callback_on_step_end=inference_callback,
                            image=self.last_latents,
                            strength=self.cfg.inversion_max_step,
                            denoising_start=1.0 - self.cfg.inversion_max_step,
                            mask_image=dummy_mask,
                            guidance_scale=self.edit_cfg).images
        clear_memory()
        return images

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = (512, 512)
    scheduler_class = MyEulerAncestralDiscreteScheduler
    
    # Load models sequentially with half precision
    pipe_inversion = SDXLDDIMPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", 
        use_safetensors=True,
        safety_checker=None, 
        cache_dir="/inputs/huggingface_cache",
        torch_dtype=torch.float16
    )
    clear_memory()
    
    pipe_inference = StableDiffusionXLInpaintPipeline.from_pretrained(  
        "stabilityai/sdxl-turbo", 
        use_safetensors=True,
        safety_checker=None,
        cache_dir="/inputs/huggingface_cache",
        torch_dtype=torch.float16
    )
    clear_memory()
    
    pipe_inference.scheduler = scheduler_class.from_config(pipe_inference.scheduler.config)
    pipe_inversion.scheduler = scheduler_class.from_config(pipe_inversion.scheduler.config)
    pipe_inversion.scheduler_inference = scheduler_class.from_config(pipe_inference.scheduler.config)

    config = RunConfig(num_inference_steps=100,
                       num_inversion_steps=100,
                       guidance_scale=1.0,
                       inversion_max_step=1.0)

    # Example with two images
    input_images = [
        "/colab/NewtonRaphsonInversion/example_images/background.png",
        "/colab/NewtonRaphsonInversion/example_images/obj_back_mask.png"  # Replace with your second image path
    ]
    description_prompts = ['', '']  # Add your prompts here
    
    # Create editor instance
    editor = ImageEditorDemo(pipe_inversion, pipe_inference, input_images, description_prompts, config)
    clear_memory()

    # Perform edit
    editing_prompts = ["", ""]  # Add your editing prompts here
    edited_images = editor.edit(editing_prompts)
    
    # Save and display results
    for i, img in enumerate(edited_images):
        img.save(f"/colab/NewtonRaphsonInversion/output/edited_image_{i}.png")
    
    # Final cleanup
    clear_memory()