import os
import torch
import numpy as np
import PIL
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, UNet2DConditionModel#, StableDiffusionXLImg2ImgPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.utils import load_image
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from src.pipeline_stable_diffusion_xl_obj_insert import StableDiffusionXLOBJInsertPipeline
# from consistory_unet_sdxl import ConsistorySDXLUNet2DConditionModel

class AttentionInjectionProcessor(AttnProcessor2_0):
    def __init__(self, injection_image, mask, injection_weight=0.7, start_injection_timestep=50):
        super().__init__()
        self.injection_image = injection_image  # [1, 4, h, w]
        self.mask = mask  # [1, 1, h, w]
        self.injection_weight = injection_weight
        self.start_injection_timestep = start_injection_timestep
        self.current_timestep = -1  # Initialize timestep

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # if input_ndim == 4:
        #     batch_size, channel, height, width = hidden_states.shape
        #     hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Check if injection should be applied based on the current timestep
        apply_injection = (
            hasattr(self, 'injection_image')
            and self.injection_image is not None
            and self.current_timestep != -1  # Ensure timestep has been set
            and self.current_timestep <= self.start_injection_timestep
        )

        # --- Debug: Visualize Attention Map ---
        # Save attention map at a specific timestep for debugging
        # You might want to make this conditional on a debug flag or specific conditions
        if 0 < self.current_timestep <= self.start_injection_timestep and attention_probs.shape[-2] == attention_probs.shape[-1]: # Only self-attention
             try:
                 # Reshape attention_probs: [batch*heads, seq_len, seq_len] -> [batch, heads, seq_len, seq_len]
                 num_heads = attn.heads
                 seq_len = attention_probs.shape[-1]
                 batch_size = attention_probs.shape[0] // num_heads
                 attn_map_reshaped = attention_probs.view(batch_size, num_heads, seq_len, seq_len)

                 # Select the map for the first batch item and first head
                 attn_map_to_plot = attn_map_reshaped[0, 0].detach().cpu().numpy()
                 h = w = int(np.sqrt(seq_len))
                 attn_map_to_plot = attn_map_to_plot.reshape(h, w, h, w) # Reshape if needed for spatial viz

                 # Example: Visualize the attention from the center pixel to all others
                 center_idx = h // 2 * w + w // 2
                 center_attn = attn_map_reshaped[0, 0, center_idx, :].detach().cpu().numpy().reshape(h, w)


                 save_dir = "/colab/consistory/out/debug_attention_maps"
                 os.makedirs(save_dir, exist_ok=True)
                 save_path = os.path.join(save_dir, f"attn_map_ts{self.current_timestep}_head0.png")

                 plt.figure(figsize=(10, 5))
                 plt.subplot(1, 2, 1)
                 plt.imshow(attn_map_to_plot.mean(axis=(0,1))) # Example: Mean attention over query pixels
                 plt.title(f"Mean Attention (t={self.current_timestep}, Head 0)")
                 plt.colorbar()

                 plt.subplot(1, 2, 2)
                 plt.imshow(center_attn)
                 plt.title(f"Center Pixel Attention (t={self.current_timestep}, Head 0)")
                 plt.colorbar()

                 plt.tight_layout()
                 plt.savefig(save_path)
                 plt.close()
                 print(f"Saved attention map debug image to: {save_path}")

                 # Optionally save the raw tensor
                 # torch.save(attention_probs.detach().cpu(), os.path.join(save_dir, f"attn_probs_ts{self.current_timestep}.pt"))

             except ImportError:
                 print("Matplotlib not found. Skipping attention map visualization.")
             except Exception as e:
                 print(f"Error during attention map visualization: {e}")
        # --- End Debug ---
        if apply_injection:
            # Reshape mask to match attention map dimensions
            h = w = int(np.sqrt(attention_probs.shape[-1]))

            # Only apply to self-attention (not cross-attention)
            if attention_probs.shape[-2] == attention_probs.shape[-1]:
                # Resize mask to match hidden state dimensions
                mask_resized = F.interpolate(self.mask, size=(h, w), mode='bilinear')
                mask_flat = mask_resized.view(mask_resized.shape[0], mask_resized.shape[1], -1)
                mask_flat = mask_flat.transpose(1, 2)  # [B, h*w, 1]

                # Apply the mask to attention weights
                # Focus attention on the masked region
                # mask_weights = mask_flat * self.injection_weight
                mask_weights = self.mask.squeeze(0).squeeze(0) * self.injection_weight
                # Ensure broadcasting works correctly if batch sizes differ
                if attention_probs.shape[0] != mask_weights.shape[0]:
                    # Assuming mask_weights has batch size 1, repeat it
                    mask_weights = mask_weights.repeat(attention_probs.shape[0] // mask_weights.shape[0], 1, 1)

                # attention_probs = attention_probs * (1 - mask_weights) + mask_weights

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SDXLObjectInserter:
    def __init__(self, device="cuda", model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        """Initialize the object inserter with SDXL model"""
        self.device = device
        self.model_id = model_id
        self.start_injection_timestep = 50  # Default value, can be overridden in insert_object
        sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
        float_type = torch.float16 if device == "cuda" else torch.float32
        self.unet = ConsistorySDXLUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=float_type)
        print(f"Loading SDXL model from {model_id}...")
        self.pipeline = StableDiffusionXLOBJInsertPipeline.from_pretrained(
            model_id,
            unet=self.unet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None
        ).to(device)

        # Use DDIM scheduler for better control
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

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

    def inject_attention_processors(self, unet, injection_image, mask):
        """Inject custom attention processors into upper layers of UNet"""
        # Convert inputs to proper format for the attention processor
        injection_image = torch.from_numpy(np.array(injection_image)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        injection_image = injection_image.to(torch.float16 if self.device == "cuda" else torch.float32) / 127.5 - 1.0

        # Process mask
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = mask_tensor.to(torch.float16 if self.device == "cuda" else torch.float32)

        # We'll only inject into the last few blocks (upper layers)
        attention_processors = {}

        # Instead of setting specific attention processors, we'll modify the existing ones
        # First get all the current attention processors
        attention_processors = unet.attn_processors

        # Target specific blocks to ensure we're getting the correct layers
        target_blocks = ["up_blocks.0", "up_blocks.1"]

        print("Injecting attention processors into upper UNet blocks...")
        modified_count = 0

        # Create a copy of the processor keys to avoid modifying during iteration
        processor_keys = list(attention_processors.keys())

        # Only modify the processors for the targeted blocks
        for key in processor_keys:
            # Only process attention modules in our target blocks
            if any(block in key for block in target_blocks):
                # Specifically target self-attention (attn1)
                if "attn1" in key:
                    print(f"Injecting attention processor into {key}")
                    attention_processors[key] = AttentionInjectionProcessor(
                        injection_image=injection_image,
                        mask=mask_tensor,
                        injection_weight=0.7,
                        start_injection_timestep=self.start_injection_timestep  # Pass the threshold
                    )
                    modified_count += 1

        print(f"Modified {modified_count} attention processors out of {len(attention_processors)} total")

        # Set the attention processors
        unet.set_attn_processor(attention_processors)
        return unet

    def _update_timestep_callback(self, step_index: int, timestep: int, latents: torch.FloatTensor):
        """Callback function to update the current timestep in attention processors."""
        for processor in self.pipeline.unet.attn_processors.values():
            if isinstance(processor, AttentionInjectionProcessor):
                processor.current_timestep = timestep

    def insert_object(
        self,
        background_path,
        object_path,
        position=(0, 0),
        object_scale=1.0,
        boundary_size=20,
        prompt="a seamless, realistic composite image",
        negative_prompt="unrealistic, seams, bad composition, deformed",
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.6,
        start_injection_timestep=50,  # Add parameter to control start step
        output_path="/colab/consistory/assets/output.png"
    ):
        """Insert object into background image using attention injection in SDXL"""
        self.start_injection_timestep = start_injection_timestep  # Store the threshold

        # Prepare composite image and mask
        composite_img, object_mask, boundary_mask, background_img, object_img = self.prepare_images(
            background_path, object_path, position, object_scale, boundary_size
        )

        # Convert composite to RGB
        if composite_img.mode == "RGBA":
            composite_img = composite_img.convert("RGB")

        # Prepare inputs for the pipeline
        init_image = composite_img.resize((1024, 1024), Image.LANCZOS)
        init_bg_image = background_img.resize((1024, 1024), Image.LANCZOS)
        init_object_image = object_img.resize((1024, 1024), Image.LANCZOS)


        # Resize masks to match model resolution
        object_mask_resized = cv2.resize(object_mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        boundary_mask_resized = cv2.resize(boundary_mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        # save the mask for debugging
        # Ensure output directory exists
        os.makedirs(os.path.dirname("/colab/consistory/out/boundary_mask.png"), exist_ok=True)
        cv2.imwrite("/colab/consistory/out/boundary_mask.png", boundary_mask_resized * 255)

        # save init_image for debugging
        init_image.save("/colab/consistory/out/init_image.png")

        # # Inject attention processors
        # self.pipeline.unet = self.inject_attention_processors(
        #     self.pipeline.unet, init_image, boundary_mask_resized
        # )

        # Run the pipeline
        inverted_image, inversion_noise = self.pipeline.ddim_inverse('', [init_bg_image, init_object_image])
        print(f"Generating seamless insertion... Injecting attention from timestep {self.start_injection_timestep}")
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            callback=self._update_timestep_callback,  # Pass the callback
            callback_steps=1  # Call callback at every step
        ).images[0]

        # Save the result
        result.save(output_path)
        print(f"Result saved to {output_path}")

        # Reset attention processors to default
        self.pipeline.unet.set_attn_processor(AttnProcessor2_0())

        return result


def main():
    """Example usage of the SDXL object inserter"""
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the object inserter
    inserter = SDXLObjectInserter(device=device)

    # Example paths
    background_path = "/colab/consistory/assets/background.png"
    object_path = "/colab/consistory/assets/obj_back_mask.png"  # Must have transparency

    # Insert the object
    result = inserter.insert_object(
        background_path=background_path,
        object_path=object_path,
        position=(200, 150),  # Position to place the object (x, y)
        object_scale=0.2,     # Scale factor for the object\
        boundary_size=5,
        # Prompt for the generation
        prompt="",
        negative_prompt="",
        num_inference_steps=30, # Number of diffusion steps
        guidance_scale=7.5, # How much to guide the model towards the prompt (0-20)
        strength=0.25,         # How much noise is added to modify the original composite (0-1)
        start_injection_timestep=500,  # Specify the timestep to start injection # TODO: adjust to to inference steps 
        output_path="/colab/consistory/out/output_t.png"  # Changed output name
    )


if __name__ == "__main__":
    main()