"""
DDIM Inversion for Stable Diffusion XL Object Insert Pipeline

This module provides DDIM (Denoising Diffusion Implicit Models) inversion functionality
specifically adapted for the StableDiffusionXLOBJInsertPipeline. It replaces the original
DDPM inversion with deterministic DDIM inversion for better controllability and editing.

Key differences from original DDPM inversion:
1. Deterministic inversion (eta=0 by default)
2. Compatible with SDXL dual text encoders
3. Supports pooled embeddings and additional time IDs
4. Optimized for object insertion workflows
5. Better error accumulation handling

Based on:
- DDIM: Denoising Diffusion Implicit Models (Song et al., 2020)
- DDPM Inversion techniques for image editing
- Stable Diffusion XL architecture
"""

import abc
import torch
from torch import inference_mode
from tqdm import tqdm
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np


LOW_RESOURCE = True


def invert_ddim(
    x0: torch.Tensor, 
    pipe, 
    prompt_src: str = "", 
    prompt_2_src: str = None,
    num_diffusion_steps: int = 50, 
    cfg_scale_src: float = 7.5, 
    eta: float = 0.0,
    prog_bar: bool = True,
    generator: Optional[torch.Generator] = None,
    capture_noise_patterns: bool = True
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Inverts a real image using DDIM inversion for SDXL pipelines.
    
    Args:
        x0: Input image tensor (PIL Image or torch.Tensor)
        pipe: StableDiffusionXLOBJInsertPipeline instance
        prompt_src: Primary text prompt for inversion
        prompt_2_src: Secondary text prompt for SDXL's second text encoder
        num_diffusion_steps: Number of inversion steps
        cfg_scale_src: Classifier-free guidance scale
        eta: DDIM eta parameter (0.0 for deterministic inversion)
        prog_bar: Whether to show progress bar
        generator: Random generator for reproducibility
        capture_noise_patterns: Whether to capture noise patterns (zs)
        
    Returns:
        Tuple of (inverted_latent, intermediate_latents, noise_patterns)
        - inverted_latent: Final inverted latent tensor
        - intermediate_latents: List of intermediate latent states
        - noise_patterns: List of noise patterns (zs) if capture_noise_patterns=True, else None
    """
    # Set timesteps
    pipe.scheduler.set_timesteps(num_diffusion_steps)
    
    # Encode image to latent space
    with inference_mode():
        if hasattr(x0, 'shape') and len(x0.shape) == 4 and x0.shape[1] == 4:
            # Already in latent space
            w0 = x0
        else:
            # Encode image to latent
            if not torch.is_tensor(x0):
                x0 = pipe.image_processor.preprocess(x0)
            
            x0 = x0.to(device=pipe.device, dtype=pipe.vae.dtype)
            w0 = pipe.vae.encode(x0).latent_dist.sample(generator=generator)
            w0 = pipe.vae.config.scaling_factor * w0
    
    # Perform DDIM inversion
    wt, wts, zs = ddim_inversion_forward_process(
        pipe, w0, 
        eta=eta, 
        prompt=prompt_src,
        prompt_2=prompt_2_src,
        cfg_scale=cfg_scale_src,
        prog_bar=prog_bar, 
        num_inference_steps=num_diffusion_steps,
        generator=generator,
        capture_noise_patterns=capture_noise_patterns
    )
    
    return wt, wts, zs


def ddim_inversion_forward_process(
    model, 
    x0: torch.Tensor,
    eta: float = 1.0,
    prog_bar: bool = False,
    prompt: str = "",
    prompt_2: Optional[str] = None,
    cfg_scale: float = 7.5,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    capture_noise_patterns: bool = True
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    DDIM inversion forward process adapted for SDXL.
    
    This function performs deterministic inversion through the diffusion process,
    storing intermediate latent states and optionally noise patterns for later reconstruction.
    
    Args:
        model: SDXL pipeline instance
        x0: Input latent tensor
        eta: DDIM eta parameter (0.0 for deterministic)
        prog_bar: Whether to show progress bar
        prompt: Primary text prompt
        prompt_2: Secondary text prompt for SDXL
        cfg_scale: Classifier-free guidance scale
        num_inference_steps: Number of inversion steps
        generator: Random generator
        capture_noise_patterns: Whether to capture noise patterns (zs)
        
    Returns:
        Tuple of (final_latent, intermediate_latents, noise_patterns)
    """
    
    # Encode prompts using SDXL dual text encoders
    if prompt or prompt_2:
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = encode_prompt_sdxl(
            model, prompt, prompt_2
        )
    else:
        # Empty prompts
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = encode_prompt_sdxl(
            model, "", ""
        )
    
    # Get timesteps
    timesteps = model.scheduler.timesteps.to(model.device)
    
    # Initialize storage for intermediate latents and noise patterns
    latent_shape = x0.shape
    intermediate_latents = []
    noise_patterns = [] if capture_noise_patterns else None
    
    # Current latent state
    xt = x0.clone()
    
    # Progress bar setup
    iterator = tqdm(reversed(timesteps), desc="DDIM Inversion") if prog_bar else reversed(timesteps)
    
    # DDIM inversion loop
    for i, t in enumerate(iterator):
        # Store current latent
        intermediate_latents.append(xt.clone())
        
        # Prepare latent model input
        latent_model_input = torch.cat([xt] * 2) if cfg_scale > 1.0 else xt
        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
        
        # Prepare added conditioning for SDXL
        added_cond_kwargs = {}
        if hasattr(model, '_get_add_time_ids'):
            # Get original size, target size for SDXL conditioning
            original_size = (latent_shape[-2] * model.vae_scale_factor, 
                           latent_shape[-1] * model.vae_scale_factor)
            target_size = original_size
            
            add_time_ids = model._get_add_time_ids(
                original_size, (0, 0), target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=pooled_prompt_embeds.shape[-1]
            )
            
            if cfg_scale > 1.0:
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
            
            added_cond_kwargs = {
                "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0) if cfg_scale > 1.0 else pooled_prompt_embeds,
                "time_ids": add_time_ids.to(model.device)
            }
        
        # Predict noise
        with torch.no_grad():
            noise_pred = model.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) if cfg_scale > 1.0 else prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        
        # Apply classifier-free guidance
        if cfg_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        
        # Capture noise pattern if required
        if capture_noise_patterns and i < len(timesteps) - 1:
            # Calculate next timestep for noise pattern calculation
            next_timestep = min(
                model.scheduler.config.num_train_timesteps - 2,
                t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            )
            
            # Extract noise pattern similar to DDPM approach
            alpha_prod_t = model.scheduler.alphas_cumprod[t]
            alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep] if next_timestep >= 0 else model.scheduler.final_alpha_cumprod
            
            # Calculate predicted original sample (x0)
            pred_original_sample = (xt - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            
            # Calculate the deterministic direction
            pred_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
            
            # Calculate what the next sample should be deterministically
            mu_xt = alpha_prod_t_next ** 0.5 * pred_original_sample + pred_sample_direction
            
            # For DDIM inversion with eta=0, we extract the "implicit noise" that would be needed
            # to reach the next latent from the current prediction
            if eta > 0.0:
                variance = ((1 - alpha_prod_t_next) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_next)
                # Get the next latent (we need to peek ahead)
                next_xt = ddim_inversion_step(model, noise_pred, t, xt, eta, generator)
                
                # Extract the noise pattern that was used
                z = (next_xt - mu_xt) / (eta * variance ** 0.5)
                noise_patterns.append(z.clone())
            else:
                # For deterministic DDIM (eta=0), we store zero noise
                noise_patterns.append(torch.zeros_like(xt))
        elif capture_noise_patterns:
            # For the last timestep, append zero noise
            noise_patterns.append(torch.zeros_like(xt))
        
        # DDIM inversion step
        xt = ddim_inversion_step(model, noise_pred, t, xt, eta, generator)
    
    # Add final latent
    intermediate_latents.append(xt.clone())
    
    return xt, intermediate_latents, noise_patterns


def ddim_inversion_step(
    model, 
    noise_pred: torch.Tensor, 
    timestep: int, 
    sample: torch.Tensor, 
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Perform a single DDIM inversion step.
    
    Args:
        model: The diffusion model
        noise_pred: Predicted noise from the model
        timestep: Current timestep
        sample: Current sample (latent)
        eta: DDIM eta parameter (0.0 for deterministic)
        generator: Random generator
        
    Returns:
        Next sample in the inversion process
    """
    # Get scheduler parameters
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    
    # Calculate next timestep
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    )
    
    alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep] if next_timestep >= 0 else model.scheduler.final_alpha_cumprod
    
    # Calculate predicted original sample (x0)
    pred_original_sample = (sample - (1 - alpha_prod_t) ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
    
    # Calculate predicted noise
    pred_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
    
    # Calculate next sample (DDIM inversion)
    next_sample = alpha_prod_t_next ** 0.5 * pred_original_sample + pred_sample_direction
    
    # Add stochastic component if eta > 0
    if eta > 0.0:
        variance = ((1 - alpha_prod_t_next) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_next)
        sigma = eta * variance ** 0.5
        
        if generator is not None:
            device = sample.device
            noise = torch.randn(sample.shape, generator=generator, device=device, dtype=sample.dtype)
        else:
            noise = torch.randn_like(sample)
            
        next_sample = next_sample + sigma * noise
    
    return next_sample


def encode_prompt_sdxl(
    model, 
    prompt: str, 
    prompt_2: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode text prompts using SDXL's dual text encoder architecture.
    
    Returns:
        Tuple of (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
    """
    device = device or model.device
    
    # Set default values
    prompt_2 = prompt_2 or prompt
    negative_prompt = negative_prompt or ""
    negative_prompt_2 = negative_prompt_2 or negative_prompt
    
    # Prepare prompts
    if isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]
    else:
        batch_size = len(prompt)
    
    if isinstance(prompt_2, str):
        prompt_2 = [prompt_2] * batch_size
    
    # Tokenizers and text encoders
    tokenizers = [model.tokenizer, model.tokenizer_2] if model.tokenizer is not None else [model.tokenizer_2]
    text_encoders = [model.text_encoder, model.text_encoder_2] if model.text_encoder is not None else [model.text_encoder_2]
    
    # Encode prompts
    prompt_embeds_list = []
    prompts = [prompt, prompt_2]
    
    for prompt_batch, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        # Tokenize
        text_inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode
        with torch.no_grad():
            prompt_embeds = text_encoder(
                text_inputs.input_ids.to(device),
                output_hidden_states=True,
            )
        
        # Extract pooled embeddings (from final encoder only)
        pooled_prompt_embeds = prompt_embeds[0]
        
        # Extract hidden states (penultimate layer)
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)
    
    # Concatenate embeddings from both encoders
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    
    # Handle negative prompts for classifier-free guidance
    if do_classifier_free_guidance:
        # Encode negative prompts
        negative_prompt_embeds_list = []
        negative_prompts = [negative_prompt, negative_prompt_2]
        
        for neg_prompt_batch, tokenizer, text_encoder in zip(negative_prompts, tokenizers, text_encoders):
            # Handle string/list conversion
            if isinstance(neg_prompt_batch, str):
                neg_prompt_batch = [neg_prompt_batch] * batch_size
            
            # Tokenize negative prompts
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                neg_prompt_batch,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Encode negative prompts
            with torch.no_grad():
                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
            
            # Extract pooled embeddings (from final encoder only)
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            
            # Extract hidden states
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            negative_prompt_embeds_list.append(negative_prompt_embeds)
        
        # Concatenate negative embeddings
        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
    else:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def ddim_reconstruction(
    model,
    latent: torch.Tensor,
    intermediate_latents: List[torch.Tensor],
    prompt: str = "",
    prompt_2: Optional[str] = None,
    cfg_scale: float = 7.5,
    eta: float = 0.0,
    prog_bar: bool = True,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Reconstruct image from inverted latents using DDIM sampling.
    
    Args:
        model: The diffusion model
        latent: Starting latent (typically the most noisy one from inversion)
        intermediate_latents: List of intermediate latents from inversion
        prompt: Text prompt for reconstruction
        prompt_2: Secondary prompt for SDXL
        cfg_scale: Classifier-free guidance scale
        eta: DDIM eta parameter
        prog_bar: Show progress bar
        generator: Random generator
        
    Returns:
        Reconstructed latent
    """
    # Encode prompts
    if prompt or prompt_2:
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = encode_prompt_sdxl(
            model, prompt, prompt_2
        )
    else:
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = encode_prompt_sdxl(
            model, "", ""
        )
    
    # Get timesteps for reconstruction (reverse order)
    timesteps = model.scheduler.timesteps.to(model.device)
    
    # Start from the most noisy latent
    current_latent = latent.clone()
    
    # Progress bar
    iterator = tqdm(timesteps, desc="DDIM Reconstruction") if prog_bar else timesteps
    
    # DDIM sampling loop
    for i, t in enumerate(iterator):
        # Prepare model input
        latent_model_input = torch.cat([current_latent] * 2) if cfg_scale > 1.0 else current_latent
        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
        
        # Prepare SDXL conditioning
        added_cond_kwargs = {}
        if hasattr(model, '_get_add_time_ids'):
            original_size = (current_latent.shape[-2] * model.vae_scale_factor, 
                           current_latent.shape[-1] * model.vae_scale_factor)
            target_size = original_size
            
            add_time_ids = model._get_add_time_ids(
                original_size, (0, 0), target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=pooled_prompt_embeds.shape[-1]
            )
            
            if cfg_scale > 1.0:
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
            
            added_cond_kwargs = {
                "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0) if cfg_scale > 1.0 else pooled_prompt_embeds,
                "time_ids": add_time_ids.to(model.device)
            }
        
        # Predict noise
        with torch.no_grad():
            noise_pred = model.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) if cfg_scale > 1.0 else prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        
        # Apply CFG
        if cfg_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        
        # DDIM step
        current_latent = model.scheduler.step(
            noise_pred, t, current_latent, eta=eta, generator=generator
        ).prev_sample
    
    return current_latent


# Attention control classes for editing (adapted from original)
class AttentionControl(abc.ABC):
    """Base class for attention control during diffusion sampling."""
    
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    """Store attention maps during diffusion sampling."""
    
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_attention_control_sdxl(model, controller):
    """
    Register attention control for SDXL models.
    Adapted to work with SDXL's attention architecture.
    """
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            batch_size, sequence_length, _ = hidden_states.shape
            
            # Get query, key, value
            query = self.to_q(hidden_states)
            
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
            
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            # Reshape for multi-head attention
            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)
            
            # Compute attention scores
            attention_probs = self.get_attention_scores(query, key, attention_mask)
            
            # Apply attention control
            if controller is not None:
                attention_probs = controller(attention_probs, is_cross, place_in_unet)
            
            # Apply attention to values
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)
            
            # Linear projection and dropout
            hidden_states = to_out(hidden_states)
            
            return hidden_states

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':  # SDXL uses 'Attention' class
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


# Utility functions for object insertion workflows
def prepare_inversion_inputs(
    image, 
    mask, 
    pipeline,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for inversion in object insertion context.
    
    Args:
        image: Input image (PIL or tensor)
        mask: Object mask (PIL or tensor) 
        pipeline: SDXL pipeline instance
        device: Target device
        
    Returns:
        Dictionary with processed inputs
    """
    device = device or pipeline.device
    
    # Process image
    if not torch.is_tensor(image):
        image = pipeline.image_processor.preprocess(image)
    image = image.to(device)
    
    # Process mask
    if not torch.is_tensor(mask):
        mask = pipeline.mask_processor.preprocess(mask)
    mask = mask.to(device)
    
    # Create masked image
    masked_image = image * (mask < 0.5)
    
    return {
        "image": image,
        "mask": mask, 
        "masked_image": masked_image
    }


def compute_inversion_metrics(
    original_latent: torch.Tensor,
    reconstructed_latent: torch.Tensor
) -> Dict[str, float]:
    """
    Compute quality metrics for inversion.
    
    Args:
        original_latent: Original image latent
        reconstructed_latent: Reconstructed latent from inversion
        
    Returns:
        Dictionary with metric values
    """
    # MSE
    mse = torch.nn.functional.mse_loss(original_latent, reconstructed_latent).item()
    
    # PSNR
    psnr = -10 * torch.log10(torch.tensor(mse)).item()
    
    # Cosine similarity
    original_flat = original_latent.flatten()
    reconstructed_flat = reconstructed_latent.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        original_flat.unsqueeze(0), reconstructed_flat.unsqueeze(0)
    ).item()
    
    return {
        "mse": mse,
        "psnr": psnr,
        "cosine_similarity": cos_sim
    }


# Convenience functions for compatibility with existing DDPM workflow
def invert_with_noise_patterns(
    x0: torch.Tensor,
    pipe,
    prompt_src: str = "",
    num_diffusion_steps: int = 100,
    cfg_scale_src: float = 3.5,
    eta: float = 0.0
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Convenience function that mimics the original DDPM invert() interface.
    
    This function provides backward compatibility with the original DDPM inversion workflow
    by returning (zs, wts) in the same format as the original implementation.
    
    Args:
        x0: Input image or latent tensor
        pipe: SDXL pipeline instance  
        prompt_src: Text prompt for inversion
        num_diffusion_steps: Number of diffusion steps
        cfg_scale_src: Classifier-free guidance scale
        eta: DDIM eta parameter
        
    Returns:
        Tuple of (zs, wts) where:
        - zs: List of noise patterns (equivalent to original DDPM zs)
        - wts: List of intermediate latents (equivalent to original DDPM wts)
    """
    # Perform DDIM inversion with noise pattern capture
    wt, wts, zs = invert_ddim(
        x0=x0,
        pipe=pipe,
        prompt_src=prompt_src,
        num_diffusion_steps=num_diffusion_steps,
        cfg_scale_src=cfg_scale_src,
        eta=eta,
        prog_bar=True,
        capture_noise_patterns=True
    )
    
    # Return in the same format as original DDPM inversion
    return zs if zs is not None else [], wts


def sample_xts_from_x0_ddim(model, x0, num_inference_steps=50):
    """
    Sample intermediate latents from x0 for DDIM (deterministic version).
    
    This function creates a trajectory of latents from x0 to pure noise
    following the DDIM deterministic path. Useful for initialization
    and trajectory planning.
    
    Args:
        model: SDXL pipeline instance
        x0: Starting latent tensor
        num_inference_steps: Number of steps
        
    Returns:
        List of intermediate latent tensors
    """
    model.scheduler.set_timesteps(num_inference_steps)
    timesteps = model.scheduler.timesteps.to(model.device)
    
    # Store intermediate states
    xts = []
    xt = x0.clone()
    
    for t in reversed(timesteps):
        xts.append(xt.clone())
        
        # Get noise schedule parameters
        alpha_prod_t = model.scheduler.alphas_cumprod[t]
        
        # Sample noise
        noise = torch.randn_like(xt)
        
        # Add noise according to schedule (forward diffusion)
        xt = alpha_prod_t**0.5 * x0 + (1 - alpha_prod_t)**0.5 * noise
    
    xts.append(xt)  # Final noisy state
    return xts


def get_variance_ddim(model, timestep):
    """
    Calculate variance for DDIM sampling at given timestep.
    
    Args:
        model: SDXL pipeline instance
        timestep: Current timestep
        
    Returns:
        Variance value
    """
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def forward_step_ddim(model, model_output, timestep, sample, eta=0.0, generator=None):
    """
    Single forward step for DDIM sampling.
    
    Args:
        model: SDXL pipeline instance
        model_output: Predicted noise from UNet
        timestep: Current timestep
        sample: Current sample
        eta: DDIM eta parameter
        generator: Random generator
        
    Returns:
        Next sample
    """
    return ddim_inversion_step(model, model_output, timestep, sample, eta, generator)


# Export main functions for easy importing
__all__ = [
    'invert_ddim',
    'ddim_inversion_forward_process', 
    'ddim_reconstruction',
    'encode_prompt_sdxl',
    'invert_with_noise_patterns',
    'AttentionControl',
    'AttentionStore',
    'register_attention_control_sdxl',
    'prepare_inversion_inputs',
    'compute_inversion_metrics'
]
