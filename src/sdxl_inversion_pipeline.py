# Code is based on ReNoise https://github.com/garibida/ReNoise-Inversion

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import deprecate

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipelineOutput,
)

# Import retrieve_timesteps from the correct location
try:
    from diffusers.schedulers.scheduling_utils import retrieve_timesteps
except ImportError:
    try:
        from diffusers.utils import retrieve_timesteps
    except ImportError:
        # Fallback implementation for older versions
        def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None):
            if timesteps is not None:
                return timesteps, len(timesteps)
            scheduler.set_timesteps(num_inference_steps, device=device)
            return scheduler.timesteps, num_inference_steps

from src.eunms import Epsilon_Update_Type


def _backward_ddim(x_tm1, alpha_t, alpha_tm1, eps_xt):
    """
    let a = alpha_t, b = alpha_{t - 1}
    We have a > b,
    x_{t} - x_{t - 1} = sqrt(a) ((sqrt(1/b) - sqrt(1/a)) * x_{t-1} + (sqrt(1/a - 1) - sqrt(1/b - 1)) * eps_{t-1})
    From https://arxiv.org/pdf/2105.05233.pdf, section F.
    """

    a, b = alpha_t, alpha_tm1
    sa = a ** 0.5
    sb = b ** 0.5

    return sa * ((1 / sb) * x_tm1 + ((1 / a - 1) ** 0.5 - (1 / b - 1) ** 0.5) * eps_xt)


def sample_xts_from_x0(scheduler, x0, num_inference_steps=50, device=None):
    """
    Samples from P(x_1:T|x_0) - generates intermediate latents for the forward diffusion process
    Similar to the DDPM inversion approach
    """
    if device is None:
        device = x0.device
        
    alpha_bar = scheduler.alphas_cumprod.to(device)
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    
    # Get timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    variance_noise_shape = (
        num_inference_steps,
        x0.shape[1],  # channels
        x0.shape[2],  # height
        x0.shape[3]   # width
    )
    
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape, device=device, dtype=x0.dtype)
    
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        # Sample x_t from q(x_t|x_0)
        noise = torch.randn_like(x0)
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) + noise * sqrt_one_minus_alpha_bar[t]
    
    # Concatenate x0 at the end
    xts = torch.cat([xts, x0.unsqueeze(0)], dim=0)
    
    return xts


def get_variance(scheduler, timestep, device=None):
    """
    Compute variance for a given timestep
    """
    if device is None:
        device = timestep.device if hasattr(timestep, 'device') else 'cpu'
        
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    
    return variance


class SDXLDDIMPipeline(StableDiffusionXLImg2ImgPipeline):
    # @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            image: PipelineImageInput = None,
            strength: float = 0.3,
            num_inversion_steps: int = 50,
            timesteps: List[int] = None,
            denoising_start: Optional[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 1.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Tuple[int, int] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            aesthetic_score: float = 6.0,
            negative_aesthetic_score: float = 2.5,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            num_inference_steps: int = 50,
            inv_hp=None,
            return_intermediate_latents: bool = True,
            return_variance_maps: bool = True,
            use_ddim_inversion: bool = False,
            **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            num_inversion_steps,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        denoising_start_fr = 1.0 - denoising_start
        denoising_start = denoising_start

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._denoising_start = denoising_start

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps
        def denoising_value_valid(dnv):
            return isinstance(self.denoising_end, float) and 0 < dnv < 1

        timesteps, num_inversion_steps = retrieve_timesteps(self.scheduler, num_inversion_steps, device, timesteps)
        timesteps_num_inference_steps, num_inference_steps = retrieve_timesteps(self.scheduler_inference,
                                                                                num_inference_steps, device, None)

        timesteps, num_inversion_steps = self.get_timesteps(
            num_inversion_steps,
            strength,
            device,
            denoising_start=self.denoising_start if denoising_value_valid else None,
        )
        # latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # add_noise = True if self.denoising_start is None else False
        # 6. Prepare latent variables
        with torch.no_grad():
            latents = self.prepare_latents(
                image,
                None,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                False,
            )
        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inversion_steps * self.scheduler.order, 0)
        prev_timestep = None

        self._num_timesteps = len(timesteps)
        self.prev_z = torch.clone(latents)
        self.prev_z4 = torch.clone(latents)
        self.z_0 = torch.clone(latents)
        g_cpu = torch.Generator().manual_seed(7865)
        self.noise = randn_tensor(self.z_0.shape, generator=g_cpu, device=self.z_0.device, dtype=self.z_0.dtype)

        # Friendly inversion params
        timesteps_for = reversed(timesteps)
        noise = randn_tensor(latents.shape, generator=g_cpu, device=latents.device, dtype=latents.dtype)
        # latents = latents
        z_T = latents.clone()

        all_latents = [latents.clone()]
        
        # Initialize intermediate latents and variance maps if requested
        xts = None
        zs = None
        if return_intermediate_latents or return_variance_maps:
            if eta > 0:  # Only compute if eta > 0 (stochastic inversion)
                # Pre-compute intermediate latents using forward diffusion
                xts = sample_xts_from_x0(self.scheduler, latents, num_inversion_steps, device)
                
                if return_variance_maps:
                    # Initialize variance maps
                    variance_noise_shape = (
                        num_inversion_steps,
                        latents.shape[1],  # channels
                        latents.shape[2],  # height  
                        latents.shape[3]   # width
                    )
                    zs = torch.zeros(variance_noise_shape, device=device, dtype=latents.dtype)
        
        # Create timestep to index mapping
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i, t in enumerate(timesteps_for):
                idx = t_to_idx[int(t)]

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                # Use pre-computed intermediate latents if available and eta > 0
                if xts is not None and eta > 0:
                    latents = xts[idx][None]

                z_tp1 = self.inversion_step(latents,
                                            t,
                                            prompt_embeds,
                                            added_cond_kwargs,
                                            prev_timestep=prev_timestep,
                                            inv_hp=inv_hp,
                                            z_0=self.z_0)

                # Compute variance if requested and eta > 0
                if return_variance_maps and eta > 0 and zs is not None:
                    # Get noise prediction for variance computation
                    noise_pred = self.unet_pass(latents, t, prompt_embeds, added_cond_kwargs)
                    
                    # Compute variance similar to DDPM inversion
                    alpha_bar = self.scheduler.alphas_cumprod
                    prev_timestep_var = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                    alpha_prod_t_prev = alpha_bar[prev_timestep_var] if prev_timestep_var >= 0 else self.scheduler.final_alpha_cumprod
                    
                    # Predict x0
                    pred_original_sample = (latents - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
                    
                    # Compute variance
                    variance = get_variance(self.scheduler, t, device)
                    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * noise_pred
                    mu_xt = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
                    
                    # Compute and store variance map
                    if idx + 1 < len(xts):
                        xtm1 = xts[idx + 1][None]
                        z = (xtm1 - mu_xt) / (eta * variance ** 0.5)
                        zs[idx] = z.squeeze(0)

                prev_timestep = t
                latents = z_tp1

                all_latents.append(latents.clone())

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    add_neg_time_ids = callback_outputs.pop("add_neg_time_ids", add_neg_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        # Prepare return values based on requested outputs
        return_values = [StableDiffusionXLPipelineOutput(images=image), all_latents]
        
        # if return_intermediate_latents and xts is not None:
        #     return_values.append(xts)
        # elif return_intermediate_latents:
        #     return_values.append(None)
            
        # if return_variance_maps and zs is not None:
        #     # Set the last variance to zero like in DDPM inversion
        #     if zs is not None:
        #         zs[-1] = torch.zeros_like(zs[-1])
        #     return_values.append(zs)
        # elif return_variance_maps:
        #     return_values.append(None)

        return tuple(return_values)

    def get_timestamp_dist(self, z_0, timesteps):
        timesteps = timesteps.to(z_0.device)
        sigma = self.scheduler.sigmas.cuda()[:-1][self.scheduler.timesteps == timesteps]
        z_0 = z_0.reshape(-1, 1)

        def gaussian_pdf(x):
            shape = x.shape
            x = x.reshape(-1, 1)
            all_probs = - 0.5 * torch.pow(((x - z_0) / sigma), 2)
            return all_probs.reshape(shape)

        return gaussian_pdf

    def inversion_step(
            self,
            z_t: torch.tensor,
            t: torch.tensor,
            prompt_embeds,
            added_cond_kwargs,
            prev_timestep: Optional[torch.tensor] = None,
            inv_hp=None,
            z_0=None,
    ) -> torch.tensor:

        n_iters, alpha, lr = inv_hp
        if n_iters == 0:
            # Use pure DDIM inversion when n_iters is 0
            noise_pred = self.unet_pass(z_t, t, prompt_embeds, added_cond_kwargs)
            return self.backward_step(noise_pred, t, z_t, prev_timestep)

        latent = z_t
        best_latent = None
        best_score = torch.inf
        curr_dist = self.get_timestamp_dist(z_0, t)
        for i in range(n_iters):
            latent.requires_grad = True
            noise_pred = self.unet_pass(latent, t, prompt_embeds, added_cond_kwargs)

            next_latent = self.backward_step(noise_pred, t, z_t, prev_timestep)
            f_x = (next_latent - latent).abs() - alpha * curr_dist(next_latent)
            l = f_x.sum()
            score = f_x.mean()

            if score < best_score:
                best_score = score
                best_latent = next_latent.detach()

            l.backward()
            latent = latent - (1 / (64 * 64 * 4)) * (l / latent.grad)
            latent.grad = None
            latent._grad_fn = None

        return best_latent

    def ddim_inversion_step(
            self,
            z_t: torch.tensor,
            t: torch.tensor,
            prompt_embeds,
            added_cond_kwargs,
            prev_timestep: Optional[torch.tensor] = None,
            eta: float = 0.0,
    ) -> torch.tensor:
        """
        Pure DDIM inversion step using the standard DDIM formulas
        """
        # Get noise prediction
        noise_pred = self.unet_pass(z_t, t, prompt_embeds, added_cond_kwargs)
        
        # Get scheduler parameters
        alpha_bar = self.scheduler.alphas_cumprod
        
        # Predict x0 using DDIM formula
        pred_original_sample = (z_t - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
        
        # Compute previous timestep
        if prev_timestep is None:
            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        
        alpha_prod_t_prev = alpha_bar[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        
        # Compute variance
        variance = get_variance(self.scheduler, t)
        
        # Direction pointing to z_t
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * noise_pred
        
        # Final DDIM inversion step
        z_prev = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # Add stochastic component if eta > 0
        if eta > 0:
            noise = torch.randn_like(z_t)
            z_prev = z_prev + eta * variance ** 0.5 * noise
            
        return z_prev

    @torch.no_grad()
    def unet_pass(self, z_t, t, prompt_embeds, added_cond_kwargs):
        latent_model_input = torch.cat([z_t] * 2) if self.do_classifier_free_guidance else z_t
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        return self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def backward_step(self, nosie_pred, t, z_t, prev_timestep):
        extra_step_kwargs = {}
        return self.scheduler.inv_step(nosie_pred, t, z_t, **extra_step_kwargs, return_dict=False)[0].detach()
