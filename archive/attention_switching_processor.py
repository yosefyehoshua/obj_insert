import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
import xformers.ops


class DDIMObjectInsertionAttnProcessor:
    """
    Attention processor for DDIM-based object insertion that switches between
    foreground, background, and noise attention maps based on boundary masks.
    
    This implements the attention manipulation strategy from the rules:
    - Use stored noise latents to re-generate input images
    - Replace keys and queries at boundary regions with those from noise
    - Preserve original keys/queries in non-masked regions
    """
    
    def __init__(
        self,
        place_in_unet: str,
        attnstore=None,
        fg_noise_latents: Optional[List[torch.Tensor]] = None,
        bg_noise_latents: Optional[List[torch.Tensor]] = None,
        boundary_mask: Optional[torch.Tensor] = None,
        blend_strength: float = 0.8,
        use_soft_blending: bool = True
    ):
        self.place_in_unet = place_in_unet
        self.attnstore = attnstore
        self.fg_noise_latents = fg_noise_latents or []
        self.bg_noise_latents = bg_noise_latents or []
        self.boundary_mask = boundary_mask
        self.blend_strength = blend_strength
        self.use_soft_blending = use_soft_blending
        self.attention_op = None
        
        # Track current timestep for accessing correct noise latents
        self.current_timestep_idx = 0
        
    def set_noise_latents(self, fg_latents: List[torch.Tensor], bg_latents: List[torch.Tensor]):
        """Set the stored noise latents from DDIM inversion."""
        self.fg_noise_latents = fg_latents
        self.bg_noise_latents = bg_latents
        
    def set_boundary_mask(self, mask: torch.Tensor):
        """Set the boundary mask for attention switching."""
        self.boundary_mask = mask
        
    def update_timestep(self, timestep_idx: int):
        """Update current timestep index for accessing correct noise latents."""
        self.current_timestep_idx = timestep_idx
        
    def get_noise_features(self, attn: Attention, timestep_idx: int, device: torch.device):
        """
        Generate keys and queries from stored noise latents for the current timestep.
        """
        if (timestep_idx >= len(self.fg_noise_latents) or 
            timestep_idx >= len(self.bg_noise_latents)):
            return None, None, None, None, None, None
            
        # Get noise latents for current timestep
        fg_noise = self.fg_noise_latents[timestep_idx].to(device)
        bg_noise = self.bg_noise_latents[timestep_idx].to(device)
        
        # Generate keys and queries from noise latents
        # Reshape noise to match attention input format
        batch_size, channels, height, width = fg_noise.shape
        fg_noise_flat = fg_noise.view(batch_size, channels, height * width).transpose(1, 2)
        bg_noise_flat = bg_noise.view(batch_size, channels, height * width).transpose(1, 2)
        
        # Generate attention features from noise
        args = () if USE_PEFT_BACKEND else (1.0,)
        
        fg_noise_q = attn.to_q(fg_noise_flat, *args)
        fg_noise_k = attn.to_k(fg_noise_flat, *args)
        fg_noise_v = attn.to_v(fg_noise_flat, *args)
        
        bg_noise_q = attn.to_q(bg_noise_flat, *args)
        bg_noise_k = attn.to_k(bg_noise_flat, *args)
        bg_noise_v = attn.to_v(bg_noise_flat, *args)
        
        return fg_noise_q, fg_noise_k, fg_noise_v, bg_noise_q, bg_noise_k, bg_noise_v
        
    def apply_boundary_mask_blending(
        self, 
        original_q: torch.Tensor,
        original_k: torch.Tensor, 
        original_v: torch.Tensor,
        noise_q: torch.Tensor,
        noise_k: torch.Tensor,
        noise_v: torch.Tensor,
        mask: torch.Tensor,
        height: int,
        width: int
    ):
        """
        Apply boundary mask to blend original and noise attention features.
        
        Args:
            original_q/k/v: Original query/key/value tensors
            noise_q/k/v: Noise-derived query/key/value tensors  
            mask: Boundary mask [H, W]
            height, width: Spatial dimensions
        """
        # Resize mask to match attention resolution
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            
        # Apply soft edges if enabled
        if self.use_soft_blending:
            # Apply Gaussian blur for soft boundaries
            kernel_size = max(3, min(height, width) // 16)
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask = F.conv2d(
                mask.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, kernel_size, kernel_size, device=mask.device) / (kernel_size ** 2),
                padding=kernel_size // 2
            ).squeeze(0).squeeze(0)
        
        # Flatten mask for attention dimensions
        mask_flat = mask.view(-1).unsqueeze(0).unsqueeze(-1)  # [1, H*W, 1]
        
        # Expand mask to match batch and head dimensions
        batch_size = original_q.shape[0]
        mask_expanded = mask_flat.expand(batch_size, -1, original_q.shape[-1])
        
        # Blend features based on mask
        # In boundary regions (mask=1), use noise features
        # In non-boundary regions (mask=0), use original features
        blended_q = original_q * (1 - mask_expanded * self.blend_strength) + \
                   noise_q * (mask_expanded * self.blend_strength)
        blended_k = original_k * (1 - mask_expanded * self.blend_strength) + \
                   noise_k * (mask_expanded * self.blend_strength)
        blended_v = original_v * (1 - mask_expanded * self.blend_strength) + \
                   noise_v * (mask_expanded * self.blend_strength)
                   
        return blended_q, blended_k, blended_v

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        **kwargs
    ) -> torch.FloatTensor:
        """
        Main attention processing with DDIM object insertion logic.
        """
        residual = hidden_states
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape
            height = width = int(sequence_length ** 0.5)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Generate original query, key, value
        query = attn.to_q(hidden_states, *args)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        # Apply DDIM object insertion logic for self-attention layers
        is_cross = encoder_hidden_states is not hidden_states
        if (not is_cross and 
            self.boundary_mask is not None and 
            len(self.fg_noise_latents) > 0 and 
            len(self.bg_noise_latents) > 0):
            
            # Get noise-derived features
            noise_features = self.get_noise_features(attn, self.current_timestep_idx, hidden_states.device)
            
            if noise_features[0] is not None:
                fg_noise_q, fg_noise_k, fg_noise_v, bg_noise_q, bg_noise_k, bg_noise_v = noise_features
                
                # For simplicity, use foreground noise features (you can extend this logic)
                # to blend between fg and bg noise based on additional masks
                noise_q, noise_k, noise_v = fg_noise_q, fg_noise_k, fg_noise_v
                
                # Apply boundary mask blending
                query, key, value = self.apply_boundary_mask_blending(
                    query, key, value,
                    noise_q, noise_k, noise_v,
                    self.boundary_mask,
                    height, width
                )

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        # Compute attention
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_ddim_object_insertion_attention(
    unet,
    fg_noise_latents: List[torch.Tensor],
    bg_noise_latents: List[torch.Tensor], 
    boundary_mask: torch.Tensor,
    blend_strength: float = 0.8,
    target_layers: List[str] = ["up", "mid"]
):
    """
    Register DDIM object insertion attention processors to the UNet.
    
    Args:
        unet: The UNet model
        fg_noise_latents: List of foreground noise latents from DDIM inversion
        bg_noise_latents: List of background noise latents from DDIM inversion
        boundary_mask: Binary mask defining boundary regions [H, W]
        blend_strength: Strength of blending (0.0 = no blending, 1.0 = full replacement)
        target_layers: Which UNet layers to apply the processing to
    """
    # Get the current attention processors from the UNet
    attn_processors = unet.attn_processors
    
    # Create new processors dict
    new_attn_processors = {}
    
    for name in attn_processors.keys():
        # Determine if this layer should use our custom processor
        layer_type = None
        if "down_blocks" in name:
            layer_type = "down"
        elif "mid_block" in name:
            layer_type = "mid"
        elif "up_blocks" in name:
            layer_type = "up"
            
        if layer_type in target_layers:
            processor = DDIMObjectInsertionAttnProcessor(
                place_in_unet=layer_type,
                fg_noise_latents=fg_noise_latents,
                bg_noise_latents=bg_noise_latents,
                boundary_mask=boundary_mask,
                blend_strength=blend_strength
            )
            new_attn_processors[name] = processor
        else:
            # Use existing processor for other layers
            new_attn_processors[name] = attn_processors[name]
    
    unet.set_attn_processor(new_attn_processors)
    return new_attn_processors 