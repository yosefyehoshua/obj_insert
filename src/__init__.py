"""
DDIM Object Insertion with Attention Switching

This package implements a custom object insertion mechanism using DDIM inversion 
in the Stable Diffusion XL pipeline, with focus on manipulating the attention 
mechanism during the forward pass to blend foreground and background images.
"""



__version__ = "1.0.0"
__author__ = "DDIM Object Insertion Team"

__all__ = [
    "DDIMObjectInsertionAttnProcessor",
    "register_ddim_object_insertion_attention", 
    "DDIMObjectInserter",
    "invert_ddim",
    "ddim_reconstruction",
    "encode_prompt_sdxl"
] 