# Code is based on ReNoise https://github.com/garibida/ReNoise-Inversion

from dataclasses import dataclass


@dataclass
class RunConfig:
    num_inference_steps: int = 4

    num_inversion_steps: int = 100

    guidance_scale: float = 0.0

    inversion_max_step: float = 1.0
    
    batch_size: int = 1  # Add batch_size parameter with default value of 1

    def __post_init__(self):
        pass
