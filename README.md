# GNRI: Lightning-Fast Image Inversion and Editing for Text-to-Image Diffusion Models

> Dvir Samuel, Barak Meiri, Haggai Maron, Yoad Tewel, Nir Darshan, Gal Chechik, Shai Avidan, Rami Ben-Ari
> OriginAI, Tel Aviv University, Technion, Bar Ilan University, NVIDIA Research

>
>
> Diffusion inversion is the problem of taking an image and a text prompt that describes it and finding a noise latent that would generate the exact same image. Most current deterministic inversion techniques operate by approximately solving an implicit equation and may converge slowly or yield poor reconstructed images. We formulate the problem by finding the roots of an implicit equation and devlop a method to solve it efficiently. Our solution is based on Newton-Raphson (NR), a well-known technique in numerical analysis. We show that a vanilla application of NR is computationally infeasible while naively transforming it to a computationally tractable alternative tends to converge to out-of-distribution solutions, resulting in poor reconstruction and editing. We therefore derive an efficient guided formulation that fastly converges and provides high-quality reconstructions and editing. We showcase our method on real image editing with three popular open-sourced diffusion models: Stable Diffusion, SDXL-Turbo, and Flux with different deterministic schedulers. Our solution, Guided Newton-Raphson Inversion, inverts an image within 0.4 sec (on an A100 GPU) for few-step models (SDXL-Turbo and Flux.1), opening the door for interactive image editing. We further show improved results in image interpolation and generation of rare objects.



<a href="https://arxiv.org/abs/2312.12540"><img src="https://img.shields.io/badge/arXiv-2312.12540-b31b1b.svg" height=22.5></a>
<a href="https://barakmam.github.io/rnri.github.io/" rel="nofollow"><img src="https://camo.githubusercontent.com/ef82193f89c1e8f821031c916df3beccd5dd2c335309055d265d647a89e064e8/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d50726f6a656374266d6573736167653d5765627369746526636f6c6f723d726564" height="20.5" data-canonical-src="https://img.shields.io/static/v1?label=Project&amp;message=Website&amp;color=red" style="max-width: 100%;"></a>
<a href="https://huggingface.co/spaces/rnri/RNRI" rel="nofollow"><img src="https://camo.githubusercontent.com/a4ff28c1dbabfaa46915ab215390308c2415c77b4b180e78909c08d74c174ad8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f25463025394625413425393725323048756767696e67253230466163652d5370616365732d626c7565" alt="Hugging Face Spaces" data-canonical-src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" style="max-width: 100%;"></a></p>


![teaser](https://github.com/user-attachments/assets/3bd550d5-cd73-4bb4-8dcc-07844761af2d)


<br>

## Requirements

Quick installation using pip:
```
pip install -r requirements.txt
```

## Usage

To run a fast Newton-Raphson inversion (using SDXL-turbo), you can simply run 'main.py':

```
python main.py
```

**Baseline: Fixed-point Inversion.**  
We have also implemented a fast fixed-point inversion for StableDiffuion2.1 (See more details in [our previous paper](https://arxiv.org/pdf/2312.12540v1)).

```
cd src/FixedPointInversion
PYTHONPATH='.' python main.py
```

## Cite Our Paper
If you find our paper and repo useful, please cite:
```
@misc{samuel2023regularized,
  author    = {Dvir Samuel and Barak Meiri and Nir Darshan and Shai Avidan and Gal Chechik and Rami Ben-Ari},
  title     = {Regularized Newton Raphson Inversion for Text-to-Image Diffusion Models},
  year      = {2024}
}
```
