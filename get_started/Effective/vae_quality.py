#更换VAE对比生成效果
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from diffusers.utils import make_image_grid

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
prompt = "portrait photo of a old warrior chief"

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config
)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16,
).to("cuda")

pipeline.vae = vae

pipeline = pipeline.to("cuda")

generator = torch.Generator("cuda").manual_seed(0)

image = pipeline(
    prompt,
    generator=generator,
    num_inference_steps=20,
).images[0]

image.save("warrior_with_better_vae.png")

print("Saved warrior_with_better_vae.png")