#采用半精度（float16）和减少采样步数的方法加速

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

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

pipeline = pipeline.to("cuda")

generator = torch.Generator("cuda").manual_seed(0)

image = pipeline(
    prompt,
    generator=generator,
    num_inference_steps=20,
).images[0]

image.save("warrior_dpm_fp16_20steps.png")

print("Saved warrior_dpm_fp16_20steps.png")