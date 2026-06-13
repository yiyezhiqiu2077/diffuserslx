import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config
)

pipeline = pipeline.to("cuda")

prompts = [
    "portrait photo of the oldest warrior chief, tribal panther makeup, blue and red face paint, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting, high detail",
    "portrait photo of an old warrior chief, tribal panther makeup, blue and red face paint, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting, high detail",
    "portrait photo of a warrior chief, tribal panther makeup, blue and red face paint, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting, high detail",
    "portrait photo of a young warrior chief, tribal panther makeup, blue and red face paint, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting, high detail",
]

generator = [
    torch.Generator("cuda").manual_seed(1)
    for _ in range(len(prompts))
]

images = pipeline(
    prompt=prompts,
    generator=generator,
    num_inference_steps=25,
).images

grid = make_image_grid(images, rows=2, cols=2)
grid.save("warrior_age_comparison.png")

print("Saved warrior_age_comparison.png")