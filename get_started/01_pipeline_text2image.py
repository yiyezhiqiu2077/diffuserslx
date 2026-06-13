import torch
from diffusers import DiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe = pipe.to("cuda")

prompt = "An image of a squirrel in Picasso style"

image = pipe(prompt).images[0]

image.save("image_of_squirrel_painting.png")

print("Saved image_of_squirrel_painting.png")