import torch
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True)
pipe=pipe.to("cuda")
prompt="An image of a squirrel in Picasso style"
image=pipe(prompt).images[0]
image.save("squirrel.png")