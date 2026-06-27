#更换上一步操作的scheduler，观察生成效果
import torch
from diffusers import DiffusionPipeline,EulerDiscreteScheduler
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe=pipe.to("cuda")
image = pipe(
    "An image of a squirrel in Picasso style",
    num_inference_steps=25,
).images[0]

image.save("squirrel_euler.png")