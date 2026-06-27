import torch
from diffusers import DiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

print("Pipeline class:")
print(pipe.__class__.__name__)

print("\nDefault scheduler:")
print(pipe.scheduler)

print("\nScheduler class:")
print(pipe.scheduler.__class__.__name__)

print("\nScheduler config:")
print(pipe.scheduler.config)

print("\nCompatible schedulers:")
for scheduler_cls in pipe.scheduler.compatibles:
    print("-", scheduler_cls.__name__)