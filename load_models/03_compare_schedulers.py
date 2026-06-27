#更换不同的schedulers进行对比实验

import torch
from diffusers import (
    DiffusionPipeline,
    PNDMScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
negative_prompt = "low quality, blurry, distorted, bad anatomy"

scheduler_classes = [
    PNDMScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
]

for scheduler_cls in scheduler_classes:
    print(f"\nRunning scheduler: {scheduler_cls.__name__}")

    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

    generator = torch.Generator(device="cuda").manual_seed(8)

    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    filename = f"03_{scheduler_cls.__name__}.png"
    image.save(filename)
    print(f"Saved: {filename}")

print("\nDone.")