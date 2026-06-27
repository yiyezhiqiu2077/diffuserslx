import torch
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
local_dir = "./local_sd15_fp16"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

pipe.save_pretrained(local_dir, variant="fp16")
print(f"Saved pipeline to {local_dir}")

pipe_local = StableDiffusionPipeline.from_pretrained(
    local_dir,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

prompt = "a watercolor painting of a mountain village"

image = pipe_local(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
).images[0]

image.save("04_local_pipeline.png")
print("Saved to 04_local_pipeline.png")