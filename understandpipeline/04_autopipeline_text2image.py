#学习AutoPipeline的用法

import torch
from diffusers import AutoPipelineForText2Image

model_id = "dreamlike-art/dreamlike-photoreal-2.0"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=dtype,
    use_safetensors=True,
)

print("Actual pipeline class:", pipe.__class__.__name__)
print("Scheduler class:", pipe.scheduler.__class__.__name__)

pipe = pipe.to(device)

prompt = (
    "cinematic photo of Godzilla eating sushi with a cat in an izakaya, "
    "35mm photograph, film, professional, 4k, highly detailed"
)

generator = torch.Generator(device=device).manual_seed(37)

image = pipe(
    prompt,
    generator=generator,
    num_inference_steps=25,
    guidance_scale=7.5,
).images[0]

image.save("autopipeline_txt2img.png")

print("Saved autopipeline_txt2img.png")