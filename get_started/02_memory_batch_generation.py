import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
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

pipeline.enable_attention_slicing()

pipeline = pipeline.to("cuda")


def get_inputs(batch_size=4):
    generator = [
        torch.Generator("cuda").manual_seed(i)
        for i in range(batch_size)
    ]

    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {
        "prompt": prompts,
        "generator": generator,
        "num_inference_steps": num_inference_steps,
    }


images = pipeline(**get_inputs(batch_size=4)).images

grid = make_image_grid(images, rows=2, cols=2)
grid.save("warrior_grid_batch4.png")

print("Saved warrior_grid_batch4.png")