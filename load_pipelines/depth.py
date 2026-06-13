import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipeline = DiffusionPipeline.from_pretrained(
    "prs-eth/marigold-lcm-v1-0",
    custom_pipeline="marigold_depth_estimation",
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None,
)

pipeline = pipeline.to(device)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/community-marigold.png"
)

output = pipeline(
    image,
    denoising_steps=4,
    ensemble_size=5,
    processing_res=768,
    match_input_res=True,
    batch_size=0,
    seed=33,
    color_map="Spectral",
    show_progress_bar=True,
)

depth_colored = output.depth_colored
depth_colored.save("depth_colored.png")

print("Saved depth_colored.png")