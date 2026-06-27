#保存模型到本地
import os
import torch
from diffusers import StableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
save_dir = "./local_sd15_multifolder"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

pipe.save_pretrained(save_dir, safe_serialization=True)
print(f"Saved Diffusers multifolder layout to: {save_dir}")

print("\n==== Directory layout ====")
for root, dirs, files in os.walk(save_dir):
    level = root.replace(save_dir, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = "  " * (level + 1)
    for file in files:
        print(f"{sub_indent}{file}")

print("\n==== Pipeline components ====")
for name, component in pipe.components.items():
    print(name, "=>", component.__class__.__name__ if component is not None else None)