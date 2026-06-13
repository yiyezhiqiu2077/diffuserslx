import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from diffusers import UNet2DModel, DDPMScheduler


def save_sample(sample, filename):
    # sample: [B, C, H, W], usually in [-1, 1]
    image = sample.detach().cpu()

    image = image.clamp(-1, 1)
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = (image + 1.0) * 127.5
    image = image.numpy().round().astype(np.uint8)

    image_pil = Image.fromarray(image[0])
    image_pil.save(filename)


repo_id = "google/ddpm-cat-256"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)

model = UNet2DModel.from_pretrained(
    repo_id,
    use_safetensors=True,
).to(device)

scheduler = DDPMScheduler.from_pretrained(repo_id)

torch.manual_seed(0)

noisy_sample = torch.randn(
    1,
    model.config.in_channels,
    model.config.sample_size,
    model.config.sample_size,
).to(device)

sample = noisy_sample

print("initial sample shape:", sample.shape)
print("num timesteps:", len(scheduler.timesteps))

for i, t in enumerate(tqdm(scheduler.timesteps)):
    # 1. UNet predicts noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. Scheduler computes x_{t-1}
    sample = scheduler.step(
        model_output=residual,
        timestep=t,
        sample=sample,
    ).prev_sample

    # 3. Save intermediate results every 100 steps
    if (i + 1) % 100 == 0:
        save_sample(sample, f"cat_step_{i + 1}.png")

save_sample(sample, "ddpm_cat_final.png")

print("Saved ddpm_cat_final.png")