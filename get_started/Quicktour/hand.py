import torch
import tqdm
import numpy as np
import PIL.Image
from diffusers import UNet2DModel, DDPMScheduler

# 1. 模型仓库
repo_id = "google/ddpm-cat-256"

# 2. 加载 UNet 模型
model = UNet2DModel.from_pretrained(
    repo_id,
    use_safetensors=True,
)

# 3. 加载 scheduler
scheduler = DDPMScheduler.from_pretrained(repo_id)

# 4. 移动到 GPU
model = model.to("cuda")

# 5. 固定随机种子
torch.manual_seed(0)

# 6. 创建初始高斯噪声
sample = torch.randn(
    1,
    model.config.in_channels,
    model.config.sample_size,
    model.config.sample_size,
).to("cuda")

# 7. 逐步去噪
for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    with torch.no_grad():
        # UNet 预测当前 sample 中的噪声
        residual = model(sample, t).sample

    # scheduler 根据预测噪声，计算更干净的 sample
    sample = scheduler.step(residual, t, sample).prev_sample

# 8. 后处理成图片
image = sample.cpu().permute(0, 2, 3, 1)
image = (image + 1.0) * 127.5
image = image.clamp(0, 255).numpy().astype(np.uint8)

image = PIL.Image.fromarray(image[0])
image.save("cat.png")