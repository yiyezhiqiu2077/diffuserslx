import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, UNet2DModel

def main():
    model_id ="google/ddpm-cat-256"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DDPMScheduler.from_pretrained(model_id)
    model = UNet2DModel.from_pretrained(model_id, use_safetensors=True).to(device)

    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
    torch.manual_seed(0)
    sample_size=model.config.sample_size
    samples = torch.randn((1, model.config.in_channels, sample_size, sample_size)).to(device)
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            noise_pred = model(samples, t).sample
        samples = scheduler.step(noise_pred, t, samples).prev_sample
    image=samples
    image = (image / 2 + 0.5).clamp(0, 1)
    image=image.squeeze(0)
    image=image.permute(1,2,0)
    image=(image*255).round().to(torch.uint8)
    image=image.cpu().numpy()
    image = Image.fromarray(image)
    output_path = "manual_ddpm_cat.png"
    image.save(output_path)
if __name__ == "__main__":
    main()