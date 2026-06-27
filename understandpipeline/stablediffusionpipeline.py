#手动加载的stable duffusion的pipeline的各个板块并用其生成图片
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    dtype=torch.float16
    #负责latent和RGB image之间的转换
    vae=AutoencoderKL.from_pretrained(
    model_id, subfolder="vae", use_safetensors=True, torch_dtype=dtype).to(device)
        
    tokenizer=CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_safetensors=True)

    text_encoder=CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", use_safetensors=True, torch_dtype=dtype).to(device)

    unet=UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_safetensors=True, torch_dtype=dtype).to(device)

    scheduler=UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.eval()
    text_encoder.eval()
    unet.eval() 
    prompt = ["a photograph of an astronaut riding a horse"]

    height = 512
    width = 512
    num_inference_steps = 25
    guidance_scale = 7.5

    batch_size = len(prompt)
    
    generator = torch.Generator(device=device).manual_seed(0)
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    max_length = text_input.input_ids.shape[-1]
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_input.input_ids.to(device)
        )[0]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(device)
        )[0]
    # 拼接 unconditional 和 conditional embeddings
    text_embeddings = torch.cat(
        [uncond_embeddings, text_embeddings],
        dim=0,
    )

    latents = torch.randn(
        (
            batch_size,
            unet.config.in_channels,
            height // 8,
            width // 8,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # 按 scheduler 要求缩放初始噪声
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        
        latent_model_input = torch.cat([latents] * 2, dim=0)

        # 按 scheduler 的要求缩放输入
        latent_model_input = scheduler.scale_model_input(
            latent_model_input,
            timestep=t,
        )

        
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        
        latents = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=latents,
        ).prev_sample
   
    # Stable Diffusion v1 系列常用 scaling factor
    latents = latents / 0.18215

    with torch.no_grad():
        image = vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)

    
    image = image.squeeze(0)

    
    image = image.permute(1, 2, 0)

    
    image = (image * 255).round().to(torch.uint8)

    
    image = image.cpu().numpy()

    
    image = Image.fromarray(image)

    
    output_path = "manual_stable_diffusion.png"
    image.save(output_path)
if __name__ == "__main__":
    main()






