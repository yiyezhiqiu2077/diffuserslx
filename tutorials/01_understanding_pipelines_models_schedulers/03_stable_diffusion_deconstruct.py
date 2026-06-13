import torch
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler


# =========================
# 0. Basic settings
# =========================

model_id = "CompVis/stable-diffusion-v1-4"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch_device == "cuda" else torch.float32

prompt = ["a photograph of an astronaut riding a horse"]

height = 512
width = 512
num_inference_steps = 25
guidance_scale = 7.5
batch_size = len(prompt)

print("device:", torch_device)
print("dtype:", dtype)


# =========================
# 1. Load components
# =========================

vae = AutoencoderKL.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=dtype,
    use_safetensors=True,
)

tokenizer = CLIPTokenizer.from_pretrained(
    model_id,
    subfolder="tokenizer",
)

text_encoder = CLIPTextModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    torch_dtype=dtype,
    use_safetensors=True,
)

unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    torch_dtype=dtype,
    use_safetensors=True,
)

scheduler = UniPCMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
)

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

vae.eval()
text_encoder.eval()
unet.eval()


# =========================
# 2. Create text embeddings
# =========================

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

with torch.no_grad():
    text_embeddings = text_encoder(
        text_input.input_ids.to(torch_device)
    )[0]

print("text_embeddings shape:", text_embeddings.shape)


# =========================
# 3. Create unconditional embeddings
# =========================

max_length = text_input.input_ids.shape[-1]

uncond_input = tokenizer(
    [""] * batch_size,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt",
)

with torch.no_grad():
    uncond_embeddings = text_encoder(
        uncond_input.input_ids.to(torch_device)
    )[0]

print("uncond_embeddings shape:", uncond_embeddings.shape)


# =========================
# 4. Concatenate unconditional + conditional embeddings
# =========================

text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

print("combined text_embeddings shape:", text_embeddings.shape)


# =========================
# 5. Create initial latent noise
# =========================

generator = torch.Generator(device=torch_device).manual_seed(0)

latents = torch.randn(
    (
        batch_size,
        unet.config.in_channels,
        height // 8,
        width // 8,
    ),
    generator=generator,
    device=torch_device,
    dtype=dtype,
)

print("initial latents shape:", latents.shape)

latents = latents * scheduler.init_noise_sigma


# =========================
# 6. Denoising loop
# =========================

scheduler.set_timesteps(num_inference_steps)

for i, t in enumerate(tqdm(scheduler.timesteps)):
    # 6.1 duplicate latents for classifier-free guidance
    latent_model_input = torch.cat([latents] * 2, dim=0)

    # 6.2 scale model input if scheduler needs it
    latent_model_input = scheduler.scale_model_input(
        latent_model_input,
        timestep=t,
    )

    # 6.3 predict noise residual with conditional UNet
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

    # 6.4 classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # 6.5 scheduler step: x_t -> x_{t-1}
    latents = scheduler.step(
        noise_pred,
        t,
        latents,
    ).prev_sample

    if (i + 1) in [1, 5, 10, 15, 20, 25]:
        print(
            f"step {i + 1:02d}, "
            f"timestep={int(t)}, "
            f"latents mean={latents.mean().item():.4f}, "
            f"std={latents.std().item():.4f}"
        )


# =========================
# 7. Decode latents to image
# =========================

latents = latents / 0.18215

with torch.no_grad():
    image = vae.decode(latents).sample


# =========================
# 8. Convert tensor to PIL image
# =========================

image = (image / 2 + 0.5).clamp(0, 1)
image = image.squeeze(0)
image = image.permute(1, 2, 0)
image = (image * 255).round().to(torch.uint8).cpu().numpy()

image = Image.fromarray(image)
image.save("stable_diffusion_deconstructed.png")

print("Saved stable_diffusion_deconstructed.png")