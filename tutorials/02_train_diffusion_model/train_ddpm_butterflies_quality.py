import os
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid

from accelerate import Accelerator
from tqdm.auto import tqdm


# ============================================================
# 1. Training configuration: Scheme A
# ============================================================

@dataclass
class TrainingConfig:
    # image / dataset
    image_size: int = 128
    dataset_name: str = "huggan/smithsonian_butterflies_subset"

    # dataloader
    train_batch_size: int = 8
    eval_batch_size: int = 8

    # training
    num_epochs: int = 30
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 300
    mixed_precision: str = "fp16"

    # DDPM
    num_train_timesteps: int = 1000
    eval_num_inference_steps: int = 250

    # saving / evaluation
    save_image_epochs: int = 5
    save_model_epochs: int = 10
    output_dir: str = "ddpm-butterflies-128-small"
    overwrite_output_dir: bool = True

    # Hugging Face Hub
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_private_repo: bool | None = None

    # reproducibility
    seed: int = 0
config = TrainingConfig()


# ============================================================
# 2. Utilities
# ============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dir(config: TrainingConfig):
    if os.path.exists(config.output_dir):
        if config.overwrite_output_dir:
            print(f"[INFO] Removing existing output_dir: {config.output_dir}")
            shutil.rmtree(config.output_dir)
        else:
            raise RuntimeError(
                f"Output directory already exists: {config.output_dir}. "
                f"Set overwrite_output_dir=True if you want to overwrite it."
            )

    os.makedirs(config.output_dir, exist_ok=True)


def get_grid_size(num_images: int):
    """
    Choose a near-square grid.
    16 -> 4 x 4
    8  -> 2 x 4
    4  -> 2 x 2
    """
    rows = int(math.sqrt(num_images))
    while rows > 1 and num_images % rows != 0:
        rows -= 1
    cols = math.ceil(num_images / rows)
    return rows, cols


def evaluate(config: TrainingConfig, epoch: int, pipeline: DDPMPipeline):
    """
    Generate sample images from random noise using the current model.
    Save them as an image grid.
    """
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=generator,
        num_inference_steps=config.eval_num_inference_steps,
    ).images

    rows, cols = get_grid_size(len(images))
    image_grid = make_image_grid(images, rows=rows, cols=cols)

    sample_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    save_path = os.path.join(sample_dir, f"{epoch:04d}.png")
    image_grid.save(save_path)

    return save_path


# ============================================================
# 3. Dataset and DataLoader
# ============================================================

def build_dataset_and_dataloader(config: TrainingConfig):
    print("[INFO] Loading dataset:", config.dataset_name)

    dataset = load_dataset(
        config.dataset_name,
        split="train",
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )

    def transform(examples):
        images = [
            preprocess(image.convert("RGB"))
            for image in examples["image"]
        ]
        return {"images": images}

    dataset.set_transform(transform)

    train_dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下先用 0，最稳
        pin_memory=torch.cuda.is_available(),
    )

    print("[INFO] Dataset size:", len(dataset))

    example = dataset[0]["images"]
    print("[INFO] One preprocessed image shape:", example.shape)
    print("[INFO] One preprocessed image range:",
          float(example.min()), float(example.max()))

    return dataset, train_dataloader


# ============================================================
# 4. Model and noise scheduler
# ============================================================

def build_model_and_scheduler(config: TrainingConfig):
    print("[INFO] Building UNet2DModel...")

    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {num_params / 1e6:.2f}M")

    return model, noise_scheduler


def sanity_check_model(model: UNet2DModel, dataset):
    """
    Check whether model input and output shapes match.
    """
    sample_image = dataset[0]["images"].unsqueeze(0)

    print("[CHECK] Input shape:", sample_image.shape)

    with torch.no_grad():
        output = model(sample_image, timestep=0).sample

    print("[CHECK] Output shape:", output.shape)

    if output.shape != sample_image.shape:
        raise RuntimeError(
            f"Model output shape {output.shape} does not match input shape {sample_image.shape}"
        )


# ============================================================
# 5. Training loop
# ============================================================

def train_loop(
    config: TrainingConfig,
    model: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    lr_scheduler,
):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.push_to_hub:
            from huggingface_hub import create_repo

            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name,
                private=config.hub_private_repo,
                exist_ok=True,
            ).repo_id
        else:
            repo_id = None

        accelerator.init_trackers("train_ddpm_butterflies_quality")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()

        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch + 1}/{config.num_epochs}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]

            # ------------------------------------------------------------
            # 1. Sample random Gaussian noise epsilon
            # ------------------------------------------------------------
            noise = torch.randn(
                clean_images.shape,
                device=clean_images.device,
            )

            batch_size = clean_images.shape[0]

            # ------------------------------------------------------------
            # 2. Sample random timesteps for each image
            # ------------------------------------------------------------
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=clean_images.device,
                dtype=torch.int64,
            )

            # ------------------------------------------------------------
            # 3. Forward diffusion: x_0 -> x_t
            # ------------------------------------------------------------
            noisy_images = noise_scheduler.add_noise(
                clean_images,
                noise,
                timesteps,
            )

            with accelerator.accumulate(model):
                # --------------------------------------------------------
                # 4. Predict the noise residual epsilon_theta(x_t, t)
                # --------------------------------------------------------
                noise_pred = model(
                    noisy_images,
                    timesteps,
                    return_dict=False,
                )[0]

                # --------------------------------------------------------
                # 5. DDPM training objective:
                #    MSE(epsilon_theta, epsilon)
                # --------------------------------------------------------
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            global_step += 1

        progress_bar.close()

        # ------------------------------------------------------------
        # Evaluation and saving
        # ------------------------------------------------------------
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)

            pipeline = DDPMPipeline(
                unet=unwrapped_model,
                scheduler=noise_scheduler,
            )

            pipeline = pipeline.to(accelerator.device)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                sample_path = evaluate(config, epoch + 1, pipeline)
                print(f"[INFO] Saved samples to: {sample_path}")

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    from huggingface_hub import upload_folder

                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch + 1}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                    print("[INFO] Uploaded model to Hub:", repo_id)
                else:
                    pipeline.save_pretrained(config.output_dir)
                    print("[INFO] Saved model to:", config.output_dir)

        accelerator.wait_for_everyone()

    accelerator.end_training()


# ============================================================
# 6. Main
# ============================================================

def main():
    set_seed(config.seed)

    print("=" * 80)
    print("[CONFIG]")
    print(config)
    print("=" * 80)

    print("[INFO] torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    prepare_output_dir(config)

    dataset, train_dataloader = build_dataset_and_dataloader(config)

    model, noise_scheduler = build_model_and_scheduler(config)

    sanity_check_model(model, dataset)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    num_training_steps = len(train_dataloader) * config.num_epochs

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print("[INFO] Number of training batches per epoch:", len(train_dataloader))
    print("[INFO] Number of total training steps:", num_training_steps)

    train_loop(
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
    )

    print("[DONE] Training complete.")
    print("[DONE] Output directory:", config.output_dir)


if __name__ == "__main__":
    main()