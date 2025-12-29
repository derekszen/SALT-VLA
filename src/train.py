from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
import multiprocessing as mp

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loader import SSv2Dataset, collate_drop_none
from src.models.salt import SALTModel


def _setup_wandb(project: str, run_name: str | None) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required for logging.") from exc

    wandb.init(project=project, name=run_name)


def train(
    data_root: str | Path = "/mnt/ssv2",
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 8,
    epochs: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    warmup_steps: int = 100,
    max_steps: int | None = None,
    log_interval: int = 10,
    mask_ratio: float = 0.75,
) -> None:
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dataset = SSv2Dataset(data_root, split=split)
    loader_kwargs = {}
    if num_workers > 0:
        mp_context = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = mp.get_context(mp_context)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_drop_none,
        drop_last=True,
        **loader_kwargs,
    )

    model = SALTModel(mask_ratio=mask_ratio, dtype=model_dtype)
    model.student.set_grad_checkpointing(True)
    model.to(device=device, dtype=model_dtype)
    if device.type == "cuda":
        model = torch.compile(model, mode="max-autotune")

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss(reduction="none")

    total_steps = max_steps or (len(dataloader) * epochs)
    def lr_lambda(step: int) -> float:
        warmup_scale = min((step + 1) / max(1, warmup_steps), 1.0)
        progress = step / max(1, total_steps)
        cosine_scale = 0.5 * (1.0 + math.cos(progress * math.pi))
        return warmup_scale * cosine_scale

    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    _setup_wandb(project="salt-vla", run_name="salt-training")

    step = 0
    model.train()
    for epoch in range(epochs):
        dataloader_iter = iter(dataloader)
        while True:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            except RuntimeError as exc:
                print(f"[data] RuntimeError in loader: {exc}. Skipping batch.")
                continue

            if batch is None:
                continue

            if batch.numel() == 0:
                continue
            if batch.shape[1:] != (3, 16, 224, 224):
                raise ValueError("Expected batch shape (B, 3, 16, 224, 224).")

            batch = batch.to(device=device, dtype=model_dtype, non_blocking=True)

            start_time = time.time()
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                student_pred, teacher_latents, mask = model(batch)
                if mask.shape[1] != student_pred.shape[1]:
                    raise ValueError(
                        f"Mask tokens mismatch: mask={mask.shape[1]} preds={student_pred.shape[1]}"
                    )
                mask = mask.unsqueeze(-1).to(dtype=student_pred.dtype)
                loss_map = mse_loss(student_pred, teacher_latents)
                loss = (loss_map * mask).sum() / (mask.sum() * loss_map.shape[-1] + 1e-6)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_schedule.step()

            clips_per_sec = batch.shape[0] / max(time.time() - start_time, 1e-6)

            if step % log_interval == 0:
                import wandb

                wandb.log(
                    {
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "clips_per_sec": clips_per_sec,
                    },
                    step=step,
                )
                print(f"[train] step={step} loss={loss.item():.4f} clips/s={clips_per_sec:.2f}")

            step += 1
            if max_steps and step >= max_steps:
                return


if __name__ == "__main__":
    train()
