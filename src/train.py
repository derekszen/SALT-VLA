from __future__ import annotations

import math
import time
import os
import tempfile
from contextlib import nullcontext
import traceback
from pathlib import Path
import multiprocessing as mp

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loader import SSv2Dataset, collate_drop_none
from src.models.salt import SALTModel

# REMOVED: file_system sharing strategy - causes disk quota issues
# Instead we'll use spawn multiprocessing context which avoids shared memory entirely


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
    num_workers: int = 8,  # Reduced from 32 to 8 for stability (confirmed via stress testing)
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
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["multiprocessing_context"] = mp.get_context(mp_context)
    else:
        mp_context = None

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
    print(
        "[startup] data_root="
        f"{data_root} split={split} device={device.type} dtype={model_dtype} "
        f"batch_size={batch_size} num_workers={num_workers} "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor')} "
        f"persistent_workers={loader_kwargs.get('persistent_workers')} "
        f"mp_context={mp_context}"
    )
    print(
        "[startup] split_path="
        f"{dataset.split_path} videos_dir={dataset.videos_dir} samples={len(dataset)}"
    )
    print(
        "[startup] torch="
        f"{torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}"
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
    loader_error_count = 0
    model.train()
    dataloader_iter = iter(dataloader)
    try:
        preflight_batch = next(dataloader_iter)
        if preflight_batch is None or preflight_batch.numel() == 0:
            print("[preflight] Warning: empty batch from loader on first fetch.")
        elif preflight_batch.shape[1:] != (3, 16, 224, 224):
            raise ValueError("[preflight] Expected batch shape (B, 3, 16, 224, 224).")
        else:
            print(f"[preflight] first batch shape={tuple(preflight_batch.shape)}")
    except StopIteration:
        raise RuntimeError("[preflight] DataLoader produced no batches.")
    except RuntimeError as exc:
        print("[preflight] RuntimeError in loader:", exc)
        raise

    for epoch in range(epochs):
        dataloader_iter = iter(dataloader)
        while True:
            fetch_start = time.time()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            except RuntimeError as exc:
                loader_error_count += 1
                compact_tb = "".join(traceback.format_exc(limit=3)).strip()
                print(
                    "[data] RuntimeError in loader: "
                    f"{exc.__class__.__name__}: {exc}. Skipping batch."
                )
                if compact_tb:
                    print(f"[data] traceback: {compact_tb}")
                if loader_error_count % 10 == 0:
                    print(f"[data] Warning: loader errors so far={loader_error_count}")
                continue

            if batch is None:
                continue

            if batch.numel() == 0:
                continue
            if batch.shape[1:] != (3, 16, 224, 224):
                raise ValueError("Expected batch shape (B, 3, 16, 224, 224).")

            batch = batch.to(device=device, dtype=model_dtype, non_blocking=True)

            decode_time = time.time() - fetch_start
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

                if device.type == "cuda":
                    gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
                    gpu_mem_max_gb = torch.cuda.max_memory_allocated() / (1024**3)
                else:
                    gpu_mem_gb = 0.0
                    gpu_mem_max_gb = 0.0

                wandb.log(
                    {
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "clips_per_sec": clips_per_sec,
                        "batch_decode_sec": decode_time,
                        "gpu_mem_gb": gpu_mem_gb,
                        "gpu_mem_max_gb": gpu_mem_max_gb,
                    },
                    step=step,
                )
                print(
                    "[train] step="
                    f"{step} loss={loss.item():.4f} clips/s={clips_per_sec:.2f} "
                    f"decode_s={decode_time:.3f} gpu_mem_gb={gpu_mem_gb:.2f}"
                )

            step += 1
            if max_steps and step >= max_steps:
                return


if __name__ == "__main__":
    train()
