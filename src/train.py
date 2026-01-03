from __future__ import annotations

import math
import time
import os
import tempfile
from contextlib import nullcontext
import traceback
from pathlib import Path
import multiprocessing as mp
import psutil  # For system memory monitoring

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
    batch_size: int = 16,  # ULTRA-CONSERVATIVE: Reduced from 32 to prevent system freezes
    num_workers: int = 4,  # ULTRA-CONSERVATIVE: Reduced from 8 for maximum stability
    epochs: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    warmup_steps: int = 100,
    max_steps: int | None = None,
    log_interval: int = 10,
    mask_ratio: float = 0.75,
    use_cached_latents: bool = False,  # NEW: Toggle for cached latents mode
    cache_dir: str | Path = "/mnt/ssv2/cached_latents",  # NEW: Cache directory
    student_model_name: str = "vit_base_patch16_224",  # NEW: Student model size (vit_tiny, vit_small, vit_base)
) -> None:
    # CRITICAL: Monitor system memory to prevent PC freezes
    mem = psutil.virtual_memory()
    mem_available_gb = mem.available / (1024**3)
    mem_total_gb = mem.total / (1024**3)
    mem_percent = mem.percent
    
    print(f"[startup] System RAM: {mem_available_gb:.2f}GB available / {mem_total_gb:.2f}GB total ({mem_percent:.1f}% used)")
    
    # Warn if less than 8GB available (risk of OOM/freeze)
    if mem_available_gb < 8.0:
        print(f"[WARNING] Low system RAM! Only {mem_available_gb:.2f}GB available. Risk of system freeze.")
        print(f"[WARNING] Recommend closing other applications or reducing num_workers further.")
    
    # CRITICAL: Set custom temp directory to avoid disk quota errors
    # Large training set (168K samples) with shuffle creates excessive temp files
    TORCH_TMP_DIR = "/mnt/ssv2/.torch_tmp"
    os.makedirs(TORCH_TMP_DIR, exist_ok=True)
    os.environ['TMPDIR'] = TORCH_TMP_DIR
    os.environ['TEMP'] = TORCH_TMP_DIR
    os.environ['TMP'] = TORCH_TMP_DIR
    tempfile.tempdir = TORCH_TMP_DIR
    print(f"[startup] temp_dir={TORCH_TMP_DIR}")

    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    # CRITICAL: Set GPU memory limits to prevent driver crashes
    if device.type == "cuda":
        # Limit PyTorch to 90% of GPU memory to prevent OOM crashes
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)
        # Enable memory-efficient features
        torch.backends.cudnn.benchmark = False  # More stable, slightly slower
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[startup] GPU: {torch.cuda.get_device_name(device)} ({torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f}GB)")
        print(f"[startup] GPU memory limit set to 90% to prevent crashes")
    
    # Dataset loading - toggle between cached and non-cached modes
    if use_cached_latents:
        from src.data.cached_loader import HybridSSv2Dataset, collate_hybrid
        print(f"[startup] Using CACHED latents mode from {cache_dir}")
        dataset = HybridSSv2Dataset(data_root=data_root, cache_dir=cache_dir, split=split)
        collate_fn = collate_hybrid
    else:
        print(f"[startup] Using NON-CACHED mode (teacher computed on-the-fly)")
        dataset = SSv2Dataset(data_root, split=split)
        collate_fn = collate_drop_none
    
    loader_kwargs = {}
    if num_workers > 0:
        mp_context = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2  # Reduced from 4 for safety
        loader_kwargs["multiprocessing_context"] = mp.get_context(mp_context)
    else:
        mp_context = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
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
        f"{dataset.split_path if hasattr(dataset, 'split_path') else 'N/A'} "
        f"videos_dir={dataset.videos_dir if hasattr(dataset, 'videos_dir') else dataset.video_dataset.videos_dir} "
        f"samples={len(dataset)}"
    )
    if use_cached_latents:
        print(f"[startup] CACHED LATENTS MODE: cache_dir={cache_dir}")
        print(f"[startup] Teacher model will NOT be loaded (using pre-cached latents)")
    else:
        print(f"[startup] NON-CACHED MODE: Teacher model will be loaded and run every step")
    print(
        "[startup] torch="
        f"{torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}"
    )

    model = SALTModel(
        mask_ratio=mask_ratio,
        dtype=model_dtype,
        student_model_name=student_model_name,
    )
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
    
    # Watchdog: track timing to detect stalls
    last_step_time = time.time()
    last_memory_check = time.time()
    
    dataloader_iter = iter(dataloader)
    try:
        preflight_batch = next(dataloader_iter)
        if use_cached_latents:
            # Cached mode: batch is (videos, latents) tuple
            if preflight_batch is None or (isinstance(preflight_batch, tuple) and (preflight_batch[0].numel() == 0 or preflight_batch[1].numel() == 0)):
                print("[preflight] Warning: empty batch from loader on first fetch.")
            else:
                videos, latents = preflight_batch
                if videos.shape[1:] != (3, 16, 224, 224):
                    raise ValueError("[preflight] Expected video shape (B, 3, 16, 224, 224).")
                if latents.ndim != 3 or latents.shape[1:] != (1568, 768):
                    raise ValueError(f"[preflight] Expected latent shape (B, 1568, 768), got {latents.shape}.")
                print(f"[preflight] first batch: videos={tuple(videos.shape)}, latents={tuple(latents.shape)}")
        else:
            # Non-cached mode: batch is just videos
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
            
            # SAFETY: Periodic memory monitoring (every 60 seconds)
            if time.time() - last_memory_check > 60.0:
                mem = psutil.virtual_memory()
                if mem.percent > 95.0:
                    print(f"[CRITICAL] System RAM usage at {mem.percent:.1f}% - ABORTING to prevent freeze!")
                    print(f"[CRITICAL] Available RAM: {mem.available / (1024**3):.2f}GB")
                    raise RuntimeError("System RAM critically low - aborting training to prevent freeze")
                elif mem.percent > 90.0:
                    print(f"[WARNING] High RAM usage: {mem.percent:.1f}% ({mem.available / (1024**3):.2f}GB available)")
                last_memory_check = time.time()
            
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

            # Handle cached vs non-cached modes
            if use_cached_latents:
                # Cached mode: batch is (videos, cached_latents) tuple
                if batch is None or (isinstance(batch, tuple) and (batch[0].numel() == 0 or batch[1].numel() == 0)):
                    continue
                videos, cached_teacher_latents = batch
                videos = videos.to(device=device, dtype=model_dtype, non_blocking=True)
                cached_teacher_latents = cached_teacher_latents.to(device=device, dtype=model_dtype, non_blocking=True)
            else:
                # Non-cached mode: batch is just videos
                if batch is None:
                    continue
                if batch.numel() == 0:
                    continue
                if batch.shape[1:] != (3, 16, 224, 224):
                    raise ValueError("Expected batch shape (B, 3, 16, 224, 224).")
                videos = batch.to(device=device, dtype=model_dtype, non_blocking=True)

            decode_time = time.time() - fetch_start
            start_time = time.time()
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                if use_cached_latents:
                    # Cached mode: mask video, run student, use cached teacher latents
                    masked_video, mask_tokens = model._mask_video(videos)
                    student_tokens = model.student(masked_video)
                    student_pred = model.proj(student_tokens)
                    teacher_latents = cached_teacher_latents
                else:
                    # Non-cached mode: original flow (computes teacher latents)
                    student_pred, teacher_latents, mask_tokens = model(videos)
                
                # Alignment (same for both modes)
                if teacher_latents.shape[1] == student_pred.shape[1] - 1:
                    student_pred = student_pred[:, 1:]
                    mask_tokens = mask_tokens[:, 1:]
                
                if mask_tokens.shape[1] != student_pred.shape[1]:
                    raise ValueError(
                        f"Mask tokens mismatch: mask={mask_tokens.shape[1]} preds={student_pred.shape[1]}"
                    )
                mask_tokens = mask_tokens.unsqueeze(-1).to(dtype=student_pred.dtype)
                loss_map = mse_loss(student_pred, teacher_latents)
                loss = (loss_map * mask_tokens).sum() / (mask_tokens.sum() * loss_map.shape[-1] + 1e-6)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_schedule.step()

            clips_per_sec = videos.shape[0] / max(time.time() - start_time, 1e-6)
            
            # Watchdog: update step time
            current_time = time.time()
            step_duration = current_time - last_step_time
            last_step_time = current_time
            
            # Warn if step took too long (possible hang/stall)
            if step_duration > 300.0 and step > 0:  # 5 minutes
                print(f"[WARNING] Step {step} took {step_duration:.1f}s (possible stall detected)")

            if step % log_interval == 0:
                import wandb

                if device.type == "cuda":
                    gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
                    gpu_mem_max_gb = torch.cuda.max_memory_allocated() / (1024**3)
                else:
                    gpu_mem_gb = 0.0
                    gpu_mem_max_gb = 0.0
                
                # Track system RAM for early leak detection
                mem = psutil.virtual_memory()
                system_ram_percent = mem.percent
                system_ram_available_gb = mem.available / (1024**3)

                wandb.log(
                    {
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "clips_per_sec": clips_per_sec,
                        "batch_decode_sec": decode_time,
                        "gpu_mem_gb": gpu_mem_gb,
                        "gpu_mem_max_gb": gpu_mem_max_gb,
                        "system_ram_percent": system_ram_percent,
                        "system_ram_available_gb": system_ram_available_gb,
                    },
                    step=step,
                )
                print(
                    "[train] step="
                    f"{step} loss={loss.item():.4f} clips/s={clips_per_sec:.2f} "
                    f"decode_s={decode_time:.3f} gpu_mem_gb={gpu_mem_gb:.2f} "
                    f"ram={system_ram_percent:.1f}%"
                )

            step += 1
            if max_steps and step >= max_steps:
                return


if __name__ == "__main__":
    train()
