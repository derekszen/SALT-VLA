from __future__ import annotations

import math
import time
import os
import tempfile
from contextlib import nullcontext
import traceback
from pathlib import Path
import multiprocessing as mp
import logging
import sys
import signal
import faulthandler
import atexit
import psutil  # For system memory monitoring

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loader import (
    SSv2Dataset,
    SSv2MaskedDataset,
    collate_drop_none,
    collate_drop_none_with_masks,
)
from src.models.salt import SALTModel

# REMOVED: file_system sharing strategy - causes disk quota issues
# Instead we'll use spawn multiprocessing context which avoids shared memory entirely


def _setup_wandb(project: str, run_name: str | None) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required for logging.") from exc

    wandb.init(project=project, name=run_name)

class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _setup_run_logging(run_name: str | None, log_dir: str | Path) -> tuple[logging.Logger, Path]:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_name = run_name or Path(sys.argv[0]).stem or "train"
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in base_name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{safe_name}_{timestamp}_pid{os.getpid()}.log"

    log_file = open(log_path, "a", buffering=1)
    stdout_original = sys.stdout
    stderr_original = sys.stderr
    sys.stdout = _Tee(stdout_original, log_file)
    sys.stderr = _Tee(stderr_original, log_file)

    logger = logging.getLogger("salt.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    logging.captureWarnings(True)
    faulthandler.enable(log_file, all_threads=True)

    def _handle_signal(signum: int, _frame) -> None:
        logger.error("Received signal %s; shutting down.", signum)
        raise SystemExit(1)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGABRT):
        signal.signal(sig, _handle_signal)

    def _excepthook(exc_type, exc, tb) -> None:
        logger.error("Unhandled exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _on_exit() -> None:
        try:
            logger.info("Process exiting.")
        except Exception:
            pass
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    atexit.register(_on_exit)
    return logger, log_path


def variance_penalty(predictions: torch.Tensor, target_std: float) -> torch.Tensor:
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    if pred_flat.numel() == 0:
        return pred_flat.sum()
    std = pred_flat.float().std(dim=0)
    return torch.relu(target_std - std).mean()


def covariance_penalty(predictions: torch.Tensor) -> torch.Tensor:
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    if pred_flat.shape[0] <= 1:
        return pred_flat.sum() * 0.0
    pred = pred_flat.float()
    pred = pred - pred.mean(dim=0)
    cov = (pred.T @ pred) / (pred.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / pred.shape[1]


def train(
    data_root: str | Path = "/mnt/ssv2",
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    epochs: int = 1,
    lr: float = 1e-4,
    min_lr: float = 1e-5,
    # Weight decay schedule (paper: 0.04 → 0.4 cosine)
    weight_decay_start: float = 0.04,
    weight_decay_end: float = 0.4,
    warmup_steps: int = 100,
    max_steps: int | None = None,
    log_interval: int = 10,
    mask_ratio: float = 0.75,
    masking_strategy: str = "multiblock",  # NEW: "random" or "multiblock"
    use_cached_latents: bool = False,
    cache_dir: str | Path = "/mnt/ssv2/cached_latents",
    student_model_name: str = "vit_base_patch16_224",
    grad_clip: float = 0.02,  # PAPER: Tight clipping (was 1.0)
    # Optimizer betas (paper: β₂=0.95 for stability)
    betas: tuple[float, float] = (0.9, 0.95),
    # Predictor config
    predictor_dim: int = 384,
    predictor_depth: int = 6,
    predictor_num_heads: int = 6,  # Must divide predictor_dim evenly
    # THROUGHPUT options
    grad_checkpointing: bool = True,  # Disable for +50-100% speed (uses more VRAM)
    cudnn_benchmark: bool = False,  # Enable for +5-15% speed (less deterministic)
    # Training stability options
    grad_accum_steps: int = 1,  # Gradient accumulation steps (virtual batch size)
    variance_loss_weight: float = 0.0,  # VICReg-style variance penalty weight
    variance_target: float = 1.0,  # Target stddev for variance penalty
    covariance_loss_weight: float = 0.0,  # VICReg-style covariance penalty weight
    use_dataloader_masks: bool = True,  # Use multi-block masks from dataloader
    tubelet_size: int = 2,
    patch_size: int = 16,
    run_name: str | None = None,
    log_dir: str | Path = "run_logs",
) -> None:
    logger, log_path = _setup_run_logging(run_name, log_dir)
    logger.info("Run log file: %s", log_path)
    logger.info("argv=%s", " ".join(sys.argv))

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

    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    # CRITICAL: Set GPU memory limits to prevent driver crashes
    if device.type == "cuda":
        # Limit PyTorch to 90% of GPU memory to prevent OOM crashes
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)
        # Enable memory-efficient features
        torch.backends.cudnn.benchmark = cudnn_benchmark  # Configurable for speed vs stability
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[startup] GPU: {torch.cuda.get_device_name(device)} ({torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f}GB)")
        print(f"[startup] GPU memory limit set to 90% to prevent crashes")
    
    # Dataset loading - toggle between cached and non-cached modes
    if use_cached_latents:
        if use_dataloader_masks:
            from src.data.cached_loader import HybridSSv2MaskedDataset, collate_hybrid_with_masks
            print(f"[startup] Using CACHED latents + DATALOADER masks from {cache_dir}")
            dataset = HybridSSv2MaskedDataset(
                data_root=data_root,
                cache_dir=cache_dir,
                split=split,
                mask_ratio=mask_ratio,
                masking_strategy=masking_strategy,
                tubelet_size=tubelet_size,
                patch_size=patch_size,
            )
            collate_fn = collate_hybrid_with_masks
        else:
            from src.data.cached_loader import HybridSSv2Dataset, collate_hybrid
            print(f"[startup] Using CACHED latents mode from {cache_dir}")
            dataset = HybridSSv2Dataset(data_root=data_root, cache_dir=cache_dir, split=split)
            collate_fn = collate_hybrid
    else:
        if use_dataloader_masks:
            print("[startup] Using NON-CACHED mode with DATALOADER masks")
            dataset = SSv2MaskedDataset(
                data_root,
                split=split,
                mask_ratio=mask_ratio,
                masking_strategy=masking_strategy,
                tubelet_size=tubelet_size,
                patch_size=patch_size,
            )
            collate_fn = collate_drop_none_with_masks
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
    if use_dataloader_masks:
        print(f"[startup] DATALOADER MASKS: strategy={masking_strategy} ratio={mask_ratio}")
    if grad_accum_steps > 1:
        effective_batch_size = batch_size * grad_accum_steps
        print(f"[startup] grad_accum_steps={grad_accum_steps} effective_batch_size={effective_batch_size}")
    if variance_loss_weight > 0:
        print(
            "[startup] variance_loss_weight="
            f"{variance_loss_weight} variance_target={variance_target}"
        )
    if covariance_loss_weight > 0:
        print(f"[startup] covariance_loss_weight={covariance_loss_weight}")
    print(
        "[startup] torch="
        f"{torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}"
    )

    model = SALTModel(
        mask_ratio=mask_ratio,
        masking_strategy=masking_strategy,  # NEW
        dtype=model_dtype,
        student_model_name=student_model_name,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        predictor_dim=predictor_dim,  # NEW
        predictor_depth=predictor_depth,  # NEW
        predictor_num_heads=predictor_num_heads,  # NEW
    )
    model.student.set_grad_checkpointing(grad_checkpointing)
    model.to(device=device, dtype=model_dtype)
    if device.type == "cuda":
        model = torch.compile(model, mode="max-autotune")

    # Optimizer with paper-aligned betas
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,  # PAPER: (0.9, 0.95)
            weight_decay=weight_decay_start,  # Initial value, will be scheduled
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay_start
        )

    total_steps = max_steps or math.ceil((len(dataloader) * epochs) / grad_accum_steps)
    min_lr_ratio = min_lr / lr
    
    # Weight decay schedule (paper: cosine from 0.04 → 0.4)
    def get_weight_decay(step: int) -> float:
        progress = step / max(1, total_steps)
        # Cosine schedule: starts at weight_decay_start, ends at weight_decay_end
        return weight_decay_start + (weight_decay_end - weight_decay_start) * 0.5 * (1 - math.cos(progress * math.pi))
    
    def lr_lambda(step: int) -> float:
        warmup_scale = min((step + 1) / max(1, warmup_steps), 1.0)
        progress = step / max(1, total_steps)
        # Cosine decay with minimum LR floor
        cosine_scale = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(progress * math.pi))
        return warmup_scale * cosine_scale

    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"[startup] masking_strategy={masking_strategy} grad_clip={grad_clip}")
    print(f"[startup] betas={betas} weight_decay={weight_decay_start}→{weight_decay_end}")

    wandb_run_name = run_name or "salt-training"
    _setup_wandb(project="salt-vla", run_name=wandb_run_name)

    step = 0
    loader_error_count = 0
    model.train()
    
    # Watchdog: track timing to detect stalls
    last_step_time = time.time()
    last_memory_check = time.time()
    
    dataloader_iter = iter(dataloader)

    accum_steps = 0
    accum_loss = 0.0
    accum_mse_loss = 0.0
    accum_var_loss = 0.0
    accum_cov_loss = 0.0
    last_clips_per_sec = 0.0
    last_decode_time = 0.0
    last_gpu_mem_gb = 0.0
    last_gpu_mem_max_gb = 0.0

    def apply_optimizer_step() -> bool:
        nonlocal step
        nonlocal accum_steps, accum_loss, accum_mse_loss, accum_var_loss, accum_cov_loss
        nonlocal last_step_time, last_clips_per_sec, last_decode_time
        nonlocal last_gpu_mem_gb, last_gpu_mem_max_gb

        if accum_steps == 0:
            return False

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_schedule.step()

        current_wd = get_weight_decay(step)
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = current_wd

        avg_loss = accum_loss / accum_steps
        avg_mse_loss = accum_mse_loss / accum_steps
        avg_var_loss = (
            accum_var_loss / accum_steps if variance_loss_weight > 0 else 0.0
        )
        avg_cov_loss = (
            accum_cov_loss / accum_steps if covariance_loss_weight > 0 else 0.0
        )

        accum_steps = 0
        accum_loss = 0.0
        accum_mse_loss = 0.0
        accum_var_loss = 0.0
        accum_cov_loss = 0.0

        current_time = time.time()
        step_duration = current_time - last_step_time
        last_step_time = current_time
        if step_duration > 300.0 and step > 0:
            print(f"[WARNING] Step {step} took {step_duration:.1f}s (possible stall detected)")

        if step % log_interval == 0:
            import wandb

            mem = psutil.virtual_memory()
            system_ram_percent = mem.percent
            system_ram_available_gb = mem.available / (1024**3)

            wandb_payload = {
                "loss": avg_loss,
                "mse_loss": avg_mse_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "clips_per_sec": last_clips_per_sec,
                "batch_decode_sec": last_decode_time,
                "gpu_mem_gb": last_gpu_mem_gb,
                "gpu_mem_max_gb": last_gpu_mem_max_gb,
                "system_ram_percent": system_ram_percent,
                "system_ram_available_gb": system_ram_available_gb,
            }
            if variance_loss_weight > 0:
                wandb_payload["variance_loss"] = avg_var_loss
            if covariance_loss_weight > 0:
                wandb_payload["covariance_loss"] = avg_cov_loss

            wandb.log(wandb_payload, step=step)
            print(
                "[train] step="
                f"{step} loss={avg_loss:.4f} clips/s={last_clips_per_sec:.2f} "
                f"decode_s={last_decode_time:.3f} gpu_mem_gb={last_gpu_mem_gb:.2f} "
                f"ram={system_ram_percent:.1f}%"
            )

        step += 1
        if max_steps and step >= max_steps:
            return True
        return False
    expected_patches = (16 // tubelet_size) * (224 // patch_size) * (224 // patch_size)
    expected_masked = int(expected_patches * mask_ratio)
    expected_visible = expected_patches - expected_masked

    try:
        preflight_batch = next(dataloader_iter)
        if use_cached_latents and use_dataloader_masks:
            if (
                preflight_batch is None
                or (isinstance(preflight_batch, tuple) and preflight_batch[0].numel() == 0)
            ):
                print("[preflight] Warning: empty batch from loader on first fetch.")
            else:
                videos, latents, visible_idx, masked_idx = preflight_batch
                if videos.shape[1:] != (3, 16, 224, 224):
                    raise ValueError("[preflight] Expected video shape (B, 3, 16, 224, 224).")
                if latents.ndim != 3 or latents.shape[1:] != (1568, 768):
                    raise ValueError(f"[preflight] Expected latent shape (B, 1568, 768), got {latents.shape}.")
                if visible_idx.shape[1] != expected_visible or masked_idx.shape[1] != expected_masked:
                    raise ValueError("[preflight] Mask indices shape mismatch for expected patch count.")
                print(
                    "[preflight] first batch: videos="
                    f"{tuple(videos.shape)}, latents={tuple(latents.shape)}, "
                    f"visible={tuple(visible_idx.shape)}, masked={tuple(masked_idx.shape)}"
                )
        elif use_cached_latents:
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
        elif use_dataloader_masks:
            # Non-cached mode with dataloader masks
            if (
                preflight_batch is None
                or (isinstance(preflight_batch, tuple) and preflight_batch[0].numel() == 0)
            ):
                print("[preflight] Warning: empty batch from loader on first fetch.")
            else:
                videos, visible_idx, masked_idx = preflight_batch
                if videos.shape[1:] != (3, 16, 224, 224):
                    raise ValueError("[preflight] Expected video shape (B, 3, 16, 224, 224).")
                if visible_idx.shape[1] != expected_visible or masked_idx.shape[1] != expected_masked:
                    raise ValueError("[preflight] Mask indices shape mismatch for expected patch count.")
                print(
                    "[preflight] first batch: videos="
                    f"{tuple(videos.shape)}, visible={tuple(visible_idx.shape)}, "
                    f"masked={tuple(masked_idx.shape)}"
                )
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
            visible_idx = None
            masked_idx = None
            if use_cached_latents and use_dataloader_masks:
                if (
                    batch is None
                    or (isinstance(batch, tuple) and (batch[0].numel() == 0 or batch[1].numel() == 0))
                ):
                    continue
                videos, cached_teacher_latents, visible_idx, masked_idx = batch
                if visible_idx.numel() == 0 or masked_idx.numel() == 0:
                    continue
                videos = videos.to(device=device, dtype=model_dtype, non_blocking=True)
                cached_teacher_latents = cached_teacher_latents.to(
                    device=device, dtype=model_dtype, non_blocking=True
                )
                visible_idx = visible_idx.to(device=device, non_blocking=True)
                masked_idx = masked_idx.to(device=device, non_blocking=True)
            elif use_cached_latents:
                # Cached mode: batch is (videos, cached_latents) tuple
                if batch is None or (isinstance(batch, tuple) and (batch[0].numel() == 0 or batch[1].numel() == 0)):
                    continue
                videos, cached_teacher_latents = batch
                videos = videos.to(device=device, dtype=model_dtype, non_blocking=True)
                cached_teacher_latents = cached_teacher_latents.to(device=device, dtype=model_dtype, non_blocking=True)
            elif use_dataloader_masks:
                if batch is None or (isinstance(batch, tuple) and batch[0].numel() == 0):
                    continue
                videos, visible_idx, masked_idx = batch
                if visible_idx.numel() == 0 or masked_idx.numel() == 0:
                    continue
                videos = videos.to(device=device, dtype=model_dtype, non_blocking=True)
                visible_idx = visible_idx.to(device=device, non_blocking=True)
                masked_idx = masked_idx.to(device=device, non_blocking=True)
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
                # V-JEPA forward: returns (pred_masked, teacher_masked)
                # Both tensors have shape (B, N_masked, D_teacher)
                if use_cached_latents:
                    if use_dataloader_masks:
                        pred_masked, teacher_masked = model(
                            videos,
                            cached_teacher_latents,
                            visible_idx=visible_idx,
                            masked_idx=masked_idx,
                        )
                    else:
                        pred_masked, teacher_masked = model(videos, cached_teacher_latents)
                else:
                    if use_dataloader_masks:
                        pred_masked, teacher_masked = model(
                            videos, visible_idx=visible_idx, masked_idx=masked_idx
                        )
                    else:
                        pred_masked, teacher_masked = model(videos)
                
                # Simple MSE loss - no masking needed (only masked positions returned)
                mse_loss = nn.functional.mse_loss(pred_masked, teacher_masked)
                if variance_loss_weight > 0:
                    var_loss = variance_penalty(pred_masked, variance_target)
                else:
                    var_loss = pred_masked.new_tensor(0.0)
                if covariance_loss_weight > 0:
                    cov_loss = covariance_penalty(pred_masked)
                else:
                    cov_loss = pred_masked.new_tensor(0.0)
                loss = (
                    mse_loss
                    + variance_loss_weight * var_loss
                    + covariance_loss_weight * cov_loss
                )
                
            accum_loss += loss.item()
            accum_mse_loss += mse_loss.item()
            if variance_loss_weight > 0:
                accum_var_loss += var_loss.item()
            if covariance_loss_weight > 0:
                accum_cov_loss += cov_loss.item()
            accum_steps += 1

            (loss / grad_accum_steps).backward()

            last_clips_per_sec = videos.shape[0] / max(time.time() - start_time, 1e-6)
            last_decode_time = decode_time
            if device.type == "cuda":
                last_gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
                last_gpu_mem_max_gb = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                last_gpu_mem_gb = 0.0
                last_gpu_mem_max_gb = 0.0

            if accum_steps >= grad_accum_steps:
                if apply_optimizer_step():
                    return

        if accum_steps > 0:
            if apply_optimizer_step():
                return


if __name__ == "__main__":
    train()
