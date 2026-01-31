#!/usr/bin/env python3
"""ViT-Large 10-epoch run with cached latents and tuned LR.

Adjustments vs 5-epoch run:
- Extend epochs to 10
- Lower min_lr for longer decay (1e-5)
- Slightly longer warmup (2000 steps) for stability at scale
"""

import os


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default

from src.train import train

if __name__ == "__main__":
    run_name = os.getenv("RUN_NAME", "vitl_10epoch_v1huge_lr2e4")
    log_dir = os.getenv("RUN_LOG_DIR", "run_logs")

    epochs = _get_env_int("EPOCHS", 10)
    batch_size = _get_env_int("BATCH_SIZE", 24)
    lr = _get_env_float("LR", 2e-4)
    min_lr = _get_env_float("MIN_LR", 1e-5)
    warmup_steps = _get_env_int("WARMUP_STEPS", 2000)
    mask_ratio = _get_env_float("MASK_RATIO", 0.75)
    weight_decay_start = _get_env_float("WEIGHT_DECAY_START", 0.04)
    weight_decay_end = _get_env_float("WEIGHT_DECAY_END", 0.4)
    grad_clip = _get_env_float("GRAD_CLIP", 0.02)

    predictor_dim = _get_env_int("PREDICTOR_DIM", 512)
    predictor_depth = _get_env_int("PREDICTOR_DEPTH", 12)
    predictor_num_heads = _get_env_int("PREDICTOR_NUM_HEADS", 8)

    train(
        data_root="/mnt/ssv2",
        split="train",

        # Reduced batch size for larger model (expect ~20-25GB VRAM)
        batch_size=batch_size,
        num_workers=8,

        # Epochs (default 10; override with EPOCHS)
        epochs=epochs,

        # Tuned LR from ViT-Base results
        lr=lr,
        min_lr=min_lr,

        # PAPER-ALIGNED: Cosine weight decay schedule
        weight_decay_start=weight_decay_start,
        weight_decay_end=weight_decay_end,

        # Longer warmup for stability
        warmup_steps=warmup_steps,

        # PAPER-ALIGNED: Tight gradient clipping
        grad_clip=grad_clip,

        # PAPER-ALIGNED: Optimizer betas
        betas=(0.9, 0.95),

        log_interval=20,

        # PAPER-ALIGNED: Multi-block masking
        mask_ratio=mask_ratio,
        masking_strategy="multiblock",

        # Cached latents
        use_cached_latents=True,
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",

        # ViT-Large student (304M params, embed_dim=1024)
        student_model_name="vit_large_patch16_224",

        # Predictor config (scaled for ViT-Large)
        predictor_dim=predictor_dim,
        predictor_depth=predictor_depth,
        predictor_num_heads=predictor_num_heads,

        run_name=run_name,
        log_dir=log_dir,
    )
