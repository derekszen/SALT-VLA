#!/usr/bin/env python3
"""ViT-Base Experiment 1: Higher Learning Rate

Hypothesis: Current LR (1.5e-4) may be too conservative for ViT-B.
V-JEPA paper uses higher LR for larger models.

Change: lr 1.5e-4 → 3e-4 (2x increase)
Expected: Faster convergence, potentially lower plateau

Quick iteration: 3 epochs only (~30 min)
"""

import os

from src.train import train

if __name__ == "__main__":
    run_name = os.getenv("RUN_NAME", "vitb_exp1_higherlr_v1huge")
    log_dir = os.getenv("RUN_LOG_DIR", "run_logs")

    train(
        data_root="/mnt/ssv2",
        split="train",

        batch_size=32,
        num_workers=8,

        # SHORT RUN for quick iteration
        epochs=3,

        # EXPERIMENT: Higher LR (2x baseline)
        lr=3e-4,
        min_lr=3e-5,

        # Paper-aligned settings
        weight_decay_start=0.04,
        weight_decay_end=0.4,
        warmup_steps=500,  # Shorter warmup for 3 epochs
        grad_clip=0.02,
        betas=(0.9, 0.95),

        log_interval=20,

        mask_ratio=0.75,
        masking_strategy="multiblock",

        use_cached_latents=True,
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",

        student_model_name="vit_base_patch16_224",

        predictor_dim=384,
        predictor_depth=12,

        # Throughput optimizations
        grad_checkpointing=False,
        cudnn_benchmark=True,

        run_name=run_name,
        log_dir=log_dir,
    )
