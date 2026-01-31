#!/usr/bin/env python3
"""ViT-Base Experiment 1: Higher LR + VICReg-style penalties (10 epochs).

Adds variance + covariance penalties to reduce mean-collapse and
encourage diverse latent dimensions.
"""

import os

from src.train import train


if __name__ == "__main__":
    run_name = os.getenv("RUN_NAME", "vitb_exp1_higherlr_vicreg10_v1huge")
    log_dir = os.getenv("RUN_LOG_DIR", "run_logs")

    train(
        data_root="/mnt/ssv2",
        split="train",

        batch_size=32,
        num_workers=8,

        epochs=10,

        lr=3e-4,
        min_lr=3e-5,

        weight_decay_start=0.04,
        weight_decay_end=0.4,
        warmup_steps=500,
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

        # VICReg-style penalties (tune if unstable)
        variance_loss_weight=1.0,
        variance_target=1.0,
        covariance_loss_weight=0.1,

        grad_checkpointing=False,
        cudnn_benchmark=True,

        run_name=run_name,
        log_dir=log_dir,
    )
