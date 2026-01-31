#!/usr/bin/env python3
"""Extreme-speed training for rapid prototyping (~15-20 minutes).

PAPER-ALIGNED configuration:
- ViT-Tiny (smallest student)
- BS=32, workers=8 (validated safe)
- 3 epochs (minimal but enough to see trends)
- Multi-block masking (V-JEPA paper)
- Tight gradient clipping (0.02 vs 1.0)
- Cosine weight decay schedule (0.04 → 0.4)
- Optimizer β₂=0.95 for stability

Expected time: ~15-20 minutes
"""

from src.train import train

if __name__ == "__main__":
    # PAPER-ALIGNED EXTREME SPEED MODE
    train(
        data_root="/mnt/ssv2",
        split="train",

        # Validated safe settings
        batch_size=32,
        num_workers=8,

        # MINIMAL EPOCHS
        epochs=3,

        # Learning rate (tuned for ViT-Tiny)
        lr=4.5e-4,
        min_lr=4.5e-5,  # 10% floor

        # PAPER-ALIGNED: Cosine weight decay schedule
        weight_decay_start=0.04,
        weight_decay_end=0.4,

        warmup_steps=200,

        # PAPER-ALIGNED: Tight gradient clipping (was 1.0)
        grad_clip=0.02,

        # PAPER-ALIGNED: Optimizer betas
        betas=(0.9, 0.95),

        log_interval=20,

        # PAPER-ALIGNED: Multi-block masking
        mask_ratio=0.75,
        masking_strategy="multiblock",

        # Cached latents
        use_cached_latents=True,
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",

        # ViT-Tiny student
        student_model_name="vit_tiny_patch16_224",

        # Predictor config
        predictor_dim=384,
        predictor_depth=6,
    )

    # EXPECTED:
    # - ~5,278 steps/epoch
    # - ~15,834 total steps for 3 epochs
    # - Multi-block masking should improve representation quality
    # - Tight grad_clip (0.02) prevents training spikes
    # - ~15-20 min total