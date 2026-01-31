#!/usr/bin/env python3
"""ViT-Base HIGH-THROUGHPUT training config.

Optimizations for maximum throughput within 25GB VRAM:
1. Gradient checkpointing DISABLED (biggest gain: +50-100%)
2. cudnn.benchmark enabled (+5-15%)

Based on current run analysis (step 11480):
- Loss plateau at 11.0-12.0
- Throughput baseline: ~105 clips/s
- Expected with optimizations: ~150-200 clips/s

VRAM: ~20-24GB with predictor_depth=12
"""

from src.train import train

if __name__ == "__main__":
    train(
        data_root="/mnt/ssv2",
        split="train",

        # Original batch size (predictor_depth=12 uses more memory)
        batch_size=32,
        num_workers=8,

        epochs=5,

        # Learning rate
        lr=1.5e-4,
        min_lr=1e-5,

        weight_decay_start=0.04,
        weight_decay_end=0.4,

        warmup_steps=1000,
        grad_clip=0.02,
        betas=(0.9, 0.95),

        log_interval=20,

        mask_ratio=0.75,
        masking_strategy="multiblock",

        use_cached_latents=True,
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",

        student_model_name="vit_base_patch16_224",

        # Predictor config (paper-aligned)
        predictor_dim=384,
        predictor_depth=12,

        # THROUGHPUT OPTIONS
        grad_checkpointing=False,  # +50-100% speed, uses ~6GB more VRAM
        cudnn_benchmark=True,      # +5-15% speed
    )