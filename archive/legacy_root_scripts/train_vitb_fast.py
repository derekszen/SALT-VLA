#!/usr/bin/env python3
"""ViT-Base training with paper-aligned settings.

OPTIMIZED based on ViT-Tiny run analysis:
- Loss converged quickly (1000 steps) then plateaued
- GPU memory was very low (0.59GB) - can scale up significantly
- Multi-block masking + tight grad_clip worked well

ViT-Base scaling (15x larger than ViT-Tiny):
- ViT-Base: 86M params, embed_dim=768
- ViT-Tiny: 5.7M params, embed_dim=192
- Predictor scaled to 512 (2/3 of student dim)
- Predictor depth increased to 8 for better predictions
- Lower LR for stability with larger model
"""

from src.train import train

if __name__ == "__main__":
    # ViT-BASE PAPER-ALIGNED TRAINING
    train(
        data_root="/mnt/ssv2",
        split="train",

        # Batch size (ViT-B is 15x larger but GPU has headroom)
        batch_size=32,
        num_workers=8,

        # More epochs - ViT-Tiny plateaued early, ViT-Base may need more
        epochs=5,

        # Learning rate (slightly lower for larger model stability)
        lr=2e-4,
        min_lr=2e-5,

        # PAPER-ALIGNED: Cosine weight decay schedule
        weight_decay_start=0.04,
        weight_decay_end=0.4,

        # More warmup for larger model
        warmup_steps=500,

        # PAPER-ALIGNED: Tight gradient clipping
        grad_clip=0.02,

        # PAPER-ALIGNED: Optimizer betas
        betas=(0.9, 0.95),

        log_interval=20,

        # PAPER-ALIGNED: Multi-block masking
        mask_ratio=0.75,
        masking_strategy="multiblock",

        # Cached latents (reuses existing cache)
        use_cached_latents=True,
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",

        # ViT-Base student (86M params, embed_dim=768)
        student_model_name="vit_base_patch16_224",

        # Predictor config (scaled for ViT-Base)
        # - predictor_dim=384 (half of student's 768, divisible by 6 heads)
        # - depth=8 (increased from 6 for better predictions)
        predictor_dim=384,
        predictor_depth=8,
    )

    # EXPECTED:
    # - ~5,278 steps/epoch
    # - ~26,390 total steps for 5 epochs
    # - ViT-Base: 86M params, embed_dim=768
    # - Predictor: 384-dim, 8 layers
    # - GPU memory: ~8-12GB (vs 0.59GB for ViT-Tiny)
    # - Time: ~45-60 minutes