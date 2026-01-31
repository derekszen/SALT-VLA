#!/usr/bin/env python3
"""ViT-Base 10-epoch training with tuned hyperparameters.

TUNED based on 5-epoch ViT-Base throughput run (2026-01-05):
- Loss: 15.76 → 11.5 (27% reduction), min=10.86
- Plateaued around 11.3-11.7 after step ~15,000
- Throughput: ~185 clips/s with optimizations
- GPU memory: 1.12GB (very low with grad checkpointing off)

Adjustments for 10-epoch run:
- 10 epochs (2x previous) to push past plateau
- Throughput optimizations enabled (grad_checkpointing=False, cudnn_benchmark=True)
- Same LR (1.5e-4) since it worked well
- Predictor depth 12 (paper-aligned)
"""

from src.train import train

if __name__ == "__main__":
    train(
        data_root="/mnt/ssv2",
        split="train",

        # Batch size (stable at 14GB, keep same)
        batch_size=32,
        num_workers=8,

        # 10 epochs for thorough training
        epochs=10,

        # Lower LR for longer training stability
        lr=1.5e-4,
        min_lr=1e-5,

        # PAPER-ALIGNED: Cosine weight decay schedule
        weight_decay_start=0.04,
        weight_decay_end=0.4,

        # More warmup for longer training
        warmup_steps=1000,

        # PAPER-ALIGNED: Tight gradient clipping
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

        # ViT-Base student (86M params, embed_dim=768)
        student_model_name="vit_base_patch16_224",

        # Predictor config (paper-aligned depth=12 for quality)
        predictor_dim=384,
        predictor_depth=12,  # Increased from 8 for paper parity

        # THROUGHPUT OPTIMIZATIONS (from 5-epoch run)
        grad_checkpointing=False,  # +50-100% speed, uses ~1GB VRAM
        cudnn_benchmark=True,      # +5-15% speed
    )

    # EXPECTED:
    # - ~5,278 steps/epoch
    # - ~52,780 total steps for 10 epochs
    # - ViT-Base: 86M params, embed_dim=768
    # - Predictor: 384-dim, 12 layers (paper-aligned)
    # - GPU memory: ~1-2GB (with grad checkpointing off)
    # - Throughput: ~185 clips/s
    # - Time: ~1.5-2 hours