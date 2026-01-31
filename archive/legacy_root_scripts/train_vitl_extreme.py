#!/usr/bin/env python3
"""ViT-Large extreme training with paper-aligned settings.

ViT-Large scaling (3.5x larger than ViT-Base):
- ViT-Large: 304M params, embed_dim=1024, 24 layers
- ViT-Base:  86M params, embed_dim=768, 12 layers

Memory considerations:
- ViT-Base used ~14GB at BS=32
- ViT-Large ~3.5x larger, expect ~20-25GB at BS=32
- RTX 5090 has 32GB, should fit with BS=24-32
- Using BS=24 for safety margin

Hyperparameter adjustments:
- Lower LR (1e-4) for larger model stability
- Predictor dim=512 (half of 1024, divisible by 8 heads)
- Predictor depth=12 (paper-aligned)
- More warmup (1500 steps)
"""

from src.train import train

if __name__ == "__main__":
    train(
        data_root="/mnt/ssv2",
        split="train",

        # Reduced batch size for larger model (expect ~20-25GB VRAM)
        batch_size=24,
        num_workers=8,

        # 5 epochs for initial ViT-Large run
        epochs=5,

        # Lower LR for large model stability
        lr=1e-4,
        min_lr=1e-5,

        # PAPER-ALIGNED: Cosine weight decay schedule
        weight_decay_start=0.04,
        weight_decay_end=0.4,

        # More warmup for larger model
        warmup_steps=1500,

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

        # ViT-Large student (304M params, embed_dim=1024)
        student_model_name="vit_large_patch16_224",

        # Predictor config (scaled for ViT-Large)
        # - predictor_dim=512 (half of 1024)
        # - num_heads=8 (512/8=64 head_dim)
        # - depth=12 (paper-aligned)
        predictor_dim=512,
        predictor_depth=12,
        predictor_num_heads=8,
    )

    # EXPECTED:
    # - ~7,038 steps/epoch (168903/24)
    # - ~35,190 total steps for 5 epochs
    # - ViT-Large: 304M params, embed_dim=1024
    # - Predictor: 512-dim, 12 layers, 8 heads
    # - GPU memory: ~20-25GB
    # - Time: ~3-4 hours