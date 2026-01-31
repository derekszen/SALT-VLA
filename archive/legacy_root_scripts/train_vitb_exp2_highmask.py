#!/usr/bin/env python3
"""ViT-Base Experiment 2: Higher Mask Ratio

Hypothesis: Higher masking creates harder pretext task → better features.
V-JEPA paper shows 0.9 masking can improve downstream performance.

Change: mask_ratio 0.75 → 0.9 (more aggressive)
Expected: Harder task, potentially stronger representations

Quick iteration: 3 epochs only (~30 min)
"""

from src.train import train

if __name__ == "__main__":
    train(
        data_root="/mnt/ssv2",
        split="train",

        batch_size=32,
        num_workers=8,

        # SHORT RUN for quick iteration
        epochs=3,

        # Same LR as baseline
        lr=1.5e-4,
        min_lr=1e-5,

        # Paper-aligned settings
        weight_decay_start=0.04,
        weight_decay_end=0.4,
        warmup_steps=500,
        grad_clip=0.02,
        betas=(0.9, 0.95),

        log_interval=20,

        # EXPERIMENT: Higher mask ratio
        mask_ratio=0.9,
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
    )