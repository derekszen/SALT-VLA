#!/usr/bin/env python3
"""Training script using CACHED latents for faster training.

This uses pre-computed VideoMAEv2 teacher latents instead of computing them
on-the-fly, resulting in ~5% faster training with identical results.
"""

from src.train import train

if __name__ == "__main__":
    train(
        data_root="/mnt/ssv2",
        split="train",
        batch_size=16,  # Ultra-conservative (safe)
        num_workers=4,  # Ultra-conservative (safe)
        epochs=100,
        lr=1e-4,
        weight_decay=0.05,
        warmup_steps=100,
        log_interval=10,
        mask_ratio=0.75,
        use_cached_latents=True,  # ← CACHED MODE ENABLED
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",
    )