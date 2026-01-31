#!/usr/bin/env python3
"""Fast iteration script for quick experiments.

Uses:
- ViT-Tiny student (5M params, ~4x faster than ViT-Base)
- Cached latents (no teacher overhead)
- Ultra-safe settings (BS=16, W=4)
- Optional: subset of data for even faster iteration

For testing architectures, hyperparameters, and debugging.
"""

from src.train import train

if __name__ == "__main__":
    # FAST ITERATION MODE
    train(
        data_root="/mnt/ssv2",
        split="train",
        
        # Ultra-conservative (safe)
        batch_size=16,
        num_workers=4,
        
        # Fast iteration settings
        epochs=10,  # Short runs for testing
        lr=1e-4,
        weight_decay=0.05,
        warmup_steps=100,
        log_interval=10,
        mask_ratio=0.75,
        
        # Use cached latents (skip teacher)
        use_cached_latents=True,
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",
        
        # KEY: Use smaller student for faster iteration
        # Options:
        #   - "vit_tiny_patch16_224"  (~5M params,  ~4x faster)  ← CURRENT
        #   - "vit_small_patch16_224" (~22M params, ~2x faster)
        #   - "vit_base_patch16_224"  (~86M params, 1x - production)
        student_model_name="vit_tiny_patch16_224",
    )