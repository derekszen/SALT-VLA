#!/usr/bin/env python3
"""Wrapper to run training with PYTHONPATH set correctly."""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Now import and run training
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
        epochs=10,
        lr=1e-4,
        weight_decay=0.05,
        warmup_steps=100,
        log_interval=10,
        mask_ratio=0.75,

        # Use cached latents (skip teacher)
        use_cached_latents=True,
        cache_dir="/mnt/ssv2/cached_latents",

        # Use smaller student for faster iteration
        student_model_name="vit_tiny_patch16_224",
    )
