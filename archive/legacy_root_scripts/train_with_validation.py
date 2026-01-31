#!/usr/bin/env python3
"""Training with validation tracking to determine when to stop.

Monitors validation loss to find optimal epoch count empirically.
Logs val/train loss curves to wandb for analysis.
"""

from src.train import train

if __name__ == "__main__":
    # PRODUCTION TRAINING WITH VALIDATION
    
    # TODO: Implement validation split in train.py
    # For now, this shows the recommended approach:
    # 1. Train for 10-20 epochs on TRAIN split
    # 2. Periodically evaluate on VAL split
    # 3. Stop when val loss plateaus (early stopping)
    
    train(
        data_root="/mnt/ssv2",
        split="train",  # TODO: Add validation_split parameter
        
        # Safe settings
        batch_size=16,
        num_workers=4,
        
        # Training config
        epochs=100,  # Max epochs, use early stopping
        lr=1e-4,
        weight_decay=0.05,
        warmup_steps=100,
        log_interval=10,
        mask_ratio=0.75,
        
        # Cached latents (faster)
        use_cached_latents=True,
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",
        
        # Production student (ViT-Base)
        student_model_name="vit_base_patch16_224",
    )
    
    # RECOMMENDED APPROACH:
    # 1. Run with ViT-Tiny for 10 epochs → check loss curve shape
    # 2. Run with ViT-Small for 20 epochs → estimate convergence time
    # 3. Run with ViT-Base for full training → use early stopping
    # 
    # Typical distillation needs 50-100 epochs (much less than pre-training)