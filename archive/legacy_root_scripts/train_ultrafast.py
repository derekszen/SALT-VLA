#!/usr/bin/env python3
"""Ultra-fast training for rapid iteration (~30-40 minutes).

Uses OPTIMIZED settings:
- ViT-Tiny (smallest student)
- BS=32 (validated safe)
- workers=12 (more aggressive, leverages 16-core CPU better)
- 10 epochs (full convergence, not cut short)
- Cached latents (no teacher overhead)
- IMPROVED: Higher LR with min_lr floor for continued learning

Expected time: ~30-40 minutes for 10 epochs
"""

from src.train import train

if __name__ == "__main__":
    # ULTRA-FAST MODE (optimized hyperparameters)
    train(
        data_root="/mnt/ssv2",
        split="train",
        
        # OPTIMIZED SETTINGS
        # BS=32 tested stable, 12 workers for better I/O saturation
        batch_size=32,
        num_workers=12,  # Increased from 8 → 12
        
        # Full 10 epochs for proper convergence
        epochs=10,
        
        # IMPROVED: Higher LR for faster convergence
        lr=3e-4,  # Increased from 1e-4
        min_lr=3e-5,  # 10% of peak LR as floor
        weight_decay=0.05,
        warmup_steps=500,  # ~1% of total steps for stability
        grad_clip=1.0,  # Gradient clipping for stability
        
        log_interval=10,
        mask_ratio=0.75,
        
        # Use cached latents (skip teacher)
        use_cached_latents=True,
        use_dataloader_masks=True,
        cache_dir="/mnt/ssv2/cached_latents_v1huge",
        
        # ViT-Tiny for maximum speed
        student_model_name="vit_tiny_patch16_224",
    )
    
    # EXPECTED (with improved LR):
    # - ~5,278 steps/epoch (168,903 / 32, after blacklist filtering)
    # - ~52,780 total steps for 10 epochs
    # - Loss should drop from ~15 to <8 with continued learning
    # - ~30-40 min total