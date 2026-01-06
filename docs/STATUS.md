# SALT-VLA Status

Last updated: 2026-01-06 (from archived log)

Project goal
- Train a SoTA ViT-Base vision encoder for VLA using V-JEPA self-supervised learning on video.

Current architecture
- Teacher: VideoMAE v1-Huge (Tianjiao-Yu/videomae-huge)
- Student: ViT-Base patch16 224
- Predictor: 12-layer transformer (384 dim, 6 heads)
- Loss: MSE on masked latent predictions, with optional VICReg-style variance/covariance penalties

Dataset and cache
- Dataset: Something-Something v2 (168,903 videos)
- Cached latents: /mnt/ssv2/cached_latents_v1huge/train (metadata.json present)
- DataLoader masking: use_dataloader_masks=True (multi-block masks generated in DataLoader)

Current run
- Script: train_vitb_exp1_higherlr_vicreg10.py
- RUN_NAME: vitb_exp1_higherlr_vicreg10_v1huge
- Log: run_logs/vitb_exp1_higherlr_vicreg10_v1huge_20260106_165129_pid1197207.log
- WandB: c1sk8d6s
- Purpose: test VICReg-style penalties over 10 epochs

Best known results
- Higher LR (3e-4) improved min loss to ~10.40 (pre v1-huge cache run).
- v1-huge 3-epoch run plateaued around 11.57 with best ~10.50 (not sustained).

Immediate next steps
- Finish the VICReg 10-epoch run and review loss trend.
- If plateau persists, tune variance/covariance weights or try the combined LR + mask ratio 0.9 run (train_vitb_exp3_combined.py).
- Target loss for publication: <10.0.
