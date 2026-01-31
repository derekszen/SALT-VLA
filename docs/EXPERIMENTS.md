# SALT-VLA Experiments

Summary of key runs and outcomes.

| Date | Run | Key changes | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-01-04 | ViT-Tiny baseline | lr=1e-4 | Plateau ~12.5 | LR too low |
| 2026-01-04 | ViT-Tiny LR fix | lr=3e-4, min_lr=3e-5 | Min 10.73, final ~11.3 | Higher LR helps |
| 2026-01-05 | ViT-Base 5-epoch | Paper-aligned params | Min 10.86, final ~11.5 | Scaling works |
| 2026-01-05 | ViT-Base 10-epoch | Longer run | Min 10.76, final ~11.4 | More epochs did not help |
| 2026-01-05 | Exp1 higher LR | lr=3e-4 | Min 10.40 | Best pre v1-huge |
| 2026-01-06 | v1-huge cache + masks | Loader masks, teacher v1-huge | Cache complete | Latents at /mnt/ssv2/cached_latents_v1huge |
| 2026-01-06 | Exp1 higher LR (v1-huge) | v1-huge cache | Best 10.499, last-100 avg ~11.575 | Plateau persists |
| 2026-01-06 | Exp1 + VICReg (10-epoch) | variance/covariance penalties | In progress | RUN_NAME=vitb_exp1_higherlr_vicreg10_v1huge |
| 2026-01-07 | ViT-Large 5-epoch | student=ViT-L, predictor_dim=512, depth=12 | Min ~10.23 | Better min, still rebounds late |
| 2026-01-07 | ViT-Large 10-epoch | lr=2e-4, warmup=2000 | Min 9.7319 @ step 42800; end 11.1865 | Late-training rebound suggests WD issue |
| 2026-01-09 | ViT-Large 10-epoch (WD groups) | AdamW param groups (exclude norm/bias/pos/mask tokens from WD) | Min 9.7144 @ step 69080; last logged 10.0499 @ step 70360 | Rebound largely fixed vs prior ViT-L 10-epoch |
| 2026-01-12 | ViT-Large 20-epoch (WD groups) | epochs=20 (same schedule as bgtest) | Epoch-20 avg 10.1201; per-step dips ~9.56 | Still not clearly better than 10-epoch baseline |
| 2026-01-13 | ViT-Large 3-epoch schedule sanity | lr=3e-4, min_lr=3e-5, grad_clip=0.1 | Epoch-3 avg 11.3528 | Quick check of early slope with looser schedule |

Next ablations to consider
- mask_ratio=0.9 (train_vitb_exp2_highmask.py)
- combined LR + mask_ratio=0.9 (train_vitb_exp3_combined.py)
- tune variance_loss_weight/covariance_loss_weight
- weight decay param-grouping (exclude bias/norm/pos/mask tokens from WD) + consider lower weight_decay_end for student stage
