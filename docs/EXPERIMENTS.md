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

Next ablations to consider
- mask_ratio=0.9 (train_vitb_exp2_highmask.py)
- combined LR + mask_ratio=0.9 (train_vitb_exp3_combined.py)
- tune variance_loss_weight/covariance_loss_weight
