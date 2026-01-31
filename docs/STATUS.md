# SALT-VLA Status

Last updated: 2026-01-09

Project goal
- Train a video encoder backbone for VLA via SALT: frozen teacher targets + masked latent prediction (no EMA teacher updates).

Current architecture (Stage 2 / student training)
- Teacher targets: cached VideoMAE latents (`/mnt/ssv2/cached_latents_v1huge`, latent_shape=[1568, 768])
- Student: ViT-Large patch16 224
- Predictor: 12-layer transformer (predictor_dim=512, heads=8)
- Loss: MSE on masked teacher latents (optional VICReg-style variance/cov penalties)
- Masking: multi-block masks generated in the DataLoader (use_dataloader_masks=True)

Latest completed run
- Script: `train_vitl_10epoch.py` (10 epochs)
- RUN_NAME: `vitl_10epoch_v1huge_lr2e4`
- Log: `run_logs/vitl_10epoch_v1huge_lr2e4_20260107_084240_pid1134145.log`
- Result: min loss 9.7319 @ step 42800, but late-training rebound (end loss 11.1865)

Most likely cause of rebound
- Weight decay schedule was being applied to all trainable params (including norms/bias/pos/mask tokens). This is a common ViT gotcha and can degrade late-training behavior.

Fixes landed
- `src/train.py`: AdamW param groups now exclude bias/norm/pos_embed/mask_token/cls_token from weight decay; only “decay” params follow the WD schedule.
- `src/models/salt.py` + `src/train.py`: cached-latents mode now skips teacher loading (`load_teacher=False`) and infers teacher_dim from cache metadata.

Compute note
- NVML/CUDA can fail inside sandboxed processes (seccomp / `NoNewPrivs=1`) even when they work in a normal shell. See `docs/nvidia-modprobe.md`.

Most recent run
- RUN_NAME: `vitl_10epoch_wdgroups_lr2e4_bgtest`
- Log: `run_logs/vitl_10epoch_wdgroups_lr2e4_bgtest_20260109_154049_pid1618342.log`
- Result: min loss 9.7144 @ step 69080; last logged 10.0499 @ step 70360; rebound largely gone vs prior ViT-L 10-epoch.

## 2026-01-12 - ViT-L 20-epoch follow-up

**Context:** Extend the best 10-epoch WD-group run to 20 epochs to test whether length alone lowers the loss floor.  
**Issue:** Epoch-20 avg loss (10.1201) and per-step lows (~9.56) are not clearly better than the 10-epoch baseline.  
**Solution:** Test schedule/regularization loosened (raise LR/min_lr and relax grad clip) on a short run before committing to longer training.

## 2026-01-13 - ViT-L 3-epoch schedule sanity

**Context:** Short A/B to test a looser schedule (LR/min_lr up, grad clip relaxed) before committing to long runs.  
**Issue:** 3-epoch avg loss 11.3528; needs validation/probe checks to judge early slope vs baseline.  
**Solution:** Run `scripts/eval_distill_loss.py` + temporal probe suite from a normal shell (GPU + network). The agent sandbox blocks HF downloads and multiprocessing semaphores.
