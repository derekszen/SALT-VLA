# SALT-VLA Operations

Quick start
- Safety check: PYTHONPATH=. uv run python safety_test.py
- Training (cached latents): PYTHONPATH=. uv run python train_vitb_exp1_higherlr_vicreg10.py
- Training (non-cached): PYTHONPATH=. uv run python src/train.py

Caching latents (VideoMAE v1-Huge)
- Cache train split:
  PYTHONPATH=. uv run python cache_teacher_latents.py --split train --cache_dir /mnt/ssv2/cached_latents_v1huge
- Dataset integrity test:
  PYTHONPATH=. uv run python -m pytest tests/test_cached_dataset_integrity.py -v

Monitoring
- Health check:
  python check_training_health.py <log_file>
- Save run summary:
  python check_training_health.py <log_file> > run_summary_$(date +%Y%m%d_%H%M%S).json
- Find active runs:
  ps aux | rg -i "train_.*\.py"

Run logging (recommended)
- Use RUN_NAME and RUN_LOG_DIR to capture stdout/stderr in run_logs/.
- Example detached launch:
  RUN_NAME=vitb_exp1_higherlr_vicreg10_v1huge \
  RUN_LOG_DIR=run_logs \
  setsid -f env PYTHONPATH=. uv run python train_vitb_exp1_higherlr_vicreg10.py > /tmp/setsid_train_vicreg10.log 2>&1 &

DataLoader masking
- Mask generation moved into DataLoader via masked dataset classes.
- Use use_dataloader_masks=True for loader-provided masks.
- Rollback: set use_dataloader_masks=False or use src/data/loader_legacy.py.
