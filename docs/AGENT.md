# Agent Notes

Health checks
- Verify a single batch shape before long runs:
  - PYTHONPATH=. ./.venv/bin/python -c "from src.data.ssv2_dataset import SSV2Dataset; d=SSV2Dataset('/mnt/ssv2'); v,_=d[0]; print(v.shape)"

Find active runs
- ps aux | rg -i "pretrain_jepa|build_teacher_cache|eval_ucf101|retrieval_msrvt"

Logs
- Training logs: stdout + JSONL in the run directory.

Common debug
- If cache build fails, confirm SSv2 path and annotation JSONs exist in /mnt/ssv2.
- If teacher download fails, confirm Hugging Face access and cached models.
