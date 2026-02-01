# SALT-VLA Operations

Quick start
- Create venv and install deps (uv):
  - uv venv .venv
  - uv pip install -r requirements.txt --python .venv/bin/python
- Verify environment (M0):
  - .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
  - .venv/bin/python -c "import transformers, einops, decord"

Cache build (SSv2)
- Build a 1k-sample cache shard:
  - ./scripts/build_cache_ssv2.sh
- Config defaults in configs/cache_ssv2.yaml (cache_dir, split, limit, batch_size).

Pretraining
- Single-GPU pretrain:
  - ./scripts/pretrain_ssv2.sh
- DDP (example):
  - torchrun --nproc_per_node=8 src/train/pretrain_jepa.py --config configs/pretrain_ssv2.yaml

Evaluation
- UCF101 linear probe:
  - ./scripts/eval_ucf101.sh
- Retrieval (MSR-VTT/MSVD):
  - ./scripts/eval_retrieval.sh

Logging
- Training logs are written to stdout and JSONL in the run directory.
