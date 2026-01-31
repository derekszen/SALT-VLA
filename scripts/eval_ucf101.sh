#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-/mnt/ucf101}
CKPT=${2:-checkpoints/checkpoint_step0.pt}

.venv/bin/python -m src.eval.ucf101_linear_probe \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CKPT"
