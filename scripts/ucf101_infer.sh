#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-/mnt/ucf101}
CKPT=${2:-checkpoints/checkpoint_step0.pt}

if [ ! -f "$CKPT" ]; then
  echo "Checkpoint not found: $CKPT"
  echo "Run training first or pass a valid checkpoint path."
  exit 1
fi

.venv/bin/python -m src.eval.ucf101_linear_probe \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CKPT"
