#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-/mnt/ssv2}
CACHE_DIR=${2:-/mnt/ssv2/cache_videomae_huge_384}
shift || true
shift || true

EXTRA_ARGS=("$@")

.venv/bin/python -m src.train.pretrain_jepa \
  --data-root "$DATA_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --split train \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 4 \
  --lr 2e-4 \
  --weight-decay 0.05 \
  --mask-ratio 0.75 \
  --amp \
  "${EXTRA_ARGS[@]}"
