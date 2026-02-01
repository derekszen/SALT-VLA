#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-/mnt/ssv2}
WORK_DIR=${2:-/mnt/ssv2/st_jepa_phase_a}
CACHE_DIR="${WORK_DIR}/cache_videomae_huge_384_512"

mkdir -p "$WORK_DIR"
if [[ -e "$CACHE_DIR" ]]; then
  echo "Cache dir already exists: $CACHE_DIR"
  echo "Delete it or choose a different WORK_DIR."
  exit 1
fi

echo "[phase-a] building 512-clip cache shard..."
.venv/bin/python -m src.cache.build_teacher_cache \
  --data-root "$DATA_ROOT" \
  --split train \
  --cache-dir "$CACHE_DIR" \
  --limit 512 \
  --batch-size 4 \
  --num-workers 4 \
  --seed 0

echo "[phase-a] overfit 32 clips for 200 steps (assert >=30% loss drop)..."
.venv/bin/python -m src.train.pretrain_jepa \
  --data-root "$DATA_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --split train \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 30 \
  --lr 5e-4 \
  --weight-decay 0.0 \
  --mask-ratio 0.75 \
  --amp \
  --seed 0 \
  --overfit \
  --overfit-samples 32 \
  --overfit-iters 200

