#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-/mnt/ssv2}
CACHE_DIR=${2:-/mnt/ssv2/cache_videomae_huge_384}
LIMIT=${3:-1000}

.venv/bin/python -m src.cache.build_teacher_cache \
  --data-root "$DATA_ROOT" \
  --split train \
  --cache-dir "$CACHE_DIR" \
  --limit "$LIMIT" \
  --batch-size 4 \
  --num-workers 4
