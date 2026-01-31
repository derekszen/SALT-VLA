#!/usr/bin/env bash
set -euo pipefail

MSR_VTT_ROOT=${1:-/mnt/msrvtt}
MSVD_ROOT=${2:-/mnt/msvd}
CKPT=${3:-checkpoints/checkpoint_step0.pt}

.venv/bin/python -m src.eval.retrieval_msrvt_msvd \
  --msrvtt-root "$MSR_VTT_ROOT" \
  --msvd-root "$MSVD_ROOT" \
  --checkpoint "$CKPT"
