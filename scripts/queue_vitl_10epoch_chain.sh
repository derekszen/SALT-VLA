#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="${RUN_LOG_DIR:-run_logs}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_vitl_10epoch.py}"

WAIT_PREFIX="${1:-}"
if [[ -z "$WAIT_PREFIX" ]]; then
  echo "Usage: $0 <run_name_prefix_ending_with_underscore>"
  echo "Example: $0 vitl_10epoch_wdgroups_lr2e4_bgtest_"
  exit 2
fi

wait_log() {
  local prefix="$1"
  local latest
  latest="$(ls -t "$LOG_DIR/${prefix}"*.log 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "[queue] No log found for prefix=$prefix in $LOG_DIR"
    return 1
  fi
  echo "[queue] Waiting for completion: $latest"
  tail -n 0 -F "$latest" | while IFS= read -r line; do
    if [[ "$line" == *"Process exiting."* ]]; then
      break
    fi
  done
}

run_train() {
  echo "[queue] Starting: RUN_NAME=${RUN_NAME:-unset} LR=${LR:-unset} MIN_LR=${MIN_LR:-unset} WARMUP_STEPS=${WARMUP_STEPS:-unset} MASK_RATIO=${MASK_RATIO:-unset}"
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-off}" \
  PYTHONPATH="${PYTHONPATH:-.}" \
    "$PY" "$TRAIN_SCRIPT"
}

echo "[queue] root=$ROOT_DIR"
echo "[queue] python=$PY"
echo "[queue] log_dir=$LOG_DIR"
echo "[queue] train_script=$TRAIN_SCRIPT"

# Run 1: wait for an already-running job (identified by log prefix).
wait_log "$WAIT_PREFIX"

# Run 2: tuned based on run 1.
"$PY" tune_vitl_10epoch.py --log-dir "$LOG_DIR" --log-prefix "$WAIT_PREFIX" --out "$LOG_DIR/vitl_10epoch_tuned.env"
set -a
. "$LOG_DIR/vitl_10epoch_tuned.env"
set +a
run_train

# Run 3: tuned again from run 2, then apply a small variant perturbation.
"$PY" tune_vitl_10epoch.py --log-dir "$LOG_DIR" --log-prefix "${RUN_NAME}_" --out "$LOG_DIR/vitl_10epoch_tuned2.env"
"$PY" tune_vitl_variant.py \
  --in "$LOG_DIR/vitl_10epoch_tuned2.env" \
  --out "$LOG_DIR/vitl_10epoch_variant.env" \
  --lr-scale 0.9 \
  --min-lr-scale 0.9 \
  --mask-delta 0.05 \
  --warmup-scale 1.1 \
  --name-suffix variant
set -a
. "$LOG_DIR/vitl_10epoch_variant.env"
set +a
run_train

echo "[queue] Done."
