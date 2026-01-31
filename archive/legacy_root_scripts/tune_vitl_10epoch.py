#!/usr/bin/env python3
"""Analyze the latest ViT-L 10-epoch log and emit tuned hyperparameters."""

from __future__ import annotations

import argparse
import re
import statistics
import time
from pathlib import Path


LOSS_RE = re.compile(r"loss=([0-9.]+)")


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return statistics.mean(values)


def _find_latest_log(log_dir: Path, prefix: str) -> Path:
    logs = sorted(log_dir.glob(f"{prefix}*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SystemExit(f"No logs found for prefix: {prefix} in {log_dir}")
    return logs[-1]


def _parse_losses(log_path: Path) -> list[float]:
    losses: list[float] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = LOSS_RE.search(line)
            if match:
                losses.append(float(match.group(1)))
    return losses


def _format_lr(value: float) -> str:
    return f"{value:.1e}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="run_logs")
    parser.add_argument("--log-prefix", default="vitl_10epoch_v1huge_lr2e4_")
    parser.add_argument("--out", default="run_logs/vitl_10epoch_tuned.env")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_path = _find_latest_log(log_dir, args.log_prefix)
    losses = _parse_losses(log_path)
    if len(losses) < 10:
        raise SystemExit(f"Insufficient loss history in {log_path}")

    min_loss = min(losses)
    window = 50
    last_window = losses[-window:]
    prev_window = losses[-2 * window : -window] if len(losses) >= 2 * window else losses[:window]
    last_avg = _mean(last_window)
    prev_avg = _mean(prev_window)
    slope = last_avg - prev_avg

    lr = 2e-4
    min_lr = 1e-5
    warmup_steps = 2000
    mask_ratio = 0.75
    notes = []

    if slope > 0.15 or last_avg > min_loss + 0.6:
        lr *= 0.8
        min_lr *= 0.8
        warmup_steps = int(warmup_steps * 1.2)
        notes.append("reduce_lr")
    elif min_loss > 12.0 and slope <= 0.05:
        lr *= 1.2
        min_lr *= 1.2
        warmup_steps = int(warmup_steps * 1.1)
        notes.append("increase_lr")
    elif abs(slope) < 0.05 and last_avg - min_loss < 0.25:
        mask_ratio = 0.85
        notes.append("increase_mask_ratio")
    else:
        notes.append("keep_defaults")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"vitl_10epoch_tuned_{_format_lr(lr)}_mr{mask_ratio:.2f}_{stamp}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"RUN_NAME={run_name}\n")
        handle.write(f"LR={lr}\n")
        handle.write(f"MIN_LR={min_lr}\n")
        handle.write(f"WARMUP_STEPS={warmup_steps}\n")
        handle.write(f"MASK_RATIO={mask_ratio}\n")

    print(f"[tuner] log={log_path}")
    print(f"[tuner] min_loss={min_loss:.4f} last_avg={last_avg:.4f} prev_avg={prev_avg:.4f} slope={slope:.4f}")
    print(f"[tuner] lr={lr} min_lr={min_lr} warmup_steps={warmup_steps} mask_ratio={mask_ratio}")
    print(f"[tuner] notes={','.join(notes)}")
    print(f"[tuner] out={out_path}")


if __name__ == "__main__":
    main()
