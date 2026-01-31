#!/usr/bin/env python3
"""Thin CLI wrapper around src.train.train.

This is the recommended entrypoint after archiving many ad-hoc root scripts.
It also exposes architecture toggles for the new hybrid ST + SALT target path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.train import train  # noqa: E402


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean: {value}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SALT student on SSv2.")
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/ssv2"))
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--log-dir", type=Path, default=Path("run_logs"))
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))

    ap.add_argument("--use-cached-latents", type=str, default="0")
    ap.add_argument("--cache-dir", type=Path, default=Path("/mnt/ssv2/cached_latents_v1huge"))

    ap.add_argument("--student-model-name", type=str, default="vit_base_patch16_224")
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--masking-strategy", type=str, default="multiblock")

    # Architecture toggles (env-backed in src.train.py)
    ap.add_argument(
        "--student-space-time-blocks",
        type=int,
        default=0,
        help="Number of early blocks to replace with divided (time->space) attention.",
    )
    ap.add_argument(
        "--use-predictor",
        type=str,
        default="1",
        help="If 0: dense mask-token path + linear head. If 1: V-JEPA predictor path.",
    )

    args = ap.parse_args()

    os.environ["SALT_STUDENT_SPACE_TIME_BLOCKS"] = str(args.student_space_time_blocks)
    os.environ["SALT_USE_PREDICTOR"] = "1" if _parse_bool(args.use_predictor) else "0"

    train(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        mask_ratio=args.mask_ratio,
        masking_strategy=args.masking_strategy,
        use_cached_latents=_parse_bool(args.use_cached_latents),
        cache_dir=args.cache_dir,
        student_model_name=args.student_model_name,
        run_name=args.run_name or None,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
