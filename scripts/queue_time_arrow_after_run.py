#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _tail_contains(path: Path, needle: str, lines: int) -> bool:
    try:
        out = subprocess.run(
            ["tail", f"-n{lines}", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        text = (out.stdout or "") + (out.stderr or "")
        return needle in text
    except FileNotFoundError:
        # Fallback: read whole file (worst-case) if tail is missing.
        try:
            return needle in path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return False


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(
        description="Wait for a training run log to finish, then run SSv2 time-arrow probe."
    )
    ap.add_argument("--log", type=Path, required=True, help="Path to run_logs/<run>.log")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path (default: checkpoints/<log_stem>/best.pth)",
    )
    ap.add_argument(
        "--which",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which checkpoint to use if --checkpoint is not set.",
    )
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--needle", type=str, default="Process exiting.")
    ap.add_argument("--tail-lines", type=int, default=200)
    ap.add_argument("--train-videos", type=int, default=2000)
    ap.add_argument("--val-videos", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = ap.parse_args()

    log_path = args.log
    if not log_path.is_absolute():
        log_path = (repo_root / log_path).resolve()

    stem = log_path.stem
    if args.checkpoint is None:
        ckpt_path = repo_root / "checkpoints" / stem / f"{args.which}.pth"
    else:
        ckpt_path = args.checkpoint
        if not ckpt_path.is_absolute():
            ckpt_path = (repo_root / ckpt_path).resolve()

    if args.out is None:
        out_path = repo_root / "benchmarks" / "time_arrow" / f"{stem}.json"
    else:
        out_path = args.out
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()

    print(f"[time-arrow] repo_root={repo_root}")
    print(f"[time-arrow] waiting on log={log_path}")
    print(f"[time-arrow] will use checkpoint={ckpt_path}")
    print(f"[time-arrow] will write out={out_path}")

    while True:
        if log_path.exists() and _tail_contains(log_path, args.needle, args.tail_lines):
            break
        time.sleep(args.poll_sec)

    # Wait briefly for checkpoint flush.
    deadline = time.time() + 300.0
    while not ckpt_path.exists() and time.time() < deadline:
        time.sleep(2.0)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found after run completion: {ckpt_path}")

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "bench_time_arrow_ssv2.py"),
        "--checkpoint",
        str(ckpt_path),
        "--train-videos",
        str(args.train_videos),
        "--val-videos",
        str(args.val_videos),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--out",
        str(out_path),
    ]
    print(f"[time-arrow] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
