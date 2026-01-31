#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _parse_probes(value: str) -> list[str]:
    probes = [p.strip() for p in value.split(",") if p.strip()]
    if not probes:
        raise ValueError("No probes specified.")
    return probes


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Run a suite of SSv2 temporal probes for a checkpoint.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/ssv2"))
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="validation")
    ap.add_argument("--train-videos", type=int, default=2000)
    ap.add_argument("--val-videos", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-epochs", type=int, default=25)
    ap.add_argument("--probe-lr", type=float, default=5e-3)
    ap.add_argument("--probe-weight-decay", type=float, default=0.0)
    ap.add_argument("--subclip-ratio", type=float, default=0.75)
    ap.add_argument(
        "--probes",
        type=str,
        default="time-arrow,motion-static,temporal-shuffle,stride,clip-consistency",
        help="Comma-separated list.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Combined JSON output path")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not ckpt.is_absolute():
        ckpt = (repo_root / ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    suite_name = "ssv2_temporal_probe_suite"
    stem = ckpt.parent.name
    out_path = args.out
    if out_path is None:
        out_path = repo_root / "benchmarks" / "temporal_suite" / f"{stem}.json"
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    probes = _parse_probes(args.probes)

    results: dict[str, object] = {}
    per_probe_dir = out_path.parent / stem
    per_probe_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for probe in probes:
        out_probe = per_probe_dir / f"{probe.replace('-', '_')}.json"
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "bench_time_arrow_ssv2.py"),
            "--probe",
            probe,
            "--checkpoint",
            str(ckpt),
            "--data-root",
            str(args.data_root),
            "--train-split",
            args.train_split,
            "--val-split",
            args.val_split,
            "--train-videos",
            str(args.train_videos),
            "--val-videos",
            str(args.val_videos),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--probe-epochs",
            str(args.probe_epochs),
            "--probe-lr",
            str(args.probe_lr),
            "--probe-weight-decay",
            str(args.probe_weight_decay),
            "--subclip-ratio",
            str(args.subclip_ratio),
            "--out",
            str(out_probe),
        ]
        if args.dtype.strip():
            cmd.extend(["--dtype", args.dtype])

        subprocess.run(cmd, cwd=str(repo_root), check=True)
        results[probe] = json.loads(out_probe.read_text(encoding="utf-8"))

    suite = {
        "suite": suite_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(ckpt),
        "checkpoint_dir": str(ckpt.parent),
        "probes": probes,
        "seconds_total": float(time.time() - t0),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(suite, indent=2))


if __name__ == "__main__":
    main()
