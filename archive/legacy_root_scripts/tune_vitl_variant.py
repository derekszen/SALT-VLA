#!/usr/bin/env python3
"""Adjust a tuned env file into a variant hyperparameter set."""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        key, _, value = line.partition("=")
        data[key.strip()] = value.strip()
    return data


def _format_lr(value: float) -> str:
    return f"{value:.1e}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--min-lr-scale", type=float, default=1.0)
    parser.add_argument("--mask-delta", type=float, default=0.0)
    parser.add_argument("--warmup-scale", type=float, default=1.0)
    parser.add_argument("--name-suffix", default="variant")
    args = parser.parse_args()

    env = _load_env(Path(args.input_path))

    lr = float(env.get("LR", "2e-4")) * args.lr_scale
    min_lr = float(env.get("MIN_LR", "1e-5")) * args.min_lr_scale
    warmup_steps = int(float(env.get("WARMUP_STEPS", "2000")) * args.warmup_scale)
    mask_ratio = min(0.9, float(env.get("MASK_RATIO", "0.75")) + args.mask_delta)

    run_name = env.get("RUN_NAME", "vitl_10epoch_tuned")
    run_name = f"{run_name}_{args.name_suffix}_{_format_lr(lr)}_mr{mask_ratio:.2f}"

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                f"RUN_NAME={run_name}",
                f"LR={lr}",
                f"MIN_LR={min_lr}",
                f"WARMUP_STEPS={warmup_steps}",
                f"MASK_RATIO={mask_ratio}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[variant] in={args.input_path} out={args.output_path}")
    print(f"[variant] lr={lr} min_lr={min_lr} warmup_steps={warmup_steps} mask_ratio={mask_ratio}")
    print(f"[variant] run_name={run_name}")


if __name__ == "__main__":
    main()
