#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_score(suite: dict) -> float:
    results = suite.get("results") or {}
    scores = []
    for name, payload in results.items():
        probe = payload.get("probe") or {}
        if isinstance(probe, dict) and probe.get("name") == "clip-consistency":
            val_stats = probe.get("val") or {}
            if "top1" in val_stats:
                scores.append(float(val_stats["top1"]))
            continue

        probe_metrics = payload.get("probe") or payload.get("probe_metrics") or {}
        val_acc = probe_metrics.get("val_acc")
        if val_acc is not None:
            scores.append(float(val_acc))
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def _distill_score(distill: dict) -> float:
    return float(distill.get("mse_over_teacher_energy", distill.get("mse", float("nan"))))


def _load_config_for_checkpoint(ckpt_path: Path) -> dict:
    config_path = ckpt_path.parent / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def _wait_for(paths: list[Path], poll_sec: float, timeout_sec: float | None) -> None:
    start = time.time()
    while True:
        missing = [p for p in paths if not p.exists()]
        if not missing:
            return
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            raise TimeoutError(f"Timed out waiting for: {', '.join(str(p) for p in missing)}")
        time.sleep(poll_sec)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Wait for metrics artifacts, then choose a run and queue the next training job."
    )
    ap.add_argument("--distill-a", type=Path, required=True, help="JSON from eval_distill_loss.py")
    ap.add_argument("--distill-b", type=Path, required=True, help="JSON from eval_distill_loss.py")
    ap.add_argument("--probe-a", type=Path, required=True, help="Temporal probe suite JSON")
    ap.add_argument("--probe-b", type=Path, required=True, help="Temporal probe suite JSON")
    ap.add_argument("--poll-sec", type=float, default=60.0)
    ap.add_argument("--timeout-sec", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight for probe score vs distill loss")
    ap.add_argument("--next-epochs", type=int, default=20)
    ap.add_argument("--train-script", type=str, default="train_vitl_10epoch.py")
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--systemd", action="store_true", help="Use systemd-run --user to launch")
    args = ap.parse_args()

    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else (repo_root / path).resolve()

    distill_a = _resolve(args.distill_a)
    distill_b = _resolve(args.distill_b)
    probe_a = _resolve(args.probe_a)
    probe_b = _resolve(args.probe_b)

    _wait_for([distill_a, distill_b, probe_a, probe_b], args.poll_sec, args.timeout_sec)

    dist_a = _read_json(distill_a)
    dist_b = _read_json(distill_b)
    suite_a = _read_json(probe_a)
    suite_b = _read_json(probe_b)

    dist_score_a = _distill_score(dist_a)
    dist_score_b = _distill_score(dist_b)
    probe_score_a = _probe_score(suite_a)
    probe_score_b = _probe_score(suite_b)

    # Lower distill score is better, higher probe score is better.
    combined_a = (-dist_score_a) + args.alpha * probe_score_a
    combined_b = (-dist_score_b) + args.alpha * probe_score_b
    winner = "a" if combined_a >= combined_b else "b"

    chosen_distill = dist_a if winner == "a" else dist_b
    chosen_suite = suite_a if winner == "a" else suite_b
    chosen_ckpt = Path(chosen_distill.get("checkpoint", ""))
    if not chosen_ckpt.exists():
        raise FileNotFoundError(f"Chosen checkpoint not found: {chosen_ckpt}")

    cfg = _load_config_for_checkpoint(chosen_ckpt)
    weight_decay_end = float(cfg.get("weight_decay_end", 0.4))
    weight_decay_start = float(cfg.get("weight_decay_start", 0.04))

    run_name = args.run_name
    if not run_name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"vitl_{args.next_epochs}epoch_postprobe_{winner}_{stamp}"

    env = {
        "RUN_NAME": run_name,
        "EPOCHS": str(args.next_epochs),
        "WEIGHT_DECAY_START": str(weight_decay_start),
        "WEIGHT_DECAY_END": str(weight_decay_end),
    }

    script_path = repo_root / args.train_script
    cmd = [sys.executable, str(script_path)]

    payload = {
        "distill": {"a": dist_score_a, "b": dist_score_b},
        "probe": {"a": probe_score_a, "b": probe_score_b},
        "combined": {"a": combined_a, "b": combined_b},
        "winner": winner,
        "chosen_checkpoint": str(chosen_ckpt),
        "chosen_probe_suite": str(chosen_suite.get("checkpoint", "")),
        "next_run": {"run_name": run_name, "epochs": args.next_epochs, "env": env},
    }
    print(json.dumps(payload, indent=2))

    if args.dry_run:
        return

    if args.systemd:
        systemd_cmd = [
            "systemd-run",
            "--user",
            "--collect",
            "--property=WorkingDirectory=" + str(repo_root),
        ]
        systemd_cmd += [f"--setenv={k}={v}" for k, v in env.items()]
        systemd_cmd += cmd
        subprocess.run(systemd_cmd, check=True)
    else:
        # Fall back to running directly in this process.
        child_env = dict(os.environ)
        child_env.update(env)
        subprocess.run(cmd, cwd=str(repo_root), env=child_env, check=True)


if __name__ == "__main__":
    main()
