#!/usr/bin/env python3
"""Sequential training queue: ViT-Base 10-epoch → ViT-Large 5-epoch.

This script runs training sequentially, analyzing results between runs.
Run with: PYTHONPATH=. uv run python queue_vitb_vitl.py

Queue order:
1. ViT-Base 10-epoch (train_vitb_10epoch.py)
   - 86M params, ~52K steps, ~2-2.5 hours
   - Predictor: 384-dim, 12 layers

2. ViT-Large 5-epoch (train_vitl_extreme.py)
   - 304M params, ~35K steps, ~3-4 hours
   - Predictor: 512-dim, 12 layers, 8 heads

Total estimated time: ~5-6.5 hours
"""

import subprocess
import sys
import time
import os
from datetime import datetime
from pathlib import Path

# Full paths for subprocess execution
UV_PATH = "/home/derekszen/.local/bin/uv"
PROJECT_ROOT = Path(__file__).parent.absolute()

def run_training(script_name: str, description: str) -> tuple[bool, float]:
    """Run a training script and return success status and duration."""
    print(f"\n{'='*60}")
    print(f"[QUEUE] Starting: {description}")
    print(f"[QUEUE] Script: {script_name}")
    print(f"[QUEUE] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n", flush=True)

    start_time = time.time()
    script_path = PROJECT_ROOT / script_name

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run(
            [UV_PATH, "run", "python", str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True,
        )
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[QUEUE] ERROR: {script_name} failed with return code {e.returncode}")
        success = False
    except Exception as e:
        print(f"[QUEUE] ERROR: {script_name} failed with exception: {e}")
        success = False

    duration = time.time() - start_time
    duration_hrs = duration / 3600

    status = "SUCCESS" if success else "FAILED"
    print(f"\n[QUEUE] {status}: {description}")
    print(f"[QUEUE] Duration: {duration_hrs:.2f} hours ({duration:.0f} seconds)", flush=True)

    return success, duration

def main():
    print(f"\n{'#'*60}")
    print(f"# SALT-VLA Training Queue")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    queue = [
        ("train_vitb_10epoch.py", "ViT-Base 10-epoch (86M params, predictor 384x12)"),
        ("train_vitl_extreme.py", "ViT-Large 5-epoch (304M params, predictor 512x12)"),
    ]

    results = []
    total_start = time.time()

    for script, description in queue:
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            print(f"[QUEUE] ERROR: Script not found: {script_path}")
            results.append((script, False, 0))
            continue

        success, duration = run_training(script, description)
        results.append((script, success, duration))

        if not success:
            print(f"[QUEUE] Stopping queue due to failure in {script}")
            break

    # Summary
    total_duration = time.time() - total_start
    total_hrs = total_duration / 3600

    print(f"\n{'#'*60}")
    print(f"# Training Queue Complete")
    print(f"# Total Duration: {total_hrs:.2f} hours")
    print(f"{'#'*60}")
    print(f"\nResults:")
    for script, success, duration in results:
        status = "✓" if success else "✗"
        hrs = duration / 3600
        print(f"  {status} {script}: {hrs:.2f}h")

    # Exit with error if any failed
    if not all(r[1] for r in results):
        sys.exit(1)

if __name__ == "__main__":
    main()
