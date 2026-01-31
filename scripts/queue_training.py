#!/usr/bin/env python3
"""Training queue manager - automatically start next run when current finishes.

Usage:
    python queue_training.py --wait-for <process_name> --then-run <script>

Example:
    python queue_training.py --wait-for train_fast_iterate.py --then-run train_ultrafast.py
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path

def is_process_running(process_name: str) -> bool:
    """Check if a process with given name is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", process_name],
            capture_output=True,
            text=True
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return False

def wait_for_completion(process_name: str, check_interval: int = 10):
    """Wait for a process to complete."""
    print(f"⏳ Waiting for '{process_name}' to finish...")
    print(f"   Checking every {check_interval} seconds")
    print(f"   Press Ctrl+C to cancel queue\n")
    
    wait_count = 0
    while is_process_running(process_name):
        wait_count += 1
        elapsed_min = (wait_count * check_interval) / 60
        print(f"\r⏳ Still running... ({elapsed_min:.1f} min elapsed)  ", end='', flush=True)
        time.sleep(check_interval)
    
    print(f"\n✅ Process '{process_name}' finished!\n")

def run_next(script_path: str):
    """Run the next training script."""
    script = Path(script_path)
    
    if not script.exists():
        print(f"❌ Error: Script not found: {script_path}")
        sys.exit(1)
    
    print(f"🚀 Starting next run: {script_path}\n")
    print("=" * 70)
    
    # Run with PYTHONPATH set
    env = {"PYTHONPATH": "."}
    
    try:
        # Run and stream output
        subprocess.run(
            ["uv", "run", "python", str(script)],
            env={**subprocess.os.environ, **env},
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(130)

def main():
    parser = argparse.ArgumentParser(
        description="Queue training runs - wait for one to finish, then start another"
    )
    parser.add_argument(
        "--wait-for",
        required=True,
        help="Process name to wait for (e.g., 'train_fast_iterate.py')"
    )
    parser.add_argument(
        "--then-run",
        required=True,
        help="Script to run after waiting (e.g., 'train_ultrafast.py')"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=10,
        help="How often to check if process finished (seconds)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 TRAINING QUEUE MANAGER")
    print("=" * 70)
    print(f"Waiting for: {args.wait_for}")
    print(f"Then run:    {args.then_run}")
    print("=" * 70 + "\n")
    
    try:
        # Wait for current process to finish
        wait_for_completion(args.wait_for, args.check_interval)
        
        # Small delay to ensure clean shutdown
        time.sleep(2)
        
        # Start next run
        run_next(args.then_run)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Queue cancelled by user")
        sys.exit(130)

if __name__ == "__main__":
    main()
