#!/usr/bin/env python3
"""Safety validation test - Run this BEFORE starting full training.

This script runs a 100-step test to ensure:
1. No system freezes occur
2. Memory usage stays safe
3. Workers remain stable
4. No disk quota errors

If this passes, your system should be safe for long training runs.
"""

from __future__ import annotations

import sys
import os
import argparse

# Disable wandb for testing
os.environ['WANDB_MODE'] = 'disabled'

from src.train import train

def main():
    parser = argparse.ArgumentParser(description="Safety validation test")
    parser.add_argument("--auto", action="store_true", help="Skip interactive prompt")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🛡️  SAFETY VALIDATION TEST")
    print("=" * 70)
    print()
    print("This will run 100 training steps to validate system stability.")
    print("Watch for:")
    print("  • RAM usage staying below 90%")
    print("  • No worker crashes")
    print("  • No disk quota errors")
    print("  • Consistent throughput (>80 videos/sec expected)")
    print()
    print("If any issues occur, the system will abort automatically.")
    print("=" * 70)
    print()

    if not args.auto:
        input("Press ENTER to start safety test (or Ctrl+C to cancel)...")
        print()

    try:
        # Run with ultra-conservative settings for 100 steps
        train(
            data_root="/mnt/ssv2",
            split="train",
            batch_size=16,      # Ultra-conservative
            num_workers=4,      # Ultra-conservative
            max_steps=100,      # Short test
            log_interval=10,    # Frequent logging
        )

        print()
        print("=" * 70)
        print("✅ SAFETY TEST PASSED!")
        print("=" * 70)
        print()
        print("Your system is stable for long training runs.")
        print("You can now start full training with:")
        print("  PYTHONPATH=. uv run python src/train.py")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except RuntimeError as e:
        print("\n\n" + "=" * 70)
        print("❌ SAFETY TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nRECOMMENDATIONS:")
        print("  1. Close other applications to free RAM")
        print("  2. Reduce num_workers to 2 in src/train.py")
        print("  3. Check /mnt/ssv2/.torch_tmp has space")
        print()
        sys.exit(1)
    except Exception as e:
        print("\n\n" + "=" * 70)
        print("❌ UNEXPECTED ERROR")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
