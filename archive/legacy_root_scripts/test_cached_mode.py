#!/usr/bin/env python3
"""Test script to validate both cached and non-cached training modes.

This script:
1. Tests non-cached mode (original, teacher computed on-the-fly)
2. Tests cached mode (pre-computed teacher latents)
3. Compares both modes to ensure they work correctly
"""

from __future__ import annotations

import os
import sys

# Prevent pytest from collecting this script as a test module.
__test__ = False

# Disable wandb for testing
os.environ['WANDB_MODE'] = 'disabled'


def run_mode(mode_name: str, use_cached: bool, max_steps: int = 50) -> bool:
    """Test a single mode (cached or non-cached)."""
    from src.train import train
    print("=" * 70)
    print(f"🧪 TESTING: {mode_name}")
    print("=" * 70)
    print()

    try:
        train(
            data_root="/mnt/ssv2",
            split="train",
            batch_size=16,
            num_workers=4,
            max_steps=max_steps,
            log_interval=10,
            use_cached_latents=use_cached,
            use_dataloader_masks=True,
            cache_dir="/mnt/ssv2/cached_latents_v1huge",
        )

        print()
        print("=" * 70)
        print(f"✅ {mode_name} - PASSED")
        print("=" * 70)
        print()
        return True

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ {mode_name} - FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    print("=" * 70)
    print("🔬 CACHED vs NON-CACHED MODE COMPARISON TEST")
    print("=" * 70)
    print()
    print("This test will:")
    print("  1. Test NON-CACHED mode (teacher computed on-the-fly)")
    print("  2. Test CACHED mode (pre-computed latents)")
    print("  3. Run 50 steps for each mode")
    print()
    print("=" * 70)
    print()

    results = {}

    # Test 1: Non-cached mode
    results['non_cached'] = run_mode(
        "NON-CACHED MODE (Original)",
        use_cached=False,
        max_steps=50
    )

    # Test 2: Cached mode
    results['cached'] = run_mode(
        "CACHED MODE (Pre-computed latents)",
        use_cached=True,
        max_steps=50
    )

    # Summary
    print()
    print("=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"Non-Cached Mode: {'✅ PASS' if results['non_cached'] else '❌ FAIL'}")
    print(f"Cached Mode:     {'✅ PASS' if results['cached'] else '❌ FAIL'}")
    print()

    if all(results.values()):
        print("🎉 ALL TESTS PASSED!")
        print()
        print("Both modes work correctly. You can use either:")
        print("  • use_cached_latents=False (default, original)")
        print("  • use_cached_latents=True (faster, uses pre-cached latents)")
        print()
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        if not results['non_cached']:
            print("Non-cached mode failed - there may be an issue with the model or data.")
        if not results['cached']:
            print("Cached mode failed - check cache directory and implementation.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
