#!/usr/bin/env python3
"""Quick test to verify blacklist filtering works."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.cached_loader import HybridSSv2Dataset

def test_blacklist():
    print("Testing blacklist filtering...")

    # Create dataset with blacklist
    dataset = HybridSSv2Dataset(
        data_root="/mnt/ssv2",
        cache_dir="/mnt/ssv2/cached_latents",
        split="train",
        blacklist_file="/mnt/ssv2/corrupted_videos.txt",
    )

    # Check that blacklisted IDs are not in the dataset
    blacklisted = {"66115", "53291", "5281", "164113", "213572",
                   "90244", "184490", "219824", "145274", "131791"}

    video_ids_set = set(dataset.video_dataset.video_ids)
    found_blacklisted = blacklisted & video_ids_set

    if found_blacklisted:
        print(f"❌ FAIL: Found blacklisted IDs in dataset: {found_blacklisted}")
        return False

    print(f"✓ All {len(blacklisted)} blacklisted videos successfully filtered")
    print(f"✓ Dataset size: {len(dataset)} videos")

    # Try loading a few samples to ensure no warnings
    print("\nTesting sample loading (should see no warnings)...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        if sample is not None:
            video, latents = sample
            print(f"  Sample {i}: video {video.shape}, latents {latents.shape}")

    print("\n✓ Blacklist filtering working correctly!")
    return True

if __name__ == "__main__":
    success = test_blacklist()
    sys.exit(0 if success else 1)
