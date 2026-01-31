#!/usr/bin/env python3
"""Cache VideoMAE v1-Huge teacher latents to disk for faster training.

This script pre-computes all teacher latents and saves them to disk,
eliminating the need to run the frozen teacher model during training.

Expected speedup: 30-50% reduction in training time per epoch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import SSv2Dataset
from src.models.salt import SALTModel


def cache_latents(
    data_root: str | Path = "/mnt/ssv2",
    split: str = "train",
    cache_dir: str | Path = "/mnt/ssv2/cached_latents_v1huge",
    batch_size: int = 32,  # Can use larger BS since no backprop
    num_workers: int = 8,
    device_id: int = 0,
) -> None:
    """Pre-compute and cache teacher latents for all videos."""

    cache_dir = Path(cache_dir)
    split_cache_dir = cache_dir / split
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[cache] start={time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} "
        f"split={split} cache_dir={split_cache_dir}",
        flush=True,
    )
    print("=" * 70)
    print("🗄️  TEACHER LATENT CACHING (VideoMAE v1-Huge)")
    print("=" * 70)
    print(f"Split: {split}")
    print(f"Cache directory: {split_cache_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print("=" * 70)
    print()

    # Check if already cached
    metadata_file = split_cache_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        print(f"⚠️  Cache already exists with {metadata['num_samples']} samples")
        response = input("Continue and overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Setup
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    # Load dataset
    dataset = SSv2Dataset(data_root, split=split)
    print(f"[setup] {split} set: {len(dataset)} videos")

    # Create dataloader (no shuffle for deterministic caching)
    from src.data.loader import collate_drop_none
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic order
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_drop_none,
        drop_last=False,
    )

    # Load teacher model only
    print("[setup] Loading VideoMAE v1-Huge teacher...")
    model = SALTModel(dtype=torch.bfloat16)
    model.teacher.eval()
    model.teacher.to(device=device, dtype=torch.bfloat16)
    print(f"[setup] Teacher loaded: {model.teacher.__class__.__name__}")

    # Cache latents
    print()
    print("Starting latent extraction...")
    print()

    cached_count = 0
    total_time = 0
    video_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching batches")):
            if batch is None or batch.numel() == 0:
                continue

            start_time = time.time()

            # Move to device
            batch = batch.to(device=device, dtype=torch.bfloat16, non_blocking=True)

            # Extract teacher latents
            teacher_latents = model._teacher_tokens(batch)  # (B, N, D)

            # Save each video's latents individually
            for i in range(batch.shape[0]):
                latent_path = split_cache_dir / f"{dataset.video_ids[video_idx]}.pt"
                torch.save(
                    {
                        'latents': teacher_latents[i].cpu(),  # (N, D)
                        'video_id': dataset.video_ids[video_idx],
                        'shape': tuple(batch[i].shape),  # (C, T, H, W)
                    },
                    latent_path
                )
                video_idx += 1
                cached_count += 1

            batch_time = time.time() - start_time
            total_time += batch_time

            if (batch_idx + 1) % 10 == 0:
                fps = batch.shape[0] / batch_time
                avg_fps = cached_count / total_time
                print(f"  Batch {batch_idx+1}: {fps:.1f} videos/sec (avg: {avg_fps:.1f} v/s)")

    # Save metadata
    metadata = {
        'split': split,
        'num_samples': cached_count,
        'model': 'Tianjiao-Yu/videomae-huge',
        'latent_shape': tuple(teacher_latents[0].shape),  # (N, D)
        'video_ids': dataset.video_ids,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 70)
    print("✅ CACHING COMPLETE")
    print("=" * 70)
    print(f"Cached: {cached_count} videos")
    print(f"Time: {total_time:.1f}s ({cached_count/total_time:.1f} videos/sec)")
    print(f"Location: {split_cache_dir}")
    print(f"Metadata: {metadata_file}")
    print()
    print("To use cached latents during training, use CachedSSv2Dataset instead of SSv2Dataset")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache VideoMAE v1-Huge teacher latents")
    parser.add_argument("--data_root", type=str, default="/mnt/ssv2", help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/ssv2/cached_latents_v1huge",
        help="Cache directory",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for caching")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")

    args = parser.parse_args()

    cache_latents(
        data_root=args.data_root,
        split=args.split,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_id=args.device,
    )
