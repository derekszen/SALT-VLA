from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.cache.cache_format import CacheSpec, create_cache, write_meta_jsonl
from src.data.ssv2_dataset import SSV2Config, SSV2CachedDataset
from src.data.transforms import transform_hash
from src.data.view import VideoViewConfig, compute_view_meta, view_hash


def _write_dummy_video(path: Path, num_frames: int = 20, size: int = 64) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (size, size))
    assert writer.isOpened(), "Failed to open VideoWriter"
    for i in range(num_frames):
        frame = np.full((size, size, 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_cached_dataset_uses_video_id_mapping_and_validates_view(tmp_path: Path) -> None:
    data_root = tmp_path / "ssv2"
    data_root.mkdir(parents=True)

    video_id_a = "123"
    video_id_b = "124"
    _write_dummy_video(data_root / f"{video_id_a}.mp4", num_frames=24, size=64)
    _write_dummy_video(data_root / f"{video_id_b}.mp4", num_frames=24, size=64)

    split_path = data_root / "train.json"
    with split_path.open("w") as f:
        json.dump([{"id": video_id_a}, {"id": video_id_b}], f)

    view_cfg = VideoViewConfig(num_frames=16, sample_mode="random", seed_base=123)

    cache_dir = tmp_path / "cache"
    spec = CacheSpec(num_samples=2, num_tokens=1568, dim=384)
    arr = create_cache(
        cache_dir,
        spec,
        extra_meta={"view_hash": view_hash(view_cfg), "transform_hash": transform_hash(view_cfg.transform)},
    )

    # Fill cache rows with distinct values.
    arr[0] = np.zeros((1568, 384), dtype=np.float16)
    arr[1] = np.ones((1568, 384), dtype=np.float16)

    meta_a = compute_view_meta(data_root / f"{video_id_a}.mp4", video_id_a, view_cfg)
    meta_b = compute_view_meta(data_root / f"{video_id_b}.mp4", video_id_b, view_cfg)

    # Swap cache rows: a->1, b->0 to ensure mapping uses video_id, not dataset index.
    write_meta_jsonl(cache_dir, [{"index": 0, **meta_b}, {"index": 1, **meta_a}])

    ds = SSV2CachedDataset(
        SSV2Config(data_root=data_root, split="train", view=view_cfg, split_path=split_path),
        cache_dir=cache_dir,
    )

    video0, target0 = ds[0]
    video1, target1 = ds[1]

    assert video0.shape == (3, 16, 224, 224)
    assert video1.shape == (3, 16, 224, 224)

    assert target0.dtype == torch.float16
    assert target1.dtype == torch.float16

    # video_id_a maps to cache row 1 filled with ones.
    assert torch.all(target0 == 1)
    # video_id_b maps to cache row 0 filled with zeros.
    assert torch.all(target1 == 0)


def test_cached_dataset_raises_on_view_hash_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "ssv2"
    data_root.mkdir(parents=True)

    video_id = "123"
    _write_dummy_video(data_root / f"{video_id}.mp4", num_frames=24, size=64)

    split_path = data_root / "train.json"
    with split_path.open("w") as f:
        json.dump([{"id": video_id}], f)

    view_cfg = VideoViewConfig(num_frames=16, sample_mode="random", seed_base=123)

    cache_dir = tmp_path / "cache"
    spec = CacheSpec(num_samples=1, num_tokens=1568, dim=384)
    create_cache(
        cache_dir,
        spec,
        extra_meta={"view_hash": "bad", "transform_hash": transform_hash(view_cfg.transform)},
    )
    meta = compute_view_meta(data_root / f"{video_id}.mp4", video_id, view_cfg)
    write_meta_jsonl(cache_dir, [{"index": 0, **meta}])

    with pytest.raises(ValueError):
        _ = SSV2CachedDataset(
            SSV2Config(data_root=data_root, split="train", view=view_cfg, split_path=split_path),
            cache_dir=cache_dir,
        )

