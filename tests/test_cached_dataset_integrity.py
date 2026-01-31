from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

from src.data.cached_loader import CachedSSv2Dataset
from src.data.loader import SSv2Dataset


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolve_root(candidates: list[str | None]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def test_cached_latents_id_alignment_hash() -> None:
    data_root = _resolve_root(
        [
            os.environ.get("SALT_DATA_ROOT"),
            os.environ.get("SSV2_DATA_ROOT"),
            "/mnt/ssv2",
        ]
    )
    cache_root = _resolve_root(
        [
            os.environ.get("SALT_CACHE_DIR"),
            "/mnt/ssv2/cached_latents",
        ]
    )
    split = "train"

    if data_root is None or cache_root is None:
        pytest.skip("SSv2 data or cache root not available")

    cache_split = cache_root / split
    if not cache_split.exists():
        pytest.skip(f"Cached latents not found: {cache_split}")

    split_json = data_root / f"{split}.json"
    repo_root = Path(__file__).resolve().parents[1]
    fallback_json = repo_root / "ssv2" / f"{split}.json"
    if not split_json.exists() and not fallback_json.exists():
        pytest.skip(f"Missing split metadata for {split}")

    videos_dir = data_root / "videos"
    has_videos_dir = videos_dir.exists()
    has_webm = False
    if not has_videos_dir:
        try:
            next(data_root.glob("*.webm"))
            has_webm = True
        except StopIteration:
            has_webm = False
    if not has_videos_dir and not has_webm:
        pytest.skip("SSv2 videos not available at data root")

    video_dataset = SSv2Dataset(root_dir=data_root, split=split)
    cached_dataset = CachedSSv2Dataset(cache_dir=cache_root, split=split)

    sample_count = min(100, len(video_dataset), len(cached_dataset))
    assert sample_count > 0, "No samples available to verify"

    assert (
        video_dataset.video_ids[:sample_count]
        == cached_dataset.video_ids[:sample_count]
    ), "Cached video ID order differs from video dataset order"

    for idx in range(sample_count):
        expected_id = video_dataset.video_ids[idx]
        video_path = video_dataset.videos_dir / f"{expected_id}.webm"
        latent_path = cache_split / f"{expected_id}.pt"

        assert video_path.exists(), f"Missing video file: {video_path}"
        assert video_path.stat().st_size > 0, f"Empty video file: {video_path}"
        assert latent_path.exists(), f"Missing cached latent: {latent_path}"
        assert latent_path.stat().st_size > 0, f"Empty cached latent: {latent_path}"

        payload = torch.load(latent_path, map_location="cpu")
        cached_id = payload.get("video_id")
        del payload

        assert cached_id is not None, f"Missing video_id metadata: {latent_path}"
        assert _sha256_text(expected_id) == _sha256_text(
            cached_id
        ), f"Cached video_id mismatch at index {idx}"
        assert _sha256_text(expected_id) == _sha256_text(
            video_path.stem
        ), f"Video filename mismatch at index {idx}"
