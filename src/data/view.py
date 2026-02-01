from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.transforms import VideoTransformConfig, apply_video_transform, transform_hash
from src.data.video_decode import decode_video, get_video_length, sample_frame_indices


def _stable_int_from_str(value: str) -> int:
    """Stable 32-bit unsigned int from an arbitrary string."""
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def video_id_to_int(video_id: str) -> int:
    """Convert a SSv2 video id to a stable integer.

    SSv2 ids are numeric strings in the official dataset, but we fall back to
    hashing to keep behavior stable for synthetic tests or alternative ids.
    """
    try:
        return int(video_id)
    except ValueError:
        return _stable_int_from_str(video_id)


@dataclass(frozen=True)
class VideoViewConfig:
    """Defines the deterministic "view" used both for caching and training."""

    num_frames: int = 16
    sample_mode: str = "random"
    seed_base: int = 0
    transform: VideoTransformConfig = field(default_factory=VideoTransformConfig)


def view_hash(config: VideoViewConfig) -> str:
    payload = asdict(config)
    # Avoid duplicating large structures; lock the transform by its hash.
    payload["transform_hash"] = transform_hash(config.transform)
    payload.pop("transform", None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def view_seed(video_id: str, seed_base: int) -> int:
    # Deterministic across shuffles/shards: seed depends on clip id, not index.
    return int(seed_base) + int(video_id_to_int(video_id))


def compute_view_meta(
    video_path: str | Path,
    video_id: str,
    config: VideoViewConfig,
) -> dict[str, Any]:
    total_frames = get_video_length(video_path)
    seed = view_seed(video_id, config.seed_base)
    frame_indices = sample_frame_indices(
        num_frames=config.num_frames,
        total_frames=total_frames,
        seed=seed,
        mode=config.sample_mode,
    )
    return {
        "video_id": str(video_id),
        "sample_seed": int(seed),
        "frame_indices": frame_indices.astype(np.int64).tolist(),
        "transform_hash": transform_hash(config.transform),
        "view_hash": view_hash(config),
    }


def build_view(
    video_path: str | Path,
    video_id: str,
    config: VideoViewConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode + transform a deterministic view, and return video + view meta.

    Returns:
        video: float32 tensor [C, T, H, W]
        meta: dict with keys video_id, sample_seed, frame_indices, transform_hash, view_hash
    """
    meta = compute_view_meta(video_path, video_id, config)
    frames = decode_video(video_path, meta["frame_indices"])
    video = apply_video_transform(frames, config.transform)
    return video, meta

