from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from decord import VideoReader, cpu


def get_video_length(path: str | Path) -> int:
    vr = VideoReader(str(path), ctx=cpu(0))
    return len(vr)


def decode_video(
    path: str | Path,
    frame_indices: Sequence[int],
) -> torch.Tensor:
    """Decode selected frames from a video using decord.

    Returns a uint8 tensor with shape [T, H, W, C].
    """
    if len(frame_indices) == 0:
        raise ValueError("frame_indices must be non-empty")
    vr = VideoReader(str(path), ctx=cpu(0))
    idx = np.asarray(frame_indices, dtype=np.int64)
    frames = vr.get_batch(idx).asnumpy()  # [T, H, W, C], uint8
    return torch.from_numpy(frames)


def sample_frame_indices(
    *,
    num_frames: int,
    total_frames: int,
    seed: int,
    mode: str = "random",
) -> np.ndarray:
    """Sample frame indices deterministically given a seed.

    mode:
      - "random": random contiguous clip
      - "uniform": uniform sampling over full video
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if total_frames <= 0:
        return np.zeros((num_frames,), dtype=np.int64)

    rng = np.random.default_rng(seed)

    if total_frames >= num_frames:
        if mode == "random":
            max_start = total_frames - num_frames
            start = int(rng.integers(0, max_start + 1))
            indices = np.arange(start, start + num_frames, dtype=np.int64)
        elif mode == "uniform":
            indices = np.linspace(0, total_frames - 1, num_frames)
            indices = np.round(indices).astype(np.int64)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
    else:
        # Pad by repeating the last frame
        indices = np.arange(total_frames, dtype=np.int64)
        pad = np.full((num_frames - total_frames,), total_frames - 1, dtype=np.int64)
        indices = np.concatenate([indices, pad], axis=0)

    return indices
