from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoTransformConfig:
    size: int = 224
    resize_mode: str = "short_side"  # "short_side" or "square"
    crop_mode: str = "center"  # "center" or "none"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


def transform_hash(config: VideoTransformConfig) -> str:
    payload = asdict(config)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def apply_video_transform(
    frames: torch.Tensor,
    config: VideoTransformConfig | None = None,
) -> torch.Tensor:
    """Apply basic resize + normalize to decoded frames.

    Args:
        frames: uint8 tensor [T, H, W, C]
        config: transform configuration

    Returns:
        float32 tensor [C, T, H, W]
    """
    if config is None:
        config = VideoTransformConfig()

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape [T, H, W, 3]")

    # [T, H, W, C] -> [T, C, H, W]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0

    if config.resize_mode == "square":
        frames = F.interpolate(
            frames, size=(config.size, config.size), mode="bilinear", align_corners=False
        )
    elif config.resize_mode == "short_side":
        _, _, h, w = frames.shape
        short = min(h, w)
        if short <= 0:
            raise ValueError("invalid frame size")
        scale = float(config.size) / float(short)
        new_h = max(int(round(h * scale)), config.size)
        new_w = max(int(round(w * scale)), config.size)
        frames = F.interpolate(
            frames, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        if config.crop_mode == "center":
            top = max((new_h - config.size) // 2, 0)
            left = max((new_w - config.size) // 2, 0)
            frames = frames[:, :, top : top + config.size, left : left + config.size]
        elif config.crop_mode != "none":
            raise ValueError(f"Unknown crop_mode: {config.crop_mode}")
    else:
        raise ValueError(f"Unknown resize_mode: {config.resize_mode}")

    if frames.shape[-2:] != (config.size, config.size):
        raise ValueError(
            f"Unexpected transformed size: got {tuple(frames.shape[-2:])} expected {(config.size, config.size)}"
        )

    # Normalize
    mean = torch.tensor(config.mean, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(config.std, device=frames.device).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # [T, C, H, W] -> [C, T, H, W]
    frames = frames.permute(1, 0, 2, 3)
    return frames
