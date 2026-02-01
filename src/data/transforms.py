from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoTransformConfig:
    size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


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

    # Resize to square
    frames = F.interpolate(
        frames, size=(config.size, config.size), mode="bilinear", align_corners=False
    )

    # Normalize
    mean = torch.tensor(config.mean, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(config.std, device=frames.device).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # [T, C, H, W] -> [C, T, H, W]
    frames = frames.permute(1, 0, 2, 3)
    return frames
