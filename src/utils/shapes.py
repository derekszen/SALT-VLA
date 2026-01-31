from __future__ import annotations


def num_tokens(
    *,
    num_frames: int = 16,
    tubelet_size: int = 2,
    img_size: int = 224,
    patch_size: int = 16,
) -> int:
    t = num_frames // tubelet_size
    h = img_size // patch_size
    w = img_size // patch_size
    return t * h * w
