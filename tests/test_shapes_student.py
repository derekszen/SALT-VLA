from __future__ import annotations

import torch

from src.models.st_videomae_student import StudentConfig, STVideoMAEStudent
from src.utils.shapes import num_tokens


def test_student_shapes_and_finite() -> None:
    config = StudentConfig()
    model = STVideoMAEStudent(config)

    video = torch.randn(2, 3, 16, 224, 224)
    out = model(video)

    expected = num_tokens(
        num_frames=config.num_frames,
        tubelet_size=config.tubelet_size,
        img_size=config.img_size,
        patch_size=config.patch_size,
    )

    assert out.shape == (2, expected, config.embed_dim)
    assert torch.isfinite(out).all()
