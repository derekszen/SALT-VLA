from __future__ import annotations

from pathlib import Path

import torch

from src.models.teacher_videomae import TeacherConfig, VideoMAETeacher
from src.utils.shapes import num_tokens


def test_teacher_shapes_and_projection(tmp_path: Path) -> None:
    config = TeacherConfig(
        projection_path=tmp_path / "proj.npy",
    )
    teacher = VideoMAETeacher(config, device=torch.device("cpu"))

    video = torch.randn(1, 3, 16, 224, 224)

    # Raw tokens from HF model
    with torch.no_grad():
        outputs = teacher.model(pixel_values=video.permute(0, 2, 1, 3, 4))
        raw = outputs.last_hidden_state

    expected_tokens = num_tokens(
        num_frames=config.num_frames,
        tubelet_size=config.tubelet_size,
        img_size=config.img_size,
        patch_size=config.patch_size,
    )

    assert raw.shape[-1] == config.teacher_dim
    assert raw.shape[1] in (expected_tokens, expected_tokens + 1)

    projected1 = teacher(video)
    projected2 = teacher(video)

    assert projected1.shape == (1, expected_tokens, config.target_dim)
    assert projected1.dtype == torch.float16
    assert torch.allclose(projected1, projected2)
