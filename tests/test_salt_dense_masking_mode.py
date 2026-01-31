from __future__ import annotations

import torch

from src.models.salt import SALTModel


def test_salt_dense_masking_forward_smoke() -> None:
    # Small spatial size for a fast CPU smoke test.
    img_size = 32
    patch_size = 16
    num_frames = 16
    tubelet_size = 2

    model = SALTModel(
        load_teacher=False,
        teacher_dim=8,
        student_model_name="vit_base_patch16_224",
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        mask_ratio=0.5,
        masking_strategy="random",
        student_space_time_blocks=2,
        use_predictor=False,
        dtype=torch.float32,
    )

    bsz = 2
    # Video input is (B, C, T, H, W)
    video = torch.randn(bsz, 3, num_frames, img_size, img_size)

    t_tokens = num_frames // tubelet_size
    s_tokens = (img_size // patch_size) * (img_size // patch_size)
    n_tokens = t_tokens * s_tokens
    cached_teacher_latents = torch.randn(bsz, n_tokens, model.teacher_dim)

    pred_masked, teacher_masked = model(video, cached_teacher_latents=cached_teacher_latents)

    assert pred_masked.shape == teacher_masked.shape
    assert pred_masked.shape[0] == bsz
    assert pred_masked.shape[2] == model.teacher_dim

