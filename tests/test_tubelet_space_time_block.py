from __future__ import annotations

import timm
import torch

from src.models.tubelet_space_time import TubeletSpaceTimeBlock


def test_tubelet_space_time_block_runs_on_patches() -> None:
    vit = timm.create_model("vit_base_patch16_224", pretrained=False)
    attn = vit.blocks[0].attn

    st_attn = TubeletSpaceTimeBlock(attn, num_frames=16, tubelet_size=2)

    bsz = 2
    t_tokens = 16 // 2
    s_tokens = 14 * 14
    n_patches = t_tokens * s_tokens
    x = torch.randn(bsz, n_patches, vit.embed_dim)

    y = st_attn(x)
    assert y.shape == x.shape

    # Gradient should flow through the wrapped attention.
    y.mean().backward()


def test_tubelet_space_time_block_handles_cls_token() -> None:
    vit = timm.create_model("vit_base_patch16_224", pretrained=False)
    attn = vit.blocks[0].attn

    st_attn = TubeletSpaceTimeBlock(attn, num_frames=16, tubelet_size=2)

    bsz = 1
    t_tokens = 16 // 2
    s_tokens = 14 * 14
    n_patches = t_tokens * s_tokens
    x = torch.randn(bsz, 1 + n_patches, vit.embed_dim)

    y = st_attn(x)
    assert y.shape == x.shape


def test_tubelet_space_time_block_works_with_hf_videomae_attention() -> None:
    from transformers import VideoMAEConfig, VideoMAEModel

    cfg = VideoMAEConfig(image_size=224, num_frames=16, tubelet_size=2, patch_size=16)
    model = VideoMAEModel(cfg)

    # Patch a single layer to validate signature + reshape logic.
    model.encoder.layer[0].attention = TubeletSpaceTimeBlock(
        model.encoder.layer[0].attention,
        num_frames=cfg.num_frames,
        tubelet_size=cfg.tubelet_size,
    )

    pixel_values = torch.randn(2, cfg.num_frames, cfg.num_channels, cfg.image_size, cfg.image_size)
    out = model(pixel_values)

    # (B, N, C) where N = (T/tubelet) * (H/patch) * (W/patch)
    t_tokens = cfg.num_frames // cfg.tubelet_size
    s_tokens = (cfg.image_size // cfg.patch_size) * (cfg.image_size // cfg.patch_size)
    n = t_tokens * s_tokens
    assert out.last_hidden_state.shape == (2, n, cfg.hidden_size)
