from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.bench_time_arrow_ssv2 as ta  # noqa: E402


def _make_ramp_video(*, b: int = 1, c: int = 1, t: int = 8, h: int = 1, w: int = 1) -> torch.Tensor:
    x = torch.arange(t, dtype=torch.float32).view(1, 1, t, 1, 1)
    return x.expand(b, c, -1, h, w).clone()


def test_make_classification_batch_time_arrow() -> None:
    videos = _make_ramp_video(b=2, c=1, t=8)
    x, y, names = ta._make_classification_batch(videos, "time-arrow")
    assert names == ["forward", "reverse"]
    assert x.shape == (4, 1, 8, 1, 1)
    assert y.shape == (4,)
    assert int((y == 0).sum().item()) == 2
    assert int((y == 1).sum().item()) == 2
    assert torch.allclose(x[:2], videos)
    assert torch.allclose(x[2:], videos.flip(2))


def test_make_classification_batch_motion_static_is_static() -> None:
    videos = torch.randn(3, 2, 8, 4, 4)
    x, y, names = ta._make_classification_batch(videos, "motion-static")
    assert names == ["motion", "static"]
    b = videos.shape[0]
    static = x[b:]
    first = static[:, :, :1]
    assert torch.allclose(static, first.expand_as(static))
    assert int((y == 0).sum().item()) == b
    assert int((y == 1).sum().item()) == b


def test_make_classification_batch_temporal_shuffle_is_permutation() -> None:
    videos = _make_ramp_video(b=1, c=1, t=8)
    rng = torch.Generator().manual_seed(123)
    x, y, names = ta._make_classification_batch(videos, "temporal-shuffle", rng=rng)
    assert names == ["ordered", "shuffled"]
    ordered = x[0, 0, :, 0, 0]
    shuffled = x[1, 0, :, 0, 0]
    assert torch.allclose(ordered, torch.arange(8, dtype=torch.float32))
    assert torch.allclose(shuffled.sort().values, ordered)
    assert not torch.allclose(shuffled, ordered)
    assert int((y == 0).sum().item()) == 1
    assert int((y == 1).sum().item()) == 1


def test_make_classification_batch_stride_repeat_patterns() -> None:
    videos = _make_ramp_video(b=1, c=1, t=16)
    x, y, names = ta._make_classification_batch(videos, "stride")
    assert names == ["stride1", "stride2", "stride4"]
    stride1 = x[0, 0, :, 0, 0]
    stride2 = x[1, 0, :, 0, 0]
    stride4 = x[2, 0, :, 0, 0]
    assert torch.allclose(stride1, torch.arange(16, dtype=torch.float32))
    assert torch.allclose(stride2[::2], torch.arange(0, 16, 2, dtype=torch.float32))
    assert torch.allclose(stride2[1::2], stride2[::2])
    assert torch.allclose(stride4[::4], torch.tensor([0.0, 4.0, 8.0, 12.0]))
    assert torch.allclose(stride4[1::4], stride4[::4])
    assert torch.allclose(stride4[2::4], stride4[::4])
    assert torch.allclose(stride4[3::4], stride4[::4])
    assert int((y == 0).sum().item()) == 1
    assert int((y == 1).sum().item()) == 1
    assert int((y == 2).sum().item()) == 1


def test_make_temporal_jitter_views_shape_preserving() -> None:
    videos = torch.randn(2, 3, 16, 8, 8)
    rng = torch.Generator().manual_seed(0)
    v1, v2 = ta._make_temporal_jitter_views(videos, rng=rng, subclip_ratio=0.5)
    assert v1.shape == videos.shape
    assert v2.shape == videos.shape


def test_retrieval_top1_and_sims_perfect_match() -> None:
    emb = torch.randn(32, 16)
    sims = ta._retrieval_top1_and_sims(emb, emb.clone())
    assert sims["top1"] == pytest.approx(1.0, abs=1e-6)
    assert sims["pos_cos_mean"] > sims["neg_cos_mean"]


class _ConstantVideoDataset(Dataset):
    def __init__(self, n: int, *, c: int = 3, t: int = 16, h: int = 8, w: int = 8) -> None:
        self.n = n
        self.c = c
        self.t = t
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.full((self.c, self.t, self.h, self.w), float(idx), dtype=torch.float32)


class _MeanStudent(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) -> make unique CLS embedding per video via global mean
        m = x.mean(dim=(1, 2, 3, 4), keepdim=True)  # (B, 1, 1, 1, 1)
        m = m.view(x.shape[0], 1)  # (B, 1)
        cls = torch.cat([m, m * 2, m * 3, torch.ones_like(m)], dim=1)  # (B, 4)
        return cls.unsqueeze(1)  # (B, 1, 4)


def test_collect_clip_consistency_toy_cpu() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    student = _MeanStudent()
    dataset = _ConstantVideoDataset(20)
    stats = ta._collect_clip_consistency(
        student=student,  # type: ignore[arg-type]
        dataset=dataset,  # type: ignore[arg-type]
        max_videos=10,
        batch_size=5,
        num_workers=0,
        device=device,
        dtype=dtype,
        seed=0,
        split_name="toy",
        subclip_ratio=0.75,
    )
    assert stats["videos_used"] == 10.0
    assert stats["top1"] == pytest.approx(1.0, abs=1e-6)
