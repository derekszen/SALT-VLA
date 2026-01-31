from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.bench_time_arrow_ssv2 as ta  # noqa: E402


def test_parse_dtype_variants() -> None:
    assert ta._parse_dtype("bf16") is torch.bfloat16
    assert ta._parse_dtype("bfloat16") is torch.bfloat16
    assert ta._parse_dtype("torch.bfloat16") is torch.bfloat16
    assert ta._parse_dtype("fp16") is torch.float16
    assert ta._parse_dtype("float16") is torch.float16
    assert ta._parse_dtype("torch.float16") is torch.float16
    assert ta._parse_dtype("fp32") is torch.float32
    assert ta._parse_dtype("float32") is torch.float32
    assert ta._parse_dtype("torch.float32") is torch.float32

    with pytest.raises(ValueError):
        ta._parse_dtype("float64")


def test_extract_embedding_shapes() -> None:
    x3 = torch.randn(2, 5, 7)
    out3 = ta._extract_embedding(x3)
    assert out3.shape == (2, 7)

    x2 = torch.randn(2, 7)
    out2 = ta._extract_embedding(x2)
    assert out2.shape == (2, 7)

    with pytest.raises(ValueError):
        ta._extract_embedding(torch.randn(2, 3, 4, 5))


def test_normalize_features_l2() -> None:
    X_train = torch.randn(10, 8)
    X_val = torch.randn(7, 8)
    train_n, val_n, stats = ta._normalize_features(X_train, X_val, "l2")
    assert stats["norm"] == "l2"
    assert torch.allclose(train_n.norm(dim=1), torch.ones(10), atol=1e-5, rtol=1e-5)
    assert torch.allclose(val_n.norm(dim=1), torch.ones(7), atol=1e-5, rtol=1e-5)


def test_normalize_features_zscore() -> None:
    X_train = torch.randn(100, 16)
    X_val = torch.randn(50, 16)
    train_n, val_n, stats = ta._normalize_features(X_train, X_val, "zscore")
    assert stats["norm"] == "zscore"
    assert torch.allclose(train_n.mean(dim=0), torch.zeros(16), atol=1e-1)
    assert torch.allclose(train_n.std(dim=0), torch.ones(16), atol=2e-1)
    assert train_n.shape == X_train.shape
    assert val_n.shape == X_val.shape


def test_load_ckpt_config_prefers_embedded_config(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.pth"
    obj = {"config": {"student_model_name": "vit_large_patch16_224"}, "state_dict": {"student": {}}}
    torch.save(obj, ckpt)
    cfg = ta._load_ckpt_config(ckpt)
    assert cfg["student_model_name"] == "vit_large_patch16_224"


def test_load_ckpt_config_falls_back_to_sibling_json(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.pth"
    torch.save({"state_dict": {"student": {}}}, ckpt)
    (tmp_path / "config.json").write_text(
        json.dumps({"student_model_name": "vit_base_patch16_224"}), encoding="utf-8"
    )
    cfg = ta._load_ckpt_config(ckpt)
    assert cfg["student_model_name"] == "vit_base_patch16_224"


def test_load_student_state_dict_from_train_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.pth"
    student_sd = {"a.weight": torch.randn(2, 2)}
    torch.save({"state_dict": {"student": student_sd}, "config": {}}, ckpt)
    out = ta._load_student_state_dict(ckpt)
    assert out.keys() == student_sd.keys()


class _ToyVideoDataset(Dataset):
    def __init__(self, n: int, *, c: int = 3, t: int = 4, h: int = 8, w: int = 8) -> None:
        self.n = n
        self.c = c
        self.t = t
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Make time direction detectable: frames increase with t.
        frames = torch.arange(self.t, dtype=torch.float32).view(1, self.t, 1, 1)
        video = frames.expand(self.c, -1, self.h, self.w).clone()
        video += float(idx) * 0.01
        return video


class _ToyStudent(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        first = x[:, :, 0].mean(dim=(1, 2, 3))
        last = x[:, :, -1].mean(dim=(1, 2, 3))
        s = (first - last).unsqueeze(1)
        cls = torch.cat([s, torch.ones_like(s), torch.zeros_like(s), -s], dim=1)
        return cls.unsqueeze(1)  # (B, 1, D)


def test_collect_features_builds_balanced_labels_cpu() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    student = _ToyStudent()
    dataset = _ToyVideoDataset(10)

    X, y, stats = ta._collect_features(
        student=student,
        dataset=dataset,  # type: ignore[arg-type]
        max_videos=4,
        batch_size=2,
        num_workers=0,
        device=device,
        dtype=dtype,
        seed=0,
        split_name="toy",
    )
    assert X.shape == (8, 4)  # forward + reversed
    assert y.shape == (8,)
    assert int((y == 0).sum().item()) == 4
    assert int((y == 1).sum().item()) == 4
    assert stats["videos_used"] == 4.0
    assert stats["cosine_delta_mean"] > 0.0
