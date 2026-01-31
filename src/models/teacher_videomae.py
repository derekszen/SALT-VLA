from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import VideoMAEModel

from src.utils.shapes import num_tokens


@dataclass(frozen=True)
class TeacherConfig:
    model_name: str = "MCG-NJU/videomae-huge-finetuned-kinetics"
    num_frames: int = 16
    tubelet_size: int = 2
    patch_size: int = 16
    img_size: int = 224
    teacher_dim: int = 1280
    target_dim: int = 384
    projection_seed: int = 0
    projection_path: Path | None = None


def load_or_create_projection(
    *,
    path: Path,
    in_dim: int,
    out_dim: int,
    seed: int,
) -> np.ndarray:
    if path.exists():
        mat = np.load(path)
        if mat.shape != (in_dim, out_dim):
            raise ValueError(f"Projection matrix shape mismatch: {mat.shape}")
        return mat
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((in_dim, out_dim), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mat)
    return mat


class VideoMAETeacher(nn.Module):
    """Frozen VideoMAE-H teacher wrapper with fixed projection."""

    def __init__(self, config: TeacherConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VideoMAEModel.from_pretrained(config.model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        proj_path = (
            config.projection_path
            if config.projection_path is not None
            else Path("checkpoints") / "projection" / "videomae_huge_1280_to_384.npy"
        )
        mat = load_or_create_projection(
            path=proj_path,
            in_dim=config.teacher_dim,
            out_dim=config.target_dim,
            seed=config.projection_seed,
        )
        self.register_buffer("proj", torch.from_numpy(mat), persistent=False)

        self.to(self.device)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Return projected teacher tokens [B, N, target_dim]."""
        if video.ndim != 5:
            raise ValueError("video must have shape [B, C, T, H, W]")

        # VideoMAE expects [B, T, C, H, W]
        video = video.permute(0, 2, 1, 3, 4).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(pixel_values=video)
            tokens = outputs.last_hidden_state  # [B, N(+1), 1280]

        expected = num_tokens(
            num_frames=self.config.num_frames,
            tubelet_size=self.config.tubelet_size,
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
        )
        if tokens.shape[1] == expected + 1:
            tokens = tokens[:, 1:, :]
        if tokens.shape[1] != expected:
            raise ValueError(
                f"Unexpected token count: got {tokens.shape[1]} expected {expected}"
            )
        if tokens.shape[-1] != self.config.teacher_dim:
            raise ValueError(
                f"Unexpected teacher dim: got {tokens.shape[-1]} expected {self.config.teacher_dim}"
            )

        proj = self.proj.to(tokens.device, dtype=tokens.dtype)
        projected = torch.matmul(tokens, proj)  # [B, N, target_dim]
        return projected.to(torch.float16)
