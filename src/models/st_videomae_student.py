from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.attention_factorized import FactorizedAttentionBlock, JointAttentionBlock
from src.utils.shapes import num_tokens


@dataclass(frozen=True)
class StudentConfig:
    num_frames: int = 16
    tubelet_size: int = 2
    patch_size: int = 16
    img_size: int = 224
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    factorized_blocks: int = 8
    mlp_ratio: float = 4.0
    drop: float = 0.0


class PatchEmbed3D(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            3,
            config.embed_dim,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> [B, D, T', H', W']
        x = self.proj(x)
        b, d, t, h, w = x.shape
        x = x.reshape(b, d, t * h * w).transpose(1, 2)  # [B, N, D]
        return x


class STVideoMAEStudent(nn.Module):
    """Hybrid ST-Transformer: factorized early blocks + joint tail."""

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = PatchEmbed3D(config)

        self.num_tokens = num_tokens(
            num_frames=config.num_frames,
            tubelet_size=config.tubelet_size,
            img_size=config.img_size,
            patch_size=config.patch_size,
        )
        self.grid_t = config.num_frames // config.tubelet_size
        self.grid_s = (config.img_size // config.patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        blocks = []
        for i in range(config.depth):
            if i < config.factorized_blocks:
                blocks.append(
                    FactorizedAttentionBlock(
                        config.embed_dim,
                        config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        drop=config.drop,
                    )
                )
            else:
                blocks.append(
                    JointAttentionBlock(
                        config.embed_dim,
                        config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        drop=config.drop,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError("video must be [B, C, T, H, W]")
        x = self.embed(video)
        if x.shape[1] != self.num_tokens:
            raise ValueError(f"Token count mismatch: {x.shape[1]} vs {self.num_tokens}")
        x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            if isinstance(blk, FactorizedAttentionBlock):
                x = blk(x, self.grid_t, self.grid_s)
            else:
                x = blk(x)

        x = self.norm(x)
        return x
