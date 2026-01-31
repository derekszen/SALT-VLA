from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.attention_factorized import JointAttentionBlock


@dataclass(frozen=True)
class PredictorConfig:
    num_tokens: int = 1568
    dim: int = 384
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop: float = 0.0


class JEPAPredictor(nn.Module):
    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_tokens, config.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                JointAttentionBlock(
                    config.dim,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop=config.drop,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply predictor on full token grid with mask tokens.

        Args:
            tokens: [B, N, D] student tokens
            mask: [B, N] boolean mask where True = masked
        """
        b, n, d = tokens.shape
        if n != self.config.num_tokens:
            raise ValueError(f"Token count mismatch: {n} vs {self.config.num_tokens}")
        if mask.shape != (b, n):
            raise ValueError("mask must be [B, N]")

        x = tokens + self.pos_embed
        mask_token = self.mask_token + self.pos_embed
        mask_token = mask_token.expand(b, n, d)
        x = torch.where(mask.unsqueeze(-1), mask_token, x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
