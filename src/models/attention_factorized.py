from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FactorizedAttentionBlock(nn.Module):
    """Temporal then spatial attention over a dense (T x S) token grid.

    Uses standard MultiheadAttention modules to preserve parameter shapes.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=drop, batch_first=True
        )
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor, t: int, s: int) -> torch.Tensor:
        # x: [B, N, D], N = t * s
        b, n, d = x.shape
        if n != t * s:
            raise ValueError(f"Token count mismatch: {n} vs t*s={t*s}")

        # Temporal attention: (B, S, T, D) -> (B*S, T, D)
        x_temp = x.view(b, t, s, d).permute(0, 2, 1, 3).reshape(b * s, t, d)
        x_temp = self.norm1(x_temp)
        x_temp, _ = self.temporal_attn(x_temp, x_temp, x_temp, need_weights=False)
        x_temp = x_temp.view(b, s, t, d).permute(0, 2, 1, 3).reshape(b, n, d)
        x = x + x_temp

        # Spatial attention: (B, T, S, D) -> (B*T, S, D)
        x_spat = x.view(b, t, s, d).reshape(b * t, s, d)
        x_spat = self.norm2(x_spat)
        x_spat, _ = self.spatial_attn(x_spat, x_spat, x_spat, need_weights=False)
        x_spat = x_spat.view(b, t, s, d).reshape(b, n, d)
        x = x + x_spat

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class JointAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x
