from __future__ import annotations

import torch
from torch import nn


class TubeletSpaceTimeBlock(nn.Module):
    """Apply divided space-time attention over a tubelet token grid.

    This is a minimal wrapper intended to *reuse* an existing MHSA module that
    already operates on (B, N, C) token sequences (e.g., timm ViT Attention).

    It reshapes patch tokens into (T_tokens, S_tokens) to run:
      1) temporal attention per spatial location, then
      2) spatial attention per time index.

    Notes:
    - This wrapper assumes tubelet tokenization (patch_size=16, tubelet_size=2 by default),
      but does not implement patch embedding itself.
    - If a leading CLS token is present, it is passed through unchanged and only patch
      tokens are processed.
    """

    def __init__(
        self,
        attn: nn.Module,
        *,
        num_frames: int,
        tubelet_size: int = 2,
    ) -> None:
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError("num_frames must be divisible by tubelet_size.")
        self.attn = attn
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)

    def _call_attn(
        self,
        x: torch.Tensor,
        *,
        head_mask: torch.Tensor | None,
        output_attentions: bool,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Call the wrapped attention with best-effort argument compatibility.

        - timm ViT Attention: forward(x) -> Tensor
        - HF VideoMAEAttention: forward(x, head_mask=None, output_attentions=False) -> tuple
        """
        kwargs: dict[str, object] = {}
        if attn_mask is not None:
            kwargs["attn_mask"] = attn_mask
        if head_mask is not None or output_attentions:
            kwargs["head_mask"] = head_mask
            kwargs["output_attentions"] = output_attentions

        # Try the most informative call first, then progressively drop kwargs until compatible.
        try:
            return self.attn(x, **kwargs)
        except TypeError:
            # timm Attention doesn't accept HF-style args.
            kwargs.pop("head_mask", None)
            kwargs.pop("output_attentions", None)
            try:
                return self.attn(x, **kwargs)
            except TypeError:
                return self.attn(x)

    def forward(
        self,
        x: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if x.ndim != 3:
            raise ValueError("Expected token tensor of shape (B, N, C).")
        if attn_mask is not None:
            raise ValueError(
                "attn_mask is not supported by TubeletSpaceTimeBlock (the token grid is reshaped "
                "into separate temporal/spatial sequences). Pass attn_mask=None."
            )

        bsz, n_tokens, dim = x.shape
        t_tokens = self.num_frames // self.tubelet_size

        # Handle optional CLS token: many ViT implementations use (1 + N_patches).
        cls: torch.Tensor | None = None
        x_patches = x
        if n_tokens % t_tokens != 0:
            if (n_tokens - 1) % t_tokens == 0:
                cls = x[:, :1]
                x_patches = x[:, 1:]
                n_tokens = n_tokens - 1
            else:
                raise ValueError(
                    f"Token count N={x.shape[1]} is incompatible with t_tokens={t_tokens} "
                    "(neither N nor N-1 is divisible by t_tokens)."
                )

        if n_tokens % t_tokens != 0:
            raise ValueError(
                f"Patch token count N={n_tokens} must be divisible by t_tokens={t_tokens}."
            )
        s_tokens = n_tokens // t_tokens

        # 1) Temporal attention: treat each spatial position as an independent sequence.
        # x_patches: (B, N, C) -> (B, T, S, C) -> (B, S, T, C) -> (B*S, T, C)
        x_ts = x_patches.view(bsz, t_tokens, s_tokens, dim)
        x_time = x_ts.permute(0, 2, 1, 3).reshape(bsz * s_tokens, t_tokens, dim)
        time_out = self._call_attn(
            x_time,
            head_mask=head_mask,
            output_attentions=False,
            attn_mask=None,
        )
        x_time = time_out[0] if isinstance(time_out, tuple) else time_out
        # Back to (B, N, C)
        x_time = (
            x_time.view(bsz, s_tokens, t_tokens, dim)
            .permute(0, 2, 1, 3)
            .reshape(bsz, n_tokens, dim)
        )

        # 2) Spatial attention: treat each time index as an independent sequence.
        # (B, N, C) -> (B, T, S, C) -> (B*T, S, C)
        x_space = x_time.view(bsz, t_tokens, s_tokens, dim).reshape(bsz * t_tokens, s_tokens, dim)
        space_out = self._call_attn(
            x_space,
            head_mask=head_mask,
            output_attentions=output_attentions,
            attn_mask=None,
        )
        x_space = space_out[0] if isinstance(space_out, tuple) else space_out
        x_out = x_space.view(bsz, t_tokens, s_tokens, dim).reshape(bsz, n_tokens, dim)

        if cls is not None:
            x_out = torch.cat([cls, x_out], dim=1)
        if isinstance(space_out, tuple):
            # Preserve HF-style API: (hidden_states,) + (attentions,) if requested.
            return (x_out,) + space_out[1:]
        return x_out
