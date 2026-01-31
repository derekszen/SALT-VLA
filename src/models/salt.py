from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from src.models.tubelet_space_time import TubeletSpaceTimeBlock


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        tubelet_size: int,
        in_chans: int,
        embed_dim: int,
        num_frames: int,
    ) -> None:
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError("num_frames must be divisible by tubelet_size.")
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.grid_size = (
            num_frames // tubelet_size,
            img_size // patch_size,
            img_size // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected video input of shape (B, C, T, H, W).")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class StudentVideoViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        tubelet_size: int,
        patch_size: int,
        num_frames: int,
        img_size: int,
        space_time_blocks: int = 0,
    ) -> None:
        super().__init__()
        import timm

        vit = timm.create_model(model_name, pretrained=False)
        self.embed_dim = vit.embed_dim

        patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=3,
            embed_dim=vit.embed_dim,
            num_frames=num_frames,
        )
        vit.patch_embed = patch_embed
        vit.num_patches = patch_embed.num_patches

        vit.pos_embed = nn.Parameter(
            self._inflate_pos_embed(vit.pos_embed, patch_embed.grid_size)
        )

        self.vit = vit
        self.grid_size = patch_embed.grid_size
        self.num_patches = patch_embed.num_patches
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.space_time_blocks = int(space_time_blocks)

        # Dense masking uses a learned token embedding in the student token space.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Optional: replace early-block attention with divided (time then space) attention.
        if self.space_time_blocks > 0:
            n_blocks = len(self.vit.blocks)
            n_st = min(self.space_time_blocks, n_blocks)
            for i in range(n_st):
                self.vit.blocks[i].attn = TubeletSpaceTimeBlock(
                    self.vit.blocks[i].attn,
                    num_frames=num_frames,
                    tubelet_size=tubelet_size,
                )

    def _inflate_pos_embed(
        self, pos_embed: torch.Tensor, grid_size: tuple[int, int, int]
    ) -> torch.Tensor:
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        num_2d = patch_pos.shape[1]
        size_2d = int(math.sqrt(num_2d))
        if size_2d * size_2d != num_2d:
            raise ValueError("Unexpected 2D pos_embed shape.")
        patch_pos = patch_pos.reshape(1, size_2d, size_2d, -1).permute(0, 3, 1, 2)
        if (size_2d, size_2d) != (grid_size[1], grid_size[2]):
            patch_pos = F.interpolate(
                patch_pos,
                size=(grid_size[1], grid_size[2]),
                mode="bilinear",
                align_corners=False,
            )
        patch_pos = patch_pos.permute(0, 2, 3, 1).unsqueeze(1)
        patch_pos = patch_pos.repeat(1, grid_size[0], 1, 1, 1)
        patch_pos = patch_pos.reshape(
            1, grid_size[0] * grid_size[1] * grid_size[2], -1
        )
        return torch.cat([cls_pos, patch_pos], dim=1)

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        if hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(enabled)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.vit.forward_features(video)

    def forward_visible_only(
        self, 
        patch_embeds: torch.Tensor, 
        visible_indices: torch.Tensor
    ) -> torch.Tensor:
        """Process only visible patches through the ViT blocks.
        
        Args:
            patch_embeds: All patch embeddings (B, N_total, D) from patch_embed layer
            visible_indices: Indices of visible patches (B, N_visible)
            
        Returns:
            Visible patch representations (B, N_visible, D)
        """
        if self.space_time_blocks > 0:
            raise ValueError(
                "forward_visible_only is not compatible with divided space-time attention, "
                "because token counts are no longer a regular (T x S) grid after masking. "
                "Use dense masking (mask-token replacement) instead."
            )
        B, N_visible = visible_indices.shape
        D = patch_embeds.shape[-1]
        
        # 1. Gather visible patches
        idx_expanded = visible_indices.unsqueeze(-1).expand(-1, -1, D)
        visible_embeds = torch.gather(patch_embeds, dim=1, index=idx_expanded)
        
        # 2. Add positional embeddings for visible positions only
        # pos_embed includes CLS token at position 0
        pos_embed_patches = self.vit.pos_embed[:, 1:]  # (1, N_total, D)
        pos_expanded = pos_embed_patches.expand(B, -1, -1)
        visible_pos = torch.gather(pos_expanded, dim=1, index=idx_expanded)
        visible_embeds = visible_embeds + visible_pos
        
        # 3. Process through ViT blocks (no CLS token for predictor input)
        x = self.vit.patch_drop(visible_embeds)
        x = self.vit.norm_pre(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        
        return x  # (B, N_visible, D)

    def forward_dense(
        self,
        patch_embeds: torch.Tensor,
        masked_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a full (dense) tubelet token grid through the ViT blocks.

        This is the day-1 path for divided space-time attention: keep tokens dense,
        replace masked positions with a learned mask token, and compute loss on masked
        positions only.

        Args:
            patch_embeds: All patch embeddings (B, N_total, D) from patch_embed layer.
            masked_indices: Optional indices of masked patches (B, N_masked). If provided,
                those positions are replaced with self.mask_token before the transformer.

        Returns:
            Patch representations (B, N_total, D) (no CLS token).
        """
        B, N_total, D = patch_embeds.shape

        # 1. Add positional embeddings for all patch positions (no CLS token).
        pos_embed_patches = self.vit.pos_embed[:, 1:]  # (1, N_total, D)
        x = patch_embeds + pos_embed_patches[:, :N_total].to(dtype=patch_embeds.dtype)

        # 2. Replace masked positions with a learned token embedding.
        if masked_indices is not None:
            idx = masked_indices.to(device=patch_embeds.device, dtype=torch.long)
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, D)
            mask_tok = self.mask_token.to(dtype=patch_embeds.dtype).expand(B, idx.shape[1], D)
            x = x.scatter(1, idx_expanded, mask_tok)

        # 3. ViT blocks (dense tokens; no CLS token)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        return x


class JEPAPredictor(nn.Module):
    """V-JEPA Predictor: reconstructs masked patch representations from visible ones.
    
    The predictor takes visible tokens from the student encoder, adds learnable
    mask tokens at masked positions, processes through transformer layers, and
    outputs predictions for the masked positions only.
    """
    
    def __init__(
        self,
        student_dim: int = 192,       # Input dim from student
        teacher_dim: int = 768,       # Output dim to match teacher
        predictor_dim: int = 384,     # Internal predictor dim
        depth: int = 6,               # Number of transformer layers
        num_heads: int = 6,           # Attention heads
        num_patches: int = 1568,      # Total patches (8 temporal x 14 x 14 spatial)
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.predictor_dim = predictor_dim
        self.num_patches = num_patches
        
        # Input projection: student_dim -> predictor_dim
        self.input_proj = nn.Linear(student_dim, predictor_dim, dtype=dtype)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim, dtype=dtype))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Positional embeddings for all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim, dtype=dtype))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dtype=dtype,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection: predictor_dim -> teacher_dim
        self.output_proj = nn.Linear(predictor_dim, teacher_dim, dtype=dtype)
        
        # Layer norm before output
        self.norm = nn.LayerNorm(predictor_dim, dtype=dtype)
    
    def forward(
        self,
        visible_tokens: torch.Tensor,
        visible_indices: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict masked token representations.
        
        Args:
            visible_tokens: Output from student encoder (B, N_visible, D_student)
            visible_indices: Indices of visible patches (B, N_visible)
            masked_indices: Indices of masked patches (B, N_masked)
            
        Returns:
            Predicted representations for masked positions (B, N_masked, D_teacher)
        """
        B = visible_tokens.shape[0]
        N_visible = visible_indices.shape[1]
        N_masked = masked_indices.shape[1]
        N_total = N_visible + N_masked
        
        # 1. Project visible tokens to predictor dimension
        visible_proj = self.input_proj(visible_tokens)  # (B, N_visible, predictor_dim)
        
        # 2. Create mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, N_masked, -1)  # (B, N_masked, predictor_dim)
        
        # 3. Combine and sort by position to create full sequence
        # We need to interleave visible and mask tokens in correct positions
        # Use visible_proj.dtype to ensure consistency with Linear layer output
        all_tokens = torch.zeros(
            B, N_total, self.predictor_dim, 
            device=visible_tokens.device, 
            dtype=visible_proj.dtype  # Match dtype with projected tokens
        )
        
        # Scatter visible tokens to their positions
        idx_visible = visible_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        all_tokens.scatter_(1, idx_visible, visible_proj)
        
        # Scatter mask tokens to their positions (cast to match all_tokens dtype)
        idx_masked = masked_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        all_tokens.scatter_(1, idx_masked, mask_tokens.to(all_tokens.dtype))
        
        # 4. Add positional embeddings
        all_tokens = all_tokens + self.pos_embed[:, :N_total]
        
        # 5. Process through transformer
        all_tokens = self.transformer(all_tokens)
        
        # 6. Extract and project masked positions only
        all_tokens = self.norm(all_tokens)
        masked_out = torch.gather(all_tokens, dim=1, index=idx_masked)
        
        # 7. Project to teacher dimension
        predictions = self.output_proj(masked_out)  # (B, N_masked, teacher_dim)
        
        return predictions


class SALTModel(nn.Module):
    """SALT model with V-JEPA architecture.
    
    Correct V-JEPA flow:
    1. Patch embed video (no masking yet)
    2. Generate mask (random or multi-block), split into visible/masked indices
    3. Student processes ONLY visible patches
    4. Predictor takes visible tokens + learnable mask tokens -> predicts masked
    5. Loss on masked positions only
    """
    
    def __init__(
        self,
        teacher_name: str = "Tianjiao-Yu/videomae-huge",
        load_teacher: bool = True,
        teacher_dim: int | None = None,
        student_model_name: str = "vit_base_patch16_224",
        tubelet_size: int = 2,
        patch_size: int = 16,
        num_frames: int = 16,
        img_size: int = 224,
        mask_ratio: float = 0.75,
        masking_strategy: str = "multiblock",  # NEW: "random" or "multiblock"
        # Student attention mode
        student_space_time_blocks: int = 0,  # 0 = joint attention everywhere (default)
        # Prediction mode
        use_predictor: bool = True,  # If False: dense mask-token path + linear head (no predictor)
        # Predictor config
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        predictor_num_heads: int = 6,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.teacher_name = teacher_name

        # 1. Teacher setup (frozen)
        if load_teacher:
            from transformers import AutoModel

            self.teacher: nn.Module | None = AutoModel.from_pretrained(
                teacher_name,
                trust_remote_code=True,
                dtype=dtype,
            )
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

            self.teacher_dim = self._get_teacher_dim(self.teacher.config)
        else:
            self.teacher = None
            if teacher_dim is None:
                raise ValueError("teacher_dim is required when load_teacher=False.")
            self.teacher_dim = int(teacher_dim)

        # 2. Student setup
        self.student = StudentVideoViT(
            model_name=student_model_name,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            num_frames=num_frames,
            img_size=img_size,
            space_time_blocks=student_space_time_blocks,
        ).to(dtype=dtype)

        num_patches = self.student.num_patches  # 1568 for default config

        if student_space_time_blocks > 0 and use_predictor:
            raise ValueError(
                "student_space_time_blocks > 0 requires dense masking (use_predictor=False). "
                "Divided space-time attention needs a dense (T x S) token grid."
            )

        self.use_predictor = bool(use_predictor)
        if self.use_predictor:
            # 3. Predictor (V-JEPA style): visible-only student + transformer predictor
            self.predictor: JEPAPredictor | None = JEPAPredictor(
                student_dim=self.student.embed_dim,
                teacher_dim=self.teacher_dim,
                predictor_dim=predictor_dim,
                depth=predictor_depth,
                num_heads=predictor_num_heads,
                num_patches=num_patches,
                dtype=dtype,
            )
            self.student_to_teacher: nn.Linear | None = None
        else:
            # 3. Minimal day-1 path: dense student tokens + linear projection to teacher dim.
            self.predictor = None
            self.student_to_teacher = nn.Linear(
                self.student.embed_dim, self.teacher_dim, bias=True, dtype=dtype
            )

        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy  # NEW
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.img_size = img_size
        self.dtype = dtype

        # Grid size for multi-block masking
        self.grid_t = num_frames // tubelet_size  # 8
        self.grid_h = img_size // patch_size       # 14
        self.grid_w = img_size // patch_size       # 14

    def forward(
        self, 
        video: torch.Tensor,
        cached_teacher_latents: torch.Tensor | None = None,
        visible_idx: torch.Tensor | None = None,
        masked_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V-JEPA forward pass.
        
        Args:
            video: Input video (B, C, T, H, W)
            cached_teacher_latents: Optional pre-computed teacher outputs (B, N, D_t)
            
        Returns:
            pred_masked: Predictions for masked positions (B, N_masked, D_teacher)
            teacher_masked: Teacher targets for masked positions (B, N_masked, D_teacher)
        """
        B = video.shape[0]
        device = video.device
        
        if video.shape[1:] != (3, self.num_frames, self.img_size, self.img_size):
            raise ValueError(
                f"Expected (B, 3, {self.num_frames}, {self.img_size}, {self.img_size}) input."
            )
        
        # 1. Patch embed (no masking at pixel level)
        patch_embeds = self.student.vit.patch_embed(video.to(dtype=self.dtype))  # (B, N, D_s)
        
        # 2. Generate or use provided mask indices
        if visible_idx is None and masked_idx is None:
            visible_idx, masked_idx = self._create_mask_indices(B, device)
        elif visible_idx is None or masked_idx is None:
            raise ValueError("Both visible_idx and masked_idx must be provided together.")
        else:
            visible_idx = visible_idx.to(device=device, dtype=torch.long)
            masked_idx = masked_idx.to(device=device, dtype=torch.long)
        
        # 3. Student processes ONLY visible patches
        if self.predictor is None:
            if self.student_to_teacher is None:
                raise RuntimeError("student_to_teacher is not initialized.")
            # Dense token grid: replace masked positions with mask token; loss only on masked positions.
            student_all = self.student.forward_dense(patch_embeds, masked_indices=masked_idx)
            student_masked = self._gather_patches(student_all, masked_idx)
            pred_masked = self.student_to_teacher(student_masked)
        else:
            student_out = self.student.forward_visible_only(patch_embeds, visible_idx)  # (B, N_visible, D_s)

            # 4. Predictor reconstructs masked positions
            pred_masked = self.predictor(
                visible_tokens=student_out,
                visible_indices=visible_idx,
                masked_indices=masked_idx,
            )  # (B, N_masked, D_teacher)

        # 5. Get teacher targets for masked positions
        if cached_teacher_latents is not None:
            # Use cached latents - gather masked positions
            teacher_masked = self._gather_patches(cached_teacher_latents, masked_idx)
        else:
            if self.teacher is None:
                raise ValueError(
                    "Teacher model is not loaded; provide cached_teacher_latents "
                    "or construct SALTModel with load_teacher=True."
                )
            # Compute teacher on-the-fly
            with torch.no_grad():
                teacher_all = self._teacher_tokens(video)
                teacher_masked = self._gather_patches(teacher_all, masked_idx)
        
        return pred_masked, teacher_masked
    
    def _create_mask_indices(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate visible and masked patch indices.
        
        Supports two strategies:
        - "random": Uniform random masking (original)
        - "multiblock": Spatiotemporally contiguous blocks (paper-aligned)
        
        Returns:
            visible_idx: Indices of visible patches (B, N_visible) - sorted
            masked_idx: Indices of masked patches (B, N_masked) - sorted
        """
        num_patches = self.student.num_patches
        num_masked = int(num_patches * self.mask_ratio)
        num_visible = num_patches - num_masked
        
        if self.masking_strategy == "random":
            return self._random_masking(batch_size, device, num_patches, num_visible)
        elif self.masking_strategy == "multiblock":
            return self._multiblock_masking(batch_size, device, num_patches, num_visible)
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")
    
    def _random_masking(
        self, batch_size: int, device: torch.device, num_patches: int, num_visible: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Original random masking strategy."""
        # Generate random permutation for each sample
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = noise.argsort(dim=1)
        
        # Split into visible and masked
        visible_idx = ids_shuffle[:, :num_visible]
        masked_idx = ids_shuffle[:, num_visible:]
        
        # Sort indices for consistent positional embedding lookup
        visible_idx = visible_idx.sort(dim=1).values
        masked_idx = masked_idx.sort(dim=1).values
        
        return visible_idx, masked_idx
    
    def _multiblock_masking(
        self, batch_size: int, device: torch.device, num_patches: int, num_visible: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-block masking: spatiotemporally contiguous blocks (V-JEPA paper).
        
        Samples 4-8 blocks per video, each covering a contiguous 3D region.
        Block sizes: 0.15-0.7 spatial, 0.1-0.5 temporal.
        """
        grid_t, grid_h, grid_w = self.grid_t, self.grid_h, self.grid_w
        num_masked = num_patches - num_visible
        
        all_visible_idx = []
        all_masked_idx = []
        
        for b in range(batch_size):
            # Create 3D mask grid (T, H, W) - 0=visible, 1=masked
            mask_grid = torch.zeros(grid_t, grid_h, grid_w, device=device)
            
            # Sample 4-8 blocks
            num_blocks = torch.randint(4, 9, (1,)).item()
            
            for _ in range(num_blocks):
                # Block dimensions (as fraction of grid)
                t_scale = 0.1 + 0.4 * torch.rand(1).item()  # 0.1-0.5 of temporal
                h_scale = 0.15 + 0.55 * torch.rand(1).item()  # 0.15-0.7 of spatial
                w_scale = 0.15 + 0.55 * torch.rand(1).item()  # 0.15-0.7 of spatial
                
                # Block size in grid units
                block_t = max(1, int(grid_t * t_scale))
                block_h = max(1, int(grid_h * h_scale))
                block_w = max(1, int(grid_w * w_scale))
                
                # Random starting position
                start_t = torch.randint(0, max(1, grid_t - block_t + 1), (1,)).item()
                start_h = torch.randint(0, max(1, grid_h - block_h + 1), (1,)).item()
                start_w = torch.randint(0, max(1, grid_w - block_w + 1), (1,)).item()
                
                # Mark block as masked
                mask_grid[start_t:start_t+block_t, start_h:start_h+block_h, start_w:start_w+block_w] = 1
            
            # Flatten and get indices
            flat_mask = mask_grid.flatten()  # (num_patches,)
            masked_positions = (flat_mask == 1).nonzero(as_tuple=False).squeeze(-1)
            visible_positions = (flat_mask == 0).nonzero(as_tuple=False).squeeze(-1)
            
            # Adjust to match target mask ratio
            current_masked = len(masked_positions)
            if current_masked < num_masked:
                # Need more masked - randomly select from visible
                extra_needed = num_masked - current_masked
                perm = torch.randperm(len(visible_positions), device=device)[:extra_needed]
                extra_masked = visible_positions[perm]
                masked_positions = torch.cat([masked_positions, extra_masked])
                visible_mask = torch.ones(len(visible_positions), dtype=torch.bool, device=device)
                visible_mask[perm] = False
                visible_positions = visible_positions[visible_mask]
            elif current_masked > num_masked:
                # Too many masked - randomly restore some to visible
                keep_masked = num_masked
                perm = torch.randperm(len(masked_positions), device=device)
                masked_positions = masked_positions[perm[:keep_masked]]
                # Recalculate visible positions
                all_positions = torch.arange(num_patches, device=device)
                mask = torch.ones(num_patches, dtype=torch.bool, device=device)
                mask[masked_positions] = False
                visible_positions = all_positions[mask]
            
            # Sort for consistent positional embedding lookup
            visible_idx = visible_positions.sort().values
            masked_idx = masked_positions.sort().values
            
            all_visible_idx.append(visible_idx)
            all_masked_idx.append(masked_idx)
        
        return torch.stack(all_visible_idx), torch.stack(all_masked_idx)

    def _gather_patches(
        self, tokens: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Gather patches at specified indices.
        
        Args:
            tokens: (B, N, D)
            indices: (B, K)
            
        Returns:
            Gathered tokens (B, K, D)
        """
        D = tokens.shape[-1]
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(tokens, dim=1, index=idx_expanded)

    def train(self, mode: bool = True) -> "SALTModel":
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    @staticmethod
    def _get_teacher_dim(config) -> int:
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "embed_dim"):
            return int(config.embed_dim)
        if hasattr(config, "encoder_embed_dim"):
            return int(config.encoder_embed_dim)
        if hasattr(config, "model_config") and isinstance(config.model_config, dict):
            if "embed_dim" in config.model_config:
                return int(config.model_config["embed_dim"])
        cfg = config.to_dict()
        for key in ("hidden_size", "embed_dim", "encoder_embed_dim", "decoder_embed_dim"):
            if key in cfg:
                return int(cfg[key])
        if "model_config" in cfg and isinstance(cfg["model_config"], dict):
            if "embed_dim" in cfg["model_config"]:
                return int(cfg["model_config"]["embed_dim"])
        raise ValueError("Unable to infer teacher hidden dimension from config.")

    def _mask_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = video.shape[0]
        device = video.device
        grid_t, grid_h, grid_w = self.student.grid_size
        num_patches = self.student.num_patches
        mask_count = int(num_patches * self.mask_ratio)

        noise = torch.rand(b, num_patches, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.zeros(b, num_patches, device=device, dtype=video.dtype)
        mask.scatter_(1, ids[:, :mask_count], 1.0)

        patch_mask = mask.reshape(b, grid_t, grid_h, grid_w)
        pixel_mask = patch_mask.repeat_interleave(self.tubelet_size, dim=1)
        pixel_mask = pixel_mask.repeat_interleave(self.patch_size, dim=2)
        pixel_mask = pixel_mask.repeat_interleave(self.patch_size, dim=3)
        pixel_mask = pixel_mask.unsqueeze(1)

        masked_video = video.clone()
        masked_video = masked_video.masked_fill(pixel_mask.bool().expand(-1, 3, -1, -1, -1), 0.0)

        cls_mask = torch.zeros(b, 1, device=device, dtype=mask.dtype)
        mask_tokens = torch.cat([cls_mask, mask], dim=1)
        return masked_video, mask_tokens

    def tokens_to_patch_mask(self, mask_tokens: torch.Tensor) -> torch.Tensor:
        if mask_tokens.ndim != 2:
            raise ValueError("mask_tokens must be (B, N).")
        if mask_tokens.shape[1] == self.student.num_patches + 1:
            patch_tokens = mask_tokens[:, 1:]
        elif mask_tokens.shape[1] == self.student.num_patches:
            patch_tokens = mask_tokens
        else:
            raise ValueError("Unexpected mask token length.")
        grid_t, grid_h, grid_w = self.student.grid_size
        return patch_tokens.reshape(mask_tokens.shape[0], grid_t, grid_h, grid_w)

    def _teacher_tokens(self, video: torch.Tensor) -> torch.Tensor:
        if self.teacher is None:
            raise ValueError("Teacher model is not loaded.")
        teacher = self.teacher.model if hasattr(self.teacher, "model") else self.teacher
        if hasattr(teacher, "embeddings") and hasattr(teacher, "encoder"):
            # VideoMAE v1 models use embeddings/encoder and expect (B, T, C, H, W)
            pixel_values = video.permute(0, 2, 1, 3, 4)
            outputs = teacher(pixel_values=pixel_values)
            return outputs.last_hidden_state

        x = teacher.patch_embed(video)
        if getattr(teacher, "pos_embed", None) is not None:
            x = x + teacher.pos_embed.expand(x.shape[0], -1, -1).type_as(x).to(
                x.device
            )
        x = teacher.pos_drop(x)
        for blk in teacher.blocks:
            x = blk(x)
        x = teacher.norm(x)
        return x
