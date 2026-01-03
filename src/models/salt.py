from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


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

class SALTModel(nn.Module):
    def __init__(
        self,
        teacher_name: str = "OpenGVLab/VideoMAEv2-Base", # Optimized choice
        student_model_name: str = "vit_base_patch16_224",  # Configurable student size
        tubelet_size: int = 2,
        patch_size: int = 16,
        num_frames: int = 16,
        img_size: int = 224,
        mask_ratio: float = 0.75,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        
        # 1. Teacher setup
        # trust_remote_code=True is required for V2's custom attention kernels
        self.teacher = AutoModel.from_pretrained(
            teacher_name, 
            trust_remote_code=True,
            dtype=dtype
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 2. Student setup
        self.student = StudentVideoViT(
            model_name=student_model_name,  # Use configurable student size
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            num_frames=num_frames,
            img_size=img_size,
        ).to(dtype=dtype)

        # 3. Projection (Only needed if dims differ)
        teacher_dim = self._get_teacher_dim(self.teacher.config)
        if self.student.embed_dim != teacher_dim:
            self.proj = nn.Linear(self.student.embed_dim, teacher_dim, dtype=dtype)
        else:
            self.proj = nn.Identity()

        self.mask_ratio = mask_ratio
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.img_size = img_size

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # video: (B, C, T, H, W)
        if video.shape[1:] != (3, self.num_frames, self.img_size, self.img_size):
            raise ValueError(
                f"Expected (B, 3, {self.num_frames}, {self.img_size}, {self.img_size}) input."
            )
        
        # 1. Get Teacher Latents (Clean)
        # VideoMAEv2 expects (B, C, T, H, W) normalized to [0, 1]
        with torch.no_grad():
            teacher_latents = self._teacher_tokens(video)

        # 2. Masking & Student Forward
        # Use your existing pixel masking logic for simplicity, 
        # or move to token masking for speed later.
        masked_video, mask_tokens = self._mask_video(video)
        
        # Student sees masked pixels
        student_tokens = self.student(masked_video.to(dtype=torch.bfloat16))
        
        # Align tokens
        student_pred = self.proj(student_tokens)

        if teacher_latents.shape[1] == student_pred.shape[1] - 1:
            student_pred = student_pred[:, 1:]
            mask_tokens = mask_tokens[:, 1:]
        elif teacher_latents.shape[1] != student_pred.shape[1]:
            raise ValueError(
                "Token mismatch between teacher and student after alignment."
            )
        
        # Return student prediction, teacher truth, and the binary mask for loss calc
        return student_pred, teacher_latents, mask_tokens

    def train(self, mode: bool = True) -> "SALTModel":
        super().train(mode)
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
        teacher = self.teacher.model if hasattr(self.teacher, "model") else self.teacher
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
