"""Shared test fixtures and configuration for SALT-VLA tests."""
from __future__ import annotations

import os
import tempfile

import pytest
import torch
import math

# Ensure a writable temp directory for imports that allocate temp files.
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TEMP", "/tmp")
os.environ.setdefault("TMP", "/tmp")
tempfile.tempdir = "/tmp"


# ============================================================================
# Device Detection
# ============================================================================

def get_test_device() -> torch.device:
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_test_dtype(device: torch.device) -> torch.dtype:
    """Get appropriate dtype for the device."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


@pytest.fixture
def device() -> torch.device:
    """Pytest fixture for test device."""
    return get_test_device()


@pytest.fixture
def dtype(device: torch.device) -> torch.dtype:
    """Pytest fixture for test dtype."""
    return get_test_dtype(device)


# ============================================================================
# Model Configuration Fixtures
# ============================================================================

@pytest.fixture
def model_config() -> dict:
    """Standard model configuration for testing."""
    return {
        "img_size": 224,
        "patch_size": 16,
        "tubelet_size": 2,
        "num_frames": 16,
        "embed_dim": 768,
        "in_chans": 3,
    }


@pytest.fixture
def small_model_config() -> dict:
    """Smaller model config for faster tests."""
    return {
        "img_size": 112,
        "patch_size": 16,
        "tubelet_size": 2,
        "num_frames": 8,
        "embed_dim": 384,
        "in_chans": 3,
    }


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================

@pytest.fixture
def synthetic_video(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a synthetic video tensor for testing.

    Returns:
        Tensor of shape (B=2, C=3, T=16, H=224, W=224)
    """
    return torch.randn(2, 3, 16, 224, 224, device=device, dtype=dtype)


@pytest.fixture
def synthetic_video_small(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a smaller synthetic video for faster tests.

    Returns:
        Tensor of shape (B=1, C=3, T=8, H=112, W=112)
    """
    return torch.randn(1, 3, 8, 112, 112, device=device, dtype=dtype)


@pytest.fixture
def synthetic_batch(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a batch matching expected training input shape.

    Returns:
        Tensor of shape (B=4, C=3, T=16, H=224, W=224)
    """
    return torch.randn(4, 3, 16, 224, 224, device=device, dtype=dtype)


# ============================================================================
# Mock Model Components
# ============================================================================

@pytest.fixture
def mock_student_output(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mock student model output.

    Returns:
        Tensor of shape (B=2, N=1569, D=768) where N = 1 (CLS) + 1568 (patches)
    """
    # 1568 patches = 8 temporal x 14 x 14 spatial
    return torch.randn(2, 1569, 768, device=device, dtype=dtype)


@pytest.fixture
def mock_teacher_output(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mock teacher model output.

    Returns:
        Tensor of shape (B=2, N=1568, D=768) - no CLS token
    """
    return torch.randn(2, 1568, 768, device=device, dtype=dtype)


@pytest.fixture
def mock_mask_tokens(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mock mask tokens (75% masked).

    Returns:
        Tensor of shape (B=2, N=1569) with ~75% ones (masked)
    """
    batch_size = 2
    num_patches = 1568
    mask_ratio = 0.75
    mask_count = int(num_patches * mask_ratio)

    # Create masks
    mask = torch.zeros(batch_size, num_patches, device=device, dtype=dtype)
    for b in range(batch_size):
        indices = torch.randperm(num_patches)[:mask_count]
        mask[b, indices] = 1.0

    # Add CLS token (never masked)
    cls_mask = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    mask_tokens = torch.cat([cls_mask, mask], dim=1)

    return mask_tokens


# ============================================================================
# Loss Computation Helpers
# ============================================================================

def compute_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of masked MSE loss.

    Args:
        pred: Predictions of shape (B, N, D)
        target: Targets of shape (B, N, D)
        mask: Binary mask of shape (B, N) where 1 = masked (compute loss)

    Returns:
        Scalar loss value
    """
    mse = torch.nn.functional.mse_loss(pred, target, reduction="none")
    mask_expanded = mask.unsqueeze(-1).to(dtype=pred.dtype)
    masked_loss = mse * mask_expanded
    loss = masked_loss.sum() / (mask_expanded.sum() * pred.shape[-1] + 1e-6)
    return loss


# ============================================================================
# LR Schedule Helpers
# ============================================================================

def reference_lr_lambda(
    step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
) -> float:
    """Reference implementation of LR schedule for testing.

    Uses warmup + cosine decay with min_lr floor, matching train.py implementation.
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of peak (default 0.1 = 10%)
    """
    warmup_scale = min((step + 1) / max(1, warmup_steps), 1.0)
    progress = step / max(1, total_steps)
    # Cosine decay with minimum LR floor
    cosine_scale = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(progress * math.pi))
    return warmup_scale * cosine_scale


# ============================================================================
# Positional Embedding Helpers
# ============================================================================

def expected_num_patches(
    img_size: int, patch_size: int, tubelet_size: int, num_frames: int
) -> int:
    """Calculate expected number of patches."""
    grid_h = img_size // patch_size
    grid_w = img_size // patch_size
    grid_t = num_frames // tubelet_size
    return grid_t * grid_h * grid_w


def expected_grid_size(
    img_size: int, patch_size: int, tubelet_size: int, num_frames: int
) -> tuple[int, int, int]:
    """Calculate expected grid size (T, H, W)."""
    grid_t = num_frames // tubelet_size
    grid_h = img_size // patch_size
    grid_w = img_size // patch_size
    return (grid_t, grid_h, grid_w)
