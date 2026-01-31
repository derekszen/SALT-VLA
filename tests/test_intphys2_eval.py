"""Tests for IntPhys2 evaluation pipeline.

Tests cover:
- Encoder output shape for evaluation
- Temporal pooling strategies
- Linear probe classifier interface
- Feature extraction in eval mode
- End-to-end evaluation pipeline
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from src.models.salt import StudentVideoViT

# Check if sklearn is available
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def device() -> torch.device:
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype(device: torch.device) -> torch.dtype:
    """Get test dtype."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


@pytest.fixture
def student_model(device: torch.device, dtype: torch.dtype) -> StudentVideoViT:
    """Create StudentVideoViT for testing."""
    model = StudentVideoViT(
        model_name="vit_tiny_patch16_224",
        tubelet_size=2,
        patch_size=16,
        num_frames=16,
        img_size=224,
    ).to(device=device, dtype=dtype)
    model.eval()
    return model


@pytest.fixture
def sample_video(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create sample video tensor."""
    return torch.randn(2, 3, 16, 224, 224, device=device, dtype=dtype)


# ============================================================================
# Encoder Output Shape Tests
# ============================================================================

class TestEncoderOutputShape:
    """Tests for encoder output shape in evaluation mode."""

    def test_encoder_output_shape(
        self, student_model: StudentVideoViT, sample_video: torch.Tensor
    ):
        """Verify encoder produces expected output shape."""
        with torch.no_grad():
            output = student_model(sample_video)

        # Expected: (B, N+1, D) where N=1568 patches + 1 CLS token
        assert output.ndim == 3
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 1569  # 1568 patches + 1 CLS
        assert output.shape[2] == 192  # ViT-Tiny embed_dim

    def test_encoder_patch_embed_shape(
        self, student_model: StudentVideoViT, sample_video: torch.Tensor
    ):
        """Verify patch embedding produces correct shape."""
        with torch.no_grad():
            patch_embeds = student_model.vit.patch_embed(sample_video)

        # Expected: (B, 1568, D) - no CLS token at patch_embed stage
        assert patch_embeds.shape == (2, 1568, 192)

    def test_forward_visible_only_full_patches(
        self, student_model: StudentVideoViT, sample_video: torch.Tensor, device: torch.device
    ):
        """Test forward_visible_only with all patches visible (eval mode)."""
        with torch.no_grad():
            patch_embeds = student_model.vit.patch_embed(sample_video)

            # All patches visible for evaluation
            B, N, D = patch_embeds.shape
            all_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

            output = student_model.forward_visible_only(patch_embeds, all_indices)

        assert output.shape == (2, 1568, 192)


# ============================================================================
# Temporal Pooling Tests
# ============================================================================

class TestTemporalPooling:
    """Tests for temporal pooling strategies."""

    def test_mean_pooling(self, device: torch.device, dtype: torch.dtype):
        """Test mean pooling over all tokens."""
        features = torch.randn(2, 1568, 192, device=device, dtype=dtype)

        pooled = features.mean(dim=1)

        assert pooled.shape == (2, 192)

    def test_spatial_temporal_pooling(self, device: torch.device, dtype: torch.dtype):
        """Test spatial-then-temporal pooling."""
        B, N, D = 2, 1568, 192
        T, HW = 8, 196  # 8 temporal x 196 spatial

        features = torch.randn(B, N, D, device=device, dtype=dtype)

        # Reshape to (B, T, H*W, D)
        features_reshaped = features.reshape(B, T, HW, D)

        # Spatial pool first
        spatial_pooled = features_reshaped.mean(dim=2)  # (B, T, D)
        assert spatial_pooled.shape == (B, T, D)

        # Then temporal pool
        temporal_pooled = spatial_pooled.mean(dim=1)  # (B, D)
        assert temporal_pooled.shape == (B, D)

    def test_cls_token_pooling(self, device: torch.device, dtype: torch.dtype):
        """Test CLS token extraction."""
        features = torch.randn(2, 1569, 192, device=device, dtype=dtype)

        cls_token = features[:, 0]

        assert cls_token.shape == (2, 192)


# ============================================================================
# Linear Probe Tests
# ============================================================================

class TestProbeClassifier:
    """Tests for linear probe classifier interface."""

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_probe_with_random_features(self):
        """Test linear probe with random features."""
        from sklearn.linear_model import LogisticRegression

        # Random features
        train_features = np.random.randn(100, 192).astype(np.float32)
        train_labels = np.random.randint(0, 2, 100)
        test_features = np.random.randn(20, 192).astype(np.float32)
        test_labels = np.random.randint(0, 2, 20)

        # Fit classifier
        clf = LogisticRegression(max_iter=100)
        clf.fit(train_features, train_labels)

        # Predict
        predictions = clf.predict(test_features)

        assert predictions.shape == (20,)
        assert set(predictions).issubset({0, 1})

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_probe_accuracy_calculation(self):
        """Test accuracy calculation."""
        from sklearn.metrics import accuracy_score

        labels = np.array([0, 1, 0, 1, 1])
        predictions = np.array([0, 1, 0, 0, 1])

        accuracy = accuracy_score(labels, predictions)

        assert accuracy == 0.8  # 4/5 correct


# ============================================================================
# Feature Extraction Mode Tests
# ============================================================================

class TestFeatureExtractionMode:
    """Tests for model in eval mode for feature extraction."""

    def test_model_eval_mode(self, student_model: StudentVideoViT):
        """Test that model is in eval mode."""
        student_model.eval()
        assert not student_model.training

    def test_no_grad_context(
        self, student_model: StudentVideoViT, sample_video: torch.Tensor
    ):
        """Test feature extraction under no_grad context."""
        student_model.eval()

        with torch.no_grad():
            output = student_model(sample_video)

        # Output should not require grad
        assert not output.requires_grad

    def test_deterministic_output(
        self, student_model: StudentVideoViT, sample_video: torch.Tensor
    ):
        """Test that eval mode produces deterministic output."""
        student_model.eval()

        with torch.no_grad():
            output1 = student_model(sample_video)
            output2 = student_model(sample_video)

        assert torch.allclose(output1, output2)


# ============================================================================
# Full Pipeline Tests
# ============================================================================

class TestFullPipeline:
    """End-to-end evaluation pipeline tests."""

    def test_feature_to_numpy_conversion(
        self, student_model: StudentVideoViT, sample_video: torch.Tensor
    ):
        """Test conversion from torch features to numpy."""
        with torch.no_grad():
            features = student_model(sample_video)
            pooled = features[:, 1:].mean(dim=1)  # Skip CLS, pool patches

        numpy_features = pooled.cpu().float().numpy()

        assert isinstance(numpy_features, np.ndarray)
        assert numpy_features.shape == (2, 192)
        assert numpy_features.dtype == np.float32

    def test_batch_feature_accumulation(
        self, student_model: StudentVideoViT, device: torch.device, dtype: torch.dtype
    ):
        """Test accumulating features across batches."""
        all_features = []
        all_labels = []

        # Simulate 3 batches
        for i in range(3):
            batch = torch.randn(4, 3, 16, 224, 224, device=device, dtype=dtype)
            labels = torch.randint(0, 2, (4,))

            with torch.no_grad():
                features = student_model(batch)
                pooled = features[:, 1:].mean(dim=1)

            all_features.append(pooled.cpu().float().numpy())
            all_labels.append(labels.numpy())

        # Concatenate
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        assert features.shape == (12, 192)  # 3 batches * 4 samples
        assert labels.shape == (12,)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_evaluator_interface(self, student_model: StudentVideoViT):
        """Test IntPhys2Evaluator can be instantiated."""
        from src.eval.intphys2 import IntPhys2Evaluator

        evaluator = IntPhys2Evaluator(
            model=student_model,
            data_root="/mnt/ssv2/intphys2",
            batch_size=4,
            pooling="mean",
        )

        assert evaluator.model is student_model
        assert evaluator.pooling == "mean"
