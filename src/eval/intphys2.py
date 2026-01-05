"""IntPhys2 physics understanding evaluation for V-JEPA encoders.

IntPhys2 is a benchmark for evaluating physical intuition in video models.
It tests whether models understand basic physical concepts like:
- Object permanence
- Gravity and support
- Collision dynamics
- Continuity

This module provides evaluation infrastructure for testing SALT-VLA
trained encoders on the IntPhys2 benchmark.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class IntPhys2Evaluator:
    """IntPhys2 physics understanding evaluation.

    Evaluates video encoder representations on physical reasoning tasks
    using a linear probe protocol.
    """

    def __init__(
        self,
        model: nn.Module,
        data_root: str | Path = "/mnt/ssv2/intphys2",
        batch_size: int = 32,
        num_workers: int = 4,
        device: torch.device | None = None,
        pooling: str = "mean",  # "mean", "cls", or "spatial_temporal"
    ):
        """Initialize IntPhys2 evaluator.

        Args:
            model: Video encoder model (StudentVideoViT or SALTModel)
            data_root: Path to IntPhys2 dataset
            batch_size: Batch size for feature extraction
            num_workers: DataLoader workers
            device: Device to run on (defaults to CUDA if available)
            pooling: Feature pooling strategy
        """
        self.model = model
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_features(
        self, video: torch.Tensor, return_all_tokens: bool = False
    ) -> torch.Tensor:
        """Extract encoder features from video.

        Args:
            video: Input video tensor (B, C, T, H, W)
            return_all_tokens: If True, return all tokens; else pool to single vector

        Returns:
            Features: (B, D) if pooled, (B, N, D) if return_all_tokens
        """
        with torch.no_grad():
            video = video.to(self.device, dtype=torch.bfloat16)

            # Get encoder output
            if hasattr(self.model, 'student'):
                # SALTModel: use student encoder
                patch_embeds = self.model.student.vit.patch_embed(video)
                # Use all patches (no masking for eval)
                all_indices = torch.arange(
                    patch_embeds.shape[1], device=self.device
                ).unsqueeze(0).expand(video.shape[0], -1)
                features = self.model.student.forward_visible_only(patch_embeds, all_indices)
            else:
                # Direct StudentVideoViT
                features = self.model(video)
                if features.shape[1] > self.model.num_patches:
                    # Remove CLS token if present
                    features = features[:, 1:]

            if return_all_tokens:
                return features

            # Pool features
            if self.pooling == "mean":
                return features.mean(dim=1)  # (B, D)
            elif self.pooling == "cls":
                return features[:, 0]  # (B, D) - first token
            elif self.pooling == "spatial_temporal":
                # Reshape to (B, T, H*W, D) and pool separately
                B, N, D = features.shape
                # Assume 8 temporal x 196 spatial
                T = 8
                HW = N // T
                features = features.reshape(B, T, HW, D)
                spatial_pool = features.mean(dim=2)  # (B, T, D)
                temporal_pool = spatial_pool.mean(dim=1)  # (B, D)
                return temporal_pool
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

    def extract_dataset_features(
        self, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features for entire dataset.

        Args:
            dataloader: DataLoader yielding (video, label) pairs

        Returns:
            features: (N, D) numpy array
            labels: (N,) numpy array
        """
        all_features = []
        all_labels = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                videos, labels = batch
            else:
                videos = batch
                labels = None

            features = self.extract_features(videos)
            all_features.append(features.cpu().float().numpy())

            if labels is not None:
                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0) if all_labels else None

        return features, labels

    def linear_probe(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        C: float = 1.0,
    ) -> dict:
        """Train linear classifier on frozen features.

        Args:
            train_features: (N_train, D) training features
            train_labels: (N_train,) training labels
            test_features: (N_test, D) test features
            test_labels: (N_test,) test labels
            C: Regularization strength

        Returns:
            Dictionary with accuracy, f1, and predictions
        """
        # Fit logistic regression
        clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', n_jobs=-1)
        clf.fit(train_features, train_labels)

        # Predict
        train_preds = clf.predict(train_features)
        test_preds = clf.predict(test_features)

        return {
            "train_accuracy": accuracy_score(train_labels, train_preds),
            "test_accuracy": accuracy_score(test_labels, test_preds),
            "train_f1": f1_score(train_labels, train_preds, average='weighted'),
            "test_f1": f1_score(test_labels, test_preds, average='weighted'),
            "test_predictions": test_preds,
        }

    def evaluate(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        task_name: str = "intphys2",
    ) -> dict:
        """Run full evaluation pipeline.

        Args:
            train_loader: Training set DataLoader
            test_loader: Test set DataLoader
            task_name: Name for logging

        Returns:
            Evaluation results dictionary
        """
        print(f"[{task_name}] Extracting training features...")
        train_features, train_labels = self.extract_dataset_features(train_loader)
        print(f"[{task_name}] Training features: {train_features.shape}")

        print(f"[{task_name}] Extracting test features...")
        test_features, test_labels = self.extract_dataset_features(test_loader)
        print(f"[{task_name}] Test features: {test_features.shape}")

        print(f"[{task_name}] Training linear probe...")
        results = self.linear_probe(
            train_features, train_labels, test_features, test_labels
        )

        print(f"[{task_name}] Results:")
        print(f"  Train accuracy: {results['train_accuracy']:.4f}")
        print(f"  Test accuracy:  {results['test_accuracy']:.4f}")
        print(f"  Test F1:        {results['test_f1']:.4f}")

        return {
            "task": task_name,
            **results,
            "n_train": len(train_labels),
            "n_test": len(test_labels),
            "feature_dim": train_features.shape[1],
        }


def evaluate_intphys2(
    model: nn.Module,
    data_root: str | Path = "/mnt/ssv2/intphys2",
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Convenience function for IntPhys2 evaluation.

    Args:
        model: Video encoder model
        data_root: Path to IntPhys2 dataset
        batch_size: Batch size
        num_workers: DataLoader workers

    Returns:
        Evaluation results
    """
    from src.data.intphys2_loader import get_intphys2_dataloaders

    train_loader, test_loader = get_intphys2_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    evaluator = IntPhys2Evaluator(
        model=model,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return evaluator.evaluate(train_loader, test_loader)
