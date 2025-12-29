from __future__ import annotations

import os
import multiprocessing as mp
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from src.data.loader import SSv2Dataset, collate_drop_none


def _resolve_data_root() -> Path:
    return Path(os.environ.get("SSV2_ROOT", "/mnt/ssv2"))


def _has_dataset(root: Path) -> bool:
    split_path = root / "train.json"
    videos_dir = root / "videos"
    if split_path.exists() and videos_dir.exists():
        return True
    if split_path.exists():
        sample_video = root / "1.webm"
        return sample_video.exists()
    return False


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_dataloader_preflight() -> None:
    data_root = _resolve_data_root()
    if not _has_dataset(data_root):
        pytest.skip(f"SSv2 dataset not found at {data_root}")

    num_workers = int(os.environ.get("PREFLIGHT_WORKERS", "8"))
    prefetch_factor = int(os.environ.get("PREFLIGHT_PREFETCH", "2"))
    batch_size = int(os.environ.get("PREFLIGHT_BATCH", "8"))
    max_batches = int(os.environ.get("PREFLIGHT_BATCHES", "20"))
    default_context = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    mp_context = os.environ.get("PREFLIGHT_MP_CONTEXT", default_context)

    dataset = SSv2Dataset(data_root, split="train")
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["multiprocessing_context"] = mp_context

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_drop_none,
        drop_last=True,
        **loader_kwargs,
    )

    ok_batches = 0
    loader_iter = iter(loader)
    try:
        for _ in range(max_batches):
            try:
                batch = next(loader_iter)
            except RuntimeError as exc:
                raise AssertionError(f"DataLoader RuntimeError: {exc}") from exc
            if batch.numel() > 0:
                ok_batches += 1
    finally:
        if hasattr(loader_iter, "_shutdown_workers"):
            loader_iter._shutdown_workers()

    assert ok_batches > 0, "No valid batches produced; check dataset or decode errors."
