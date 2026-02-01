from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.video_decode import decode_video, get_video_length, sample_frame_indices
from src.data.transforms import VideoTransformConfig, apply_video_transform


@dataclass(frozen=True)
class SSV2Config:
    data_root: Path
    split: str = "train"
    num_frames: int = 16
    sample_mode: str = "random"
    seed: int = 0
    transform: VideoTransformConfig = VideoTransformConfig()
    video_extensions: tuple[str, ...] = (".webm", ".mp4")
    video_dir: Path | None = None
    split_path: Path | None = None


class SSV2Dataset(Dataset):
    """Something-Something v2 dataset loader with deterministic sampling."""

    def __init__(self, config: SSV2Config) -> None:
        self.config = config
        self.data_root = Path(config.data_root)
        self.split_path = self._resolve_split_path(config.split_path)
        self.video_dir = self._resolve_video_dir(config.video_dir)

        self.items = self._load_split(self.split_path)
        self.video_ids = [self._get_video_id(item) for item in self.items]

    def _resolve_split_path(self, split_path: Path | None) -> Path:
        if split_path is not None:
            if not split_path.exists():
                raise FileNotFoundError(f"split file not found: {split_path}")
            return split_path

        candidates = [
            self.data_root / f"{self.config.split}.json",
            self.data_root / "annotations" / f"{self.config.split}.json",
            Path(__file__).resolve().parents[2] / "ssv2" / f"{self.config.split}.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not locate split json for '{self.config.split}'. "
            "If SSv2 is not present, download it from Qualcomm and set split_path."
        )

    def _resolve_video_dir(self, video_dir: Path | None) -> Path:
        if video_dir is not None:
            if not video_dir.exists():
                raise FileNotFoundError(f"video_dir not found: {video_dir}")
            return video_dir

        candidates = [
            self.data_root,
            self.data_root / "videos",
            self.data_root / "something-something-v2",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Could not locate SSv2 videos directory. Set video_dir explicitly."
        )

    @staticmethod
    def _load_split(path: Path) -> list[dict[str, Any]]:
        with path.open("r") as f:
            data = json.load(f)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return data
            # list of ids
            return [{"id": item} for item in data]
        raise ValueError("Split json must be a list of dicts or ids.")

    @staticmethod
    def _get_video_id(item: dict[str, Any]) -> str:
        for key in ("id", "video_id", "video", "vid"):
            if key in item:
                return str(item[key])
        raise KeyError("Split item missing video id key.")

    def _resolve_video_path(self, video_id: str) -> Path:
        for ext in self.config.video_extensions:
            path = self.video_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        # Fallback: try any matching file
        matches = list(self.video_dir.glob(f"{video_id}.*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Video file not found for id={video_id}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        video_id = self.video_ids[index]
        video_path = self._resolve_video_path(video_id)

        total_frames = get_video_length(video_path)
        seed = int(self.config.seed) + int(index)
        frame_indices = sample_frame_indices(
            num_frames=self.config.num_frames,
            total_frames=total_frames,
            seed=seed,
            mode=self.config.sample_mode,
        )

        frames = decode_video(video_path, frame_indices)
        video = apply_video_transform(frames, self.config.transform)
        return video


class SSV2CachedDataset(SSV2Dataset):
    \"\"\"SSV2 dataset that returns cached teacher targets alongside video.\"\"\"

    def __init__(self, config: SSV2Config, cache_dir: Path) -> None:
        super().__init__(config)
        from src.cache.cache_format import open_cache

        self.cache_dir = cache_dir
        self.cache = open_cache(cache_dir)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video = super().__getitem__(index)
        target = torch.from_numpy(self.cache[index])
        return video, target
