from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.transforms import transform_hash
from src.data.view import VideoViewConfig, build_view, view_hash


@dataclass(frozen=True)
class SSV2Config:
    data_root: Path
    split: str = "train"
    view: VideoViewConfig = field(default_factory=VideoViewConfig)
    video_extensions: tuple[str, ...] = (".webm", ".mp4")
    video_dir: Path | None = None
    split_path: Path | None = None
    return_meta: bool = False


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

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        video_id = self.video_ids[index]
        video_path = self._resolve_video_path(video_id)
        video, meta = build_view(video_path, video_id, self.config.view)
        if self.config.return_meta:
            return video, meta
        return video


class SSV2CachedDataset(SSV2Dataset):
    """SSV2 dataset that returns cached teacher targets alongside video."""

    def __init__(self, config: SSV2Config, cache_dir: Path) -> None:
        super().__init__(config)
        from src.cache.cache_format import open_cache, read_cache_meta, read_meta_jsonl

        self.cache_dir = cache_dir
        self.cache = open_cache(cache_dir)
        self.cache_meta = read_cache_meta(cache_dir)
        self.meta_rows = read_meta_jsonl(cache_dir)

        if not self.meta_rows:
            raise FileNotFoundError(
                f"Cache meta.jsonl not found or empty in: {cache_dir}. Rebuild the cache."
            )

        expected_transform_hash = transform_hash(self.config.view.transform)
        expected_view_hash = view_hash(self.config.view)
        cache_transform_hash = self.cache_meta.get("transform_hash")
        cache_view_hash = self.cache_meta.get("view_hash")
        if cache_transform_hash is not None and cache_transform_hash != expected_transform_hash:
            raise ValueError(
                "Transform mismatch between training and cache. "
                f"expected={expected_transform_hash} cache={cache_transform_hash}"
            )
        if cache_view_hash is not None and cache_view_hash != expected_view_hash:
            raise ValueError(
                "View pipeline mismatch between training and cache. "
                f"expected={expected_view_hash} cache={cache_view_hash}"
            )

        self.video_id_to_cache_row: dict[str, int] = {}
        self.video_id_to_meta: dict[str, dict[str, Any]] = {}
        for row in self.meta_rows:
            vid = str(row.get("video_id"))
            if not vid:
                continue
            cache_row = row.get("index", row.get("cache_row"))
            if cache_row is None:
                continue
            self.video_id_to_cache_row[vid] = int(cache_row)
            self.video_id_to_meta[vid] = row

        # Filter dataset to only clip ids present in the cache mapping.
        keep = [i for i, vid in enumerate(self.video_ids) if vid in self.video_id_to_cache_row]
        self.items = [self.items[i] for i in keep]
        self.video_ids = [self.video_ids[i] for i in keep]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_id = self.video_ids[index]
        video_path = self._resolve_video_path(video_id)
        video, meta = build_view(video_path, video_id, self.config.view)

        cache_meta = self.video_id_to_meta.get(video_id)
        if cache_meta is None:
            raise KeyError(f"video_id missing in cache meta: {video_id}")

        for key in ("transform_hash", "view_hash", "sample_seed", "frame_indices"):
            if key in cache_meta and meta.get(key) != cache_meta.get(key):
                raise ValueError(
                    f"Cache mismatch for video_id={video_id} key={key}: "
                    f"train={meta.get(key)} cache={cache_meta.get(key)}"
                )

        cache_row = self.video_id_to_cache_row[video_id]
        target = torch.from_numpy(self.cache[cache_row])
        return video, target
