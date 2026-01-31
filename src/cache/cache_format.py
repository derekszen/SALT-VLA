from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


@dataclass(frozen=True)
class CacheSpec:
    num_samples: int
    num_tokens: int
    dim: int
    dtype: str = "float16"


def create_cache(cache_dir: Path, spec: CacheSpec) -> zarr.Array:
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(cache_dir / "targets.zarr"))
    root = zarr.group(store=store)
    targets = root.create_dataset(
        "targets",
        shape=(spec.num_samples, spec.num_tokens, spec.dim),
        chunks=(1, spec.num_tokens, spec.dim),
        dtype=spec.dtype,
    )
    meta = {
        "num_samples": spec.num_samples,
        "num_tokens": spec.num_tokens,
        "dim": spec.dim,
        "dtype": spec.dtype,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return targets


def open_cache(cache_dir: Path) -> zarr.Array:
    store = zarr.DirectoryStore(str(cache_dir / "targets.zarr"))
    root = zarr.group(store=store)
    return root["targets"]


def write_meta_jsonl(cache_dir: Path, entries: list[dict]) -> None:
    path = cache_dir / "meta.jsonl"
    with path.open("w") as f:
        for row in entries:
            f.write(json.dumps(row) + "\n")


def read_meta_jsonl(cache_dir: Path) -> list[dict]:
    path = cache_dir / "meta.jsonl"
    if not path.exists():
        return []
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_projection_matrix(cache_dir: Path, matrix: np.ndarray) -> None:
    path = cache_dir / "projection.npy"
    np.save(path, matrix)
