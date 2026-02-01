from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cache.cache_format import CacheSpec, create_cache, write_meta_jsonl, write_projection_matrix
from src.data.ssv2_dataset import SSV2Config, SSV2Dataset
from src.data.transforms import transform_hash
from src.data.view import VideoViewConfig, view_hash
from src.models.teacher_videomae import TeacherConfig, VideoMAETeacher
from src.utils.shapes import num_tokens


def _unbatch_meta(metas: object, batch_size: int) -> list[dict]:
    # Default torch collate turns a list[dict] into dict[list|tensor].
    if isinstance(metas, list):
        return [dict(m) for m in metas]
    if not isinstance(metas, dict):
        raise TypeError(f"Unexpected metas type: {type(metas)}")
    out: list[dict] = []
    for i in range(batch_size):
        row: dict = {}
        for k, v in metas.items():
            if torch.is_tensor(v):
                row[k] = v[i].tolist()
            else:
                row[k] = v[i]
        out.append(row)
    return out


def build_cache(
    *,
    data_root: Path,
    split: str,
    cache_dir: Path,
    limit: int | None,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> None:
    view_cfg = VideoViewConfig(
        num_frames=16,
        sample_mode="random",
        seed_base=seed,
    )
    ds_config = SSV2Config(
        data_root=data_root,
        split=split,
        view=view_cfg,
        return_meta=True,
    )
    dataset = SSV2Dataset(ds_config)
    if limit is not None:
        dataset.video_ids = dataset.video_ids[:limit]
        dataset.items = dataset.items[:limit]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    teacher_cfg = TeacherConfig()
    teacher = VideoMAETeacher(teacher_cfg)

    n_tokens = num_tokens(
        num_frames=teacher_cfg.num_frames,
        tubelet_size=teacher_cfg.tubelet_size,
        img_size=teacher_cfg.img_size,
        patch_size=teacher_cfg.patch_size,
    )

    total = len(dataset)
    spec = CacheSpec(num_samples=total, num_tokens=n_tokens, dim=teacher_cfg.target_dim)
    targets = create_cache(
        cache_dir,
        spec,
        extra_meta={
            "view": asdict(view_cfg),
            "view_hash": view_hash(view_cfg),
            "transform_hash": transform_hash(view_cfg.transform),
            "seed_rule": "seed_base + int(video_id)",
        },
    )

    meta_entries: list[dict] = []
    index = 0

    for videos, metas in tqdm(loader, desc="Caching", total=len(loader)):
        with torch.no_grad():
            projected = teacher(videos)
        projected = projected.cpu().numpy().astype(np.float16)
        targets[index : index + projected.shape[0]] = projected

        batch_metas = _unbatch_meta(metas, projected.shape[0])
        for i, m in enumerate(batch_metas):
            meta_entries.append({"index": index + i, **m})
        index += projected.shape[0]

    write_meta_jsonl(cache_dir, meta_entries)
    write_projection_matrix(cache_dir, teacher.proj.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssv2"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_cache(
        data_root=args.data_root,
        split=args.split,
        cache_dir=args.cache_dir,
        limit=args.limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
