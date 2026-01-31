from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cache.cache_format import CacheSpec, create_cache, write_meta_jsonl, write_projection_matrix
from src.data.ssv2_dataset import SSV2Config, SSV2Dataset
from src.models.teacher_videomae import TeacherConfig, VideoMAETeacher
from src.utils.shapes import num_tokens


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
    ds_config = SSV2Config(
        data_root=data_root,
        split=split,
        num_frames=16,
        seed=seed,
        sample_mode="random",
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
    targets = create_cache(cache_dir, spec)

    meta_entries: list[dict] = []
    index = 0

    for batch in tqdm(loader, desc="Caching", total=len(loader)):
        with torch.no_grad():
            projected = teacher(batch)
        projected = projected.cpu().numpy().astype(np.float16)
        targets[index : index + projected.shape[0]] = projected

        for i in range(projected.shape[0]):
            meta_entries.append({"index": index + i, "video_id": dataset.video_ids[index + i]})
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
