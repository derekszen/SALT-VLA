from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2
import torch

from src.data.ssv2_dataset import SSV2Config, SSV2Dataset


def _write_dummy_video(path: Path, num_frames: int = 20, size: int = 64) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (size, size))
    assert writer.isOpened(), "Failed to open VideoWriter"
    for i in range(num_frames):
        frame = np.full((size, size, 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_ssv2_dataset_deterministic_sampling(tmp_path: Path) -> None:
    data_root = tmp_path / "ssv2"
    data_root.mkdir(parents=True)

    video_id = "123"
    video_path = data_root / f"{video_id}.mp4"
    _write_dummy_video(video_path, num_frames=24, size=64)

    split_path = data_root / "train.json"
    with split_path.open("w") as f:
        json.dump([{"id": video_id}], f)

    config = SSV2Config(
        data_root=data_root,
        split="train",
        num_frames=16,
        seed=123,
        sample_mode="random",
    )
    ds1 = SSV2Dataset(config)
    ds2 = SSV2Dataset(config)

    vid1 = ds1[0]
    vid2 = ds2[0]

    assert vid1.shape == (3, 16, 224, 224)
    assert torch.allclose(vid1, vid2), "Deterministic sampling failed"
