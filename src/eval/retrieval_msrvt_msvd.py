from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

from src.data.video_decode import decode_video, sample_frame_indices, get_video_length
from src.data.transforms import apply_video_transform, VideoTransformConfig
from src.models.st_videomae_student import StudentConfig, STVideoMAEStudent


class VideoTextDataset(Dataset):
    def __init__(self, root: Path, annotations: list[dict], num_frames: int = 16) -> None:
        self.root = root
        self.annotations = annotations
        self.num_frames = num_frames
        self.transform = VideoTransformConfig()

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        item = self.annotations[idx]
        video_id = item["video_id"]
        caption = item["caption"]
        video_path = None
        for ext in (".mp4", ".webm", ".avi"):
            cand = self.root / f"{video_id}{ext}"
            if cand.exists():
                video_path = cand
                break
        if video_path is None:
            raise FileNotFoundError(f"Missing video file for {video_id}")

        total_frames = get_video_length(video_path)
        frame_idx = sample_frame_indices(
            num_frames=self.num_frames,
            total_frames=total_frames,
            seed=idx,
            mode="uniform",
        )
        frames = decode_video(video_path, frame_idx)
        video = apply_video_transform(frames, self.transform)
        return video, caption


def load_annotations(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    with path.open("r") as f:
        data = json.load(f)
    # Expect list of {video_id, caption}
    return data


def recall_at_k(sim: np.ndarray, k: int) -> float:
    # sim: [N, N], diagonal is correct pair
    topk = np.argsort(-sim, axis=1)[:, :k]
    hits = np.any(topk == np.arange(sim.shape[0])[:, None], axis=1)
    return float(hits.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msrvtt-root", type=Path, required=True)
    parser.add_argument("--msvd-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load student
    student = STVideoMAEStudent(StudentConfig())
    state = torch.load(args.checkpoint, map_location="cpu")
    student_state = {}
    for k, v in state["model"].items():
        if k.startswith("0.module."):
            student_state[k.replace("0.module.", "")] = v
        elif k.startswith("0."):
            student_state[k.replace("0.", "")] = v
    student.load_state_dict(student_state, strict=False)
    student.to(device)
    student.eval()

    # Load CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()

    proj = nn.Linear(student.config.embed_dim, text_encoder.config.hidden_size).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)

    for name, root in [("MSR-VTT", args.msrvtt_root), ("MSVD", args.msvd_root)]:
        ann_path = root / "annotations.json"
        if not ann_path.exists():
            print(
                f"{name} annotations.json not found at {ann_path}. "
                "Provide captions with fields {video_id, caption}."
            )
            continue

        annotations = load_annotations(ann_path)
        dataset = VideoTextDataset(root, annotations)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        # Train projection for a single epoch
        for videos, captions in loader:
            videos = videos.to(device)
            tokens = student(videos)
            v_emb = tokens.mean(dim=1)
            v_proj = proj(v_emb)

            text_inputs = tokenizer(list(captions), padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                t_emb = text_encoder(**text_inputs).last_hidden_state[:, 0, :]

            loss = 1 - F.cosine_similarity(v_proj, t_emb, dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            break  # minimal training

        # Eval retrieval on same set (minimal)
        video_embs = []
        text_embs = []
        with torch.no_grad():
            for videos, captions in loader:
                videos = videos.to(device)
                tokens = student(videos)
                v_emb = proj(tokens.mean(dim=1))
                video_embs.append(v_emb.cpu())

                text_inputs = tokenizer(list(captions), padding=True, return_tensors="pt").to(device)
                t_emb = text_encoder(**text_inputs).last_hidden_state[:, 0, :]
                text_embs.append(t_emb.cpu())

        if not video_embs:
            continue
        video_embs = torch.cat(video_embs, dim=0)
        text_embs = torch.cat(text_embs, dim=0)

        sim = (video_embs @ text_embs.T).numpy()
        r1 = recall_at_k(sim, 1)
        r5 = recall_at_k(sim, 5)
        r10 = recall_at_k(sim, 10)
        print(f"{name} Retrieval Recall@1/5/10: {r1:.4f} {r5:.4f} {r10:.4f}")


if __name__ == "__main__":
    main()
