#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data.loader import SSv2Dataset, collate_drop_none
from src.models.salt import StudentVideoViT


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_dtype(value: str) -> torch.dtype:
    value = value.strip().lower()
    if value in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if value in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _load_ckpt_config(checkpoint: Path) -> dict[str, Any]:
    # Prefer config embedded in the checkpoint, but fall back to sibling config.json if present.
    ckpt_obj = torch.load(checkpoint, map_location="cpu")
    config = ckpt_obj.get("config") if isinstance(ckpt_obj, dict) else None
    if isinstance(config, dict) and config:
        return config
    sibling = checkpoint.parent / "config.json"
    if sibling.exists():
        return json.loads(sibling.read_text(encoding="utf-8"))
    return {}


def _load_student_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    ckpt_obj = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ckpt_obj, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt_obj)}")
    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        state = ckpt_obj["state_dict"]
        if "student" in state and isinstance(state["student"], dict):
            return state["student"]
        return state
    if "student" in ckpt_obj and isinstance(ckpt_obj["student"], dict):
        return ckpt_obj["student"]
    raise KeyError("Could not find student weights in checkpoint.")


def _extract_embedding(tokens: Any) -> torch.Tensor:
    # Timm ViT forward_features usually returns (B, N+1, D). Some variants may return (B, D).
    if isinstance(tokens, (tuple, list)) and tokens:
        tokens = tokens[0]
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Unexpected model output type: {type(tokens)}")
    if tokens.ndim == 3:
        return tokens[:, 0]  # CLS
    if tokens.ndim == 2:
        return tokens
    raise ValueError(f"Unexpected model output shape: {tuple(tokens.shape)}")


def _make_static(videos: torch.Tensor) -> torch.Tensor:
    """Repeat the first frame across time (shape-preserving)."""
    if videos.ndim != 5:
        raise ValueError(f"Expected videos shape (B, C, T, H, W); got {tuple(videos.shape)}")
    t = videos.shape[2]
    return videos[:, :, :1].expand(-1, -1, t, -1, -1)


def _make_temporal_shuffle(videos: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
    """Shuffle frames independently per sample (shape-preserving)."""
    if videos.ndim != 5:
        raise ValueError(f"Expected videos shape (B, C, T, H, W); got {tuple(videos.shape)}")
    b, c, t, h, w = videos.shape
    noise = torch.rand(b, t, generator=rng)  # CPU generator for determinism; indices moved to device below.
    perm = noise.argsort(dim=1).to(device=videos.device)
    idx = perm.view(b, 1, t, 1, 1).expand(b, c, t, h, w)
    return videos.gather(dim=2, index=idx)


def _stride_repeat(videos: torch.Tensor, stride: int) -> torch.Tensor:
    """Downsample time by stride, then repeat to recover original T."""
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    if videos.ndim != 5:
        raise ValueError(f"Expected videos shape (B, C, T, H, W); got {tuple(videos.shape)}")
    t = videos.shape[2]
    if stride == 1:
        return videos
    x = videos[:, :, ::stride]
    x = x.repeat_interleave(stride, dim=2)
    return x[:, :, :t]


def _make_classification_batch(
    videos: torch.Tensor, probe: str, rng: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Create a labeled batch for temporal classification probes.

    Returns:
        x: (B * num_variants, C, T, H, W)
        y: (B * num_variants,)
        variant_names: list of names (len=num_variants) in label order
    """
    probe = probe.strip().lower()
    variants: list[torch.Tensor]
    names: list[str]
    if probe in {"time-arrow", "time_arrow"}:
        variants = [videos, videos.flip(2)]
        names = ["forward", "reverse"]
    elif probe in {"motion-static", "motion_static"}:
        variants = [videos, _make_static(videos)]
        names = ["motion", "static"]
    elif probe in {"temporal-shuffle", "temporal_shuffle"}:
        variants = [videos, _make_temporal_shuffle(videos, rng=rng)]
        names = ["ordered", "shuffled"]
    elif probe in {"stride", "speed", "stride-probe", "stride_probe"}:
        variants = [videos, _stride_repeat(videos, 2), _stride_repeat(videos, 4)]
        names = ["stride1", "stride2", "stride4"]
    else:
        raise ValueError(f"Unknown probe: {probe}")

    bsz = videos.shape[0]
    x = torch.cat(variants, dim=0)
    y = torch.cat([torch.full((bsz,), i, dtype=torch.long) for i in range(len(variants))], dim=0)
    return x, y, names


def _make_temporal_jitter_views(
    videos: torch.Tensor,
    *,
    rng: torch.Generator | None = None,
    subclip_ratio: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create two temporally-jittered views from the same decoded clip (shape-preserving).

    Note: This is intentionally lightweight and avoids re-decoding different windows from disk.
    """
    if videos.ndim != 5:
        raise ValueError(f"Expected videos shape (B, C, T, H, W); got {tuple(videos.shape)}")
    b, c, t, h, w = videos.shape
    sub_len = int(max(2, min(t, round(t * subclip_ratio))))
    max_start = max(0, t - sub_len)
    if max_start == 0:
        return videos, videos

    starts1 = torch.randint(0, max_start + 1, (b,), generator=rng)
    starts2 = torch.randint(0, max_start + 1, (b,), generator=rng)
    base = torch.arange(sub_len).view(1, sub_len)
    idx1 = (starts1.view(b, 1) + base).to(device=videos.device)
    idx2 = (starts2.view(b, 1) + base).to(device=videos.device)

    idx1_exp = idx1.view(b, 1, sub_len, 1, 1).expand(b, c, sub_len, h, w)
    idx2_exp = idx2.view(b, 1, sub_len, 1, 1).expand(b, c, sub_len, h, w)
    v1 = videos.gather(dim=2, index=idx1_exp)
    v2 = videos.gather(dim=2, index=idx2_exp)

    if sub_len == t:
        return v1, v2
    pad = t - sub_len
    v1 = torch.cat([v1, v1[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
    v2 = torch.cat([v2, v2[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
    return v1, v2


def _retrieval_top1_and_sims(emb_a: torch.Tensor, emb_b: torch.Tensor) -> dict[str, float]:
    """Compute top-1 retrieval accuracy and mean pos/neg cosine similarities."""
    if emb_a.ndim != 2 or emb_b.ndim != 2:
        raise ValueError("Expected embeddings of shape (N, D).")
    if emb_a.shape != emb_b.shape:
        raise ValueError(f"Embedding shape mismatch: {tuple(emb_a.shape)} vs {tuple(emb_b.shape)}")
    n = emb_a.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for retrieval.")

    a = emb_a / emb_a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = emb_b / emb_b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = a @ b.T  # (N, N)
    top1 = (sim.argmax(dim=1) == torch.arange(n)).float().mean().item()
    pos = sim.diag().mean().item()
    neg = (sim.sum() - sim.diag().sum()).item() / float(n * (n - 1))
    return {"top1": float(top1), "pos_cos_mean": float(pos), "neg_cos_mean": float(neg)}


@torch.inference_mode()
def _collect_features(
    *,
    student: StudentVideoViT,
    dataset: SSv2Dataset,
    max_videos: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    split_name: str,
    probe: str = "time-arrow",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    from torch.utils.data import DataLoader

    # Deterministic-ish sampling for probe reproducibility.
    g = torch.Generator()
    g.manual_seed(seed)

    variant_rng = torch.Generator()
    variant_rng.manual_seed(seed + 999)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        collate_fn=collate_drop_none,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp_ctx = torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext()

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    cosine_deltas: list[torch.Tensor] = []
    cosine_deltas_extra: list[torch.Tensor] = []

    num_seen = 0
    t0 = time.time()
    for videos in loader:
        if videos.numel() == 0:
            continue
        if num_seen >= max_videos:
            break
        remaining = max_videos - num_seen
        if videos.shape[0] > remaining:
            videos = videos[:remaining]
        bsz = videos.shape[0]

        x_cpu, y_cpu, variant_names = _make_classification_batch(videos, probe, rng=variant_rng)
        x = x_cpu.to(device=device, dtype=dtype, non_blocking=True)
        y = y_cpu

        with amp_ctx:
            tokens = student(x)
            emb = _extract_embedding(tokens).float()

        num_variants = len(variant_names)
        emb_by_variant = [emb[i * bsz : (i + 1) * bsz] for i in range(num_variants)]
        if num_variants >= 2:
            emb_a = emb_by_variant[0]
            emb_b = emb_by_variant[1]
            denom = (emb_a.norm(dim=1) * emb_b.norm(dim=1)).clamp_min(1e-12)
            cos_sim = (emb_a * emb_b).sum(dim=1) / denom
            cosine_deltas.append((1.0 - cos_sim).cpu())
        if num_variants >= 3:
            emb_a = emb_by_variant[0]
            emb_c = emb_by_variant[2]
            denom = (emb_a.norm(dim=1) * emb_c.norm(dim=1)).clamp_min(1e-12)
            cos_sim = (emb_a * emb_c).sum(dim=1) / denom
            cosine_deltas_extra.append((1.0 - cos_sim).cpu())

        features.append(emb.cpu())
        labels.append(y.cpu())
        num_seen += bsz

    if not features:
        raise RuntimeError(f"No valid videos decoded for split={split_name}.")

    X = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)
    cos_delta = torch.cat(cosine_deltas, dim=0) if cosine_deltas else torch.tensor([])
    cos_delta_extra = (
        torch.cat(cosine_deltas_extra, dim=0) if cosine_deltas_extra else torch.tensor([])
    )
    dt = time.time() - t0
    stats = {
        "videos_used": float(num_seen),
        "examples_used": float(X.shape[0]),
        "seconds": float(dt),
        "examples_per_sec": float(X.shape[0] / max(dt, 1e-6)),
        "cosine_delta_mean": float(cos_delta.mean().item()) if cos_delta.numel() else float("nan"),
        "cosine_delta_median": float(cos_delta.median().item()) if cos_delta.numel() else float("nan"),
    }
    if cos_delta_extra.numel():
        stats["cosine_delta_mean_extra"] = float(cos_delta_extra.mean().item())
        stats["cosine_delta_median_extra"] = float(cos_delta_extra.median().item())
    return X, y, stats


@torch.inference_mode()
def _collect_clip_consistency(
    *,
    student: StudentVideoViT,
    dataset: SSv2Dataset,
    max_videos: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    split_name: str,
    subclip_ratio: float = 0.75,
) -> dict[str, float]:
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)
    variant_rng = torch.Generator()
    variant_rng.manual_seed(seed + 999)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        collate_fn=collate_drop_none,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp_ctx = torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext()

    embs_a: list[torch.Tensor] = []
    embs_b: list[torch.Tensor] = []
    num_seen = 0
    t0 = time.time()
    for videos in loader:
        if videos.numel() == 0:
            continue
        if num_seen >= max_videos:
            break
        remaining = max_videos - num_seen
        if videos.shape[0] > remaining:
            videos = videos[:remaining]
        bsz = videos.shape[0]

        v1, v2 = _make_temporal_jitter_views(videos, rng=variant_rng, subclip_ratio=subclip_ratio)
        x = torch.cat([v1, v2], dim=0).to(device=device, dtype=dtype, non_blocking=True)
        with amp_ctx:
            tokens = student(x)
            emb = _extract_embedding(tokens).float()
        embs_a.append(emb[:bsz].cpu())
        embs_b.append(emb[bsz:].cpu())
        num_seen += bsz

    if not embs_a:
        raise RuntimeError(f"No valid videos decoded for split={split_name}.")

    emb_a = torch.cat(embs_a, dim=0)
    emb_b = torch.cat(embs_b, dim=0)
    sims = _retrieval_top1_and_sims(emb_a, emb_b)
    dt = time.time() - t0
    return {
        "videos_used": float(num_seen),
        "seconds": float(dt),
        "examples_per_sec": float((2.0 * num_seen) / max(dt, 1e-6)),
        **sims,
    }


def _normalize_features(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    mode = mode.lower()
    if mode == "none":
        return X_train, X_val, {"norm": "none"}

    if mode == "l2":
        train = X_train / X_train.norm(dim=1, keepdim=True).clamp_min(1e-12)
        val = X_val / X_val.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return train, val, {"norm": "l2"}

    if mode == "zscore":
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        train = (X_train - mean) / std
        val = (X_val - mean) / std
        return train, val, {"norm": "zscore", "z_mean_l2": float(mean.norm().item()), "z_std_mean": float(std.mean().item())}

    raise ValueError(f"Unknown normalization mode: {mode}")


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description="SSv2 temporal probes (time-arrow and variants).")
    ap.add_argument(
        "--probe",
        type=str,
        default="time-arrow",
        choices=[
            "time-arrow",
            "motion-static",
            "temporal-shuffle",
            "stride",
            "clip-consistency",
        ],
        help="Which probe to run (default: time-arrow).",
    )
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoints/<run>/best.pth or last.pth")
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/ssv2"))
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="validation")
    ap.add_argument("--train-videos", type=int, default=2000)
    ap.add_argument("--val-videos", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="", help="bf16|fp16|fp32 (default: infer)")
    ap.add_argument("--probe-epochs", type=int, default=25)
    ap.add_argument("--probe-lr", type=float, default=5e-3)
    ap.add_argument("--probe-weight-decay", type=float, default=0.0)
    ap.add_argument(
        "--subclip-ratio",
        type=float,
        default=0.75,
        help="For clip-consistency: subclip length ratio inside decoded T.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalize", type=str, default="l2", help="none|l2|zscore")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    _set_seed(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    config = _load_ckpt_config(args.checkpoint)
    student_sd = _load_student_state_dict(args.checkpoint)

    student_model_name = str(config.get("student_model_name", "vit_large_patch16_224"))
    tubelet_size = int(config.get("tubelet_size", 2))
    patch_size = int(config.get("patch_size", 16))
    num_frames = int(config.get("num_frames", 16) or 16)
    img_size = int(config.get("img_size", 224) or 224)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if args.dtype.strip():
        dtype = _parse_dtype(args.dtype)
    else:
        cfg_dtype = str(config.get("dtype", "")).strip()
        try:
            dtype = _parse_dtype(cfg_dtype) if cfg_dtype else (torch.bfloat16 if device.type == "cuda" else torch.float32)
        except ValueError:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    student = StudentVideoViT(
        model_name=student_model_name,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        num_frames=num_frames,
        img_size=img_size,
    )
    missing, unexpected = student.load_state_dict(student_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) <= 16:
            print(f"[warn] missing: {missing}")
        if len(unexpected) <= 16:
            print(f"[warn] unexpected: {unexpected}")
    student = student.to(device=device, dtype=dtype)
    student.eval()

    # Avoid accidental gradient allocations.
    for p in student.parameters():
        p.requires_grad_(False)

    train_ds = SSv2Dataset(
        root_dir=args.data_root,
        split=args.train_split,
        num_frames=num_frames,
        frame_size=img_size,
        seed=args.seed,
    )
    val_ds = SSv2Dataset(
        root_dir=args.data_root,
        split=args.val_split,
        num_frames=num_frames,
        frame_size=img_size,
        seed=args.seed + 1,
    )

    probe = args.probe.strip().lower()
    if probe == "clip-consistency":
        t_feat0 = time.time()
        train_stats = _collect_clip_consistency(
            student=student,
            dataset=train_ds,
            max_videos=args.train_videos,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            dtype=dtype,
            seed=args.seed,
            split_name=args.train_split,
            subclip_ratio=args.subclip_ratio,
        )
        val_stats = _collect_clip_consistency(
            student=student,
            dataset=val_ds,
            max_videos=args.val_videos,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            dtype=dtype,
            seed=args.seed + 123,
            split_name=args.val_split,
            subclip_ratio=args.subclip_ratio,
        )
        feat_seconds = time.time() - t_feat0

        result = {
            "benchmark": "ssv2_clip_consistency",
            "checkpoint": str(args.checkpoint),
            "checkpoint_dir": str(args.checkpoint.parent),
            "student_model_name": student_model_name,
            "tubelet_size": tubelet_size,
            "patch_size": patch_size,
            "num_frames": num_frames,
            "img_size": img_size,
            "device": str(device),
            "dtype": str(dtype),
            "seed": args.seed,
            "splits": {"train": args.train_split, "val": args.val_split},
            "sizes": {"train_videos": args.train_videos, "val_videos": args.val_videos},
            "feature_extraction": {"seconds_total": float(feat_seconds)},
            "probe": {
                "name": "clip-consistency",
                "subclip_ratio": float(args.subclip_ratio),
                "train": train_stats,
                "val": val_stats,
            },
        }
    else:
        t_feat0 = time.time()
        X_train, y_train, train_stats = _collect_features(
            student=student,
            dataset=train_ds,
            max_videos=args.train_videos,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            dtype=dtype,
            seed=args.seed,
            split_name=args.train_split,
            probe=probe,
        )
        X_val, y_val, val_stats = _collect_features(
            student=student,
            dataset=val_ds,
            max_videos=args.val_videos,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            dtype=dtype,
            seed=args.seed + 123,
            split_name=args.val_split,
            probe=probe,
        )
        feat_seconds = time.time() - t_feat0

        X_train, X_val, norm_stats = _normalize_features(X_train, X_val, args.normalize)

        num_classes = int(y_train.max().item()) + 1 if y_train.numel() else 0
        if num_classes < 2:
            raise RuntimeError(f"Unexpected num_classes={num_classes} for probe={probe}")

        # Train a tiny linear probe.
        dim = X_train.shape[1]
        clf = nn.Linear(dim, num_classes, bias=True).to(device=device, dtype=torch.float32)
        opt = torch.optim.AdamW(
            clf.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(X_train, y_train)
        dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)

        t_train0 = time.time()
        for epoch in range(args.probe_epochs):
            clf.train()
            for xb, yb in dl:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
                yb = yb.to(device=device, non_blocking=True)
                logits = clf(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            if epoch in {0, args.probe_epochs - 1}:
                clf.eval()
                with torch.inference_mode():
                    train_acc = _accuracy(
                        clf(X_train.to(device=device, dtype=torch.float32)),
                        y_train.to(device=device),
                    )
                    val_acc = _accuracy(
                        clf(X_val.to(device=device, dtype=torch.float32)),
                        y_val.to(device=device),
                    )
                print(
                    f"[probe] epoch={epoch+1}/{args.probe_epochs} "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
                )

        train_seconds = time.time() - t_train0

        clf.eval()
        with torch.inference_mode():
            train_logits = clf(X_train.to(device=device, dtype=torch.float32))
            val_logits = clf(X_val.to(device=device, dtype=torch.float32))
            train_acc = _accuracy(train_logits, y_train.to(device=device))
            val_acc = _accuracy(val_logits, y_val.to(device=device))

        bench_name = "ssv2_time_arrow" if probe == "time-arrow" else f"ssv2_{probe.replace('-', '_')}"
        variant_names_map = {
            "time-arrow": ["forward", "reverse"],
            "motion-static": ["motion", "static"],
            "temporal-shuffle": ["ordered", "shuffled"],
            "stride": ["stride1", "stride2", "stride4"],
        }
        result = {
            "benchmark": bench_name,
            "checkpoint": str(args.checkpoint),
            "checkpoint_dir": str(args.checkpoint.parent),
            "student_model_name": student_model_name,
            "tubelet_size": tubelet_size,
            "patch_size": patch_size,
            "num_frames": num_frames,
            "img_size": img_size,
            "device": str(device),
            "dtype": str(dtype),
            "seed": args.seed,
            "splits": {"train": args.train_split, "val": args.val_split},
            "sizes": {"train_videos": args.train_videos, "val_videos": args.val_videos},
            "feature_extraction": {
                "seconds_total": float(feat_seconds),
                "train": train_stats,
                "val": val_stats,
            },
            "probe": {
                "name": probe,
                "variant_names": variant_names_map.get(probe, []),
                "num_classes": int(num_classes),
                "epochs": args.probe_epochs,
                "lr": args.probe_lr,
                "weight_decay": args.probe_weight_decay,
                "seconds": float(train_seconds),
                "normalize": norm_stats,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            },
        }

    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # Make dataloader behavior friendlier on massive-core machines.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
