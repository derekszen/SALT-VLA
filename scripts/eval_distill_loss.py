#!/usr/bin/env python3
"""Evaluate SALT distillation loss on a dataset split.

This loads a checkpoint produced by `src/train.py` (student + predictor only)
and computes MSE(pred_masked, teacher_masked) on the requested split.

Example:
  PYTHONPATH=. ./.venv/bin/python scripts/eval_distill_loss.py \\
    --checkpoint checkpoints/<run>/best.pth \\
    --split validation \\
    --max-batches 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.loader import SSv2MaskedDataset, collate_drop_none_with_masks
from src.models.salt import SALTModel


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Unexpected checkpoint format (expected dict).")
    if "state_dict" not in ckpt:
        raise ValueError("Checkpoint missing state_dict.")
    return ckpt


def _resolve_config(
    ckpt: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    cfg = dict(ckpt.get("config") or {})
    # Allow CLI overrides (useful if you renamed things).
    cfg.setdefault("data_root", args.data_root)
    cfg.setdefault("split", args.split)
    cfg.setdefault("mask_ratio", args.mask_ratio)
    cfg.setdefault("masking_strategy", args.masking_strategy)
    cfg.setdefault("tubelet_size", args.tubelet_size)
    cfg.setdefault("patch_size", args.patch_size)
    cfg.setdefault("student_model_name", args.student_model_name)
    cfg.setdefault("predictor_dim", args.predictor_dim)
    cfg.setdefault("predictor_depth", args.predictor_depth)
    cfg.setdefault("predictor_num_heads", args.predictor_num_heads)
    cfg.setdefault("teacher_name", args.teacher_name)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SALT distillation loss on a split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth/last.pth")
    parser.add_argument("--data-root", type=str, default="/mnt/ssv2")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = evaluate full split")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--masking-strategy", type=str, default="multiblock")
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--student-model-name", type=str, default="vit_large_patch16_224")
    parser.add_argument("--predictor-dim", type=int, default=512)
    parser.add_argument("--predictor-depth", type=int, default=12)
    parser.add_argument("--predictor-num-heads", type=int, default=8)
    parser.add_argument("--teacher-name", type=str, default="Tianjiao-Yu/videomae-huge")
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = _load_checkpoint(ckpt_path)
    cfg = _resolve_config(ckpt, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = SALTModel(
        teacher_name=str(cfg.get("teacher_name") or args.teacher_name),
        load_teacher=True,
        student_model_name=str(cfg.get("student_model_name") or args.student_model_name),
        tubelet_size=int(cfg.get("tubelet_size") or args.tubelet_size),
        patch_size=int(cfg.get("patch_size") or args.patch_size),
        num_frames=16,
        img_size=224,
        mask_ratio=float(cfg.get("mask_ratio") or args.mask_ratio),
        masking_strategy=str(cfg.get("masking_strategy") or args.masking_strategy),
        predictor_dim=int(cfg.get("predictor_dim") or args.predictor_dim),
        predictor_depth=int(cfg.get("predictor_depth") or args.predictor_depth),
        predictor_num_heads=int(cfg.get("predictor_num_heads") or args.predictor_num_heads),
        dtype=dtype,
    )

    state = ckpt.get("state_dict") or {}
    if "student" not in state or "predictor" not in state:
        raise ValueError("Checkpoint state_dict must include 'student' and 'predictor'.")

    missing_student = model.student.load_state_dict(state["student"], strict=True)
    missing_predictor = model.predictor.load_state_dict(state["predictor"], strict=True)
    if missing_student.missing_keys or missing_student.unexpected_keys:
        raise ValueError(f"Student state_dict mismatch: {missing_student}")
    if missing_predictor.missing_keys or missing_predictor.unexpected_keys:
        raise ValueError(f"Predictor state_dict mismatch: {missing_predictor}")

    model.eval()
    model.to(device=device, dtype=dtype)

    dataset = SSv2MaskedDataset(
        root_dir=Path(args.data_root),
        split=args.split,
        num_frames=16,
        frame_size=224,
        mask_ratio=float(cfg.get("mask_ratio") or args.mask_ratio),
        masking_strategy=str(cfg.get("masking_strategy") or args.masking_strategy),
        tubelet_size=int(cfg.get("tubelet_size") or args.tubelet_size),
        patch_size=int(cfg.get("patch_size") or args.patch_size),
    )

    loader_kwargs: dict[str, Any] = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_drop_none_with_masks,
        **loader_kwargs,
    )

    mse_sum = 0.0
    teacher_energy_sum = 0.0
    num_batches = 0

    max_batches = args.max_batches if args.max_batches and args.max_batches > 0 else None
    with torch.no_grad():
        for batch in loader:
            videos, visible_idx, masked_idx = batch
            if videos.numel() == 0:
                continue

            videos = videos.to(device=device, dtype=dtype, non_blocking=True)
            visible_idx = visible_idx.to(device=device, non_blocking=True)
            masked_idx = masked_idx.to(device=device, non_blocking=True)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_masked, teacher_masked = model(
                        videos, visible_idx=visible_idx, masked_idx=masked_idx
                    )
                    mse = torch.nn.functional.mse_loss(pred_masked, teacher_masked)
                    teacher_energy = (teacher_masked.float().pow(2)).mean()
            else:
                pred_masked, teacher_masked = model(
                    videos, visible_idx=visible_idx, masked_idx=masked_idx
                )
                mse = torch.nn.functional.mse_loss(pred_masked, teacher_masked)
                teacher_energy = (teacher_masked.float().pow(2)).mean()

            mse_sum += float(mse.item())
            teacher_energy_sum += float(teacher_energy.item())
            num_batches += 1

            if args.log_interval and num_batches % args.log_interval == 0:
                avg = mse_sum / num_batches
                baseline = teacher_energy_sum / num_batches
                print(
                    f"[eval] batches={num_batches} mse={avg:.4f} "
                    f"teacher_E[x^2]={baseline:.4f} mse/baseline={avg/max(baseline,1e-12):.3f}",
                    flush=True,
                )

            if max_batches is not None and num_batches >= max_batches:
                break

    if num_batches == 0:
        raise RuntimeError("No valid batches produced (all empty / decode failures).")

    avg_mse = mse_sum / num_batches
    avg_teacher_energy = teacher_energy_sum / num_batches
    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "batches": num_batches,
        "mse": avg_mse,
        "teacher_energy": avg_teacher_energy,
        "mse_over_teacher_energy": avg_mse / max(avg_teacher_energy, 1e-12),
        "device": str(device),
        "dtype": str(dtype),
    }
    payload = json.dumps(summary, indent=2)
    print(payload)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
