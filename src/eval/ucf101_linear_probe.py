from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import UCF101

from src.models.st_videomae_student import StudentConfig, STVideoMAEStudent
from src.utils.checkpoint import load_checkpoint


def extract_features(model: STVideoMAEStudent, video: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        tokens = model(video)
        return tokens.mean(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # UCF101 expects a directory with videos and split files
    try:
        dataset = UCF101(
            root=str(args.data_root),
            annotation_path=str(args.data_root / "ucfTrainTestlist"),
            frames_per_clip=16,
            step_between_clips=1,
            train=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "UCF101 dataset not found or not prepared. "
            "Ensure videos and ucfTrainTestlist are present under data-root."
        ) from exc

    num_classes = len(dataset.class_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    student = STVideoMAEStudent(StudentConfig())
    predictor = None
    state = torch.load(args.checkpoint, map_location="cpu")
    # Checkpoint stores ModuleList([student, predictor]) state dict with "0." and "1." prefixes.
    student_state = {}
    for k, v in state["model"].items():
        if k.startswith("0.module."):
            student_state[k.replace("0.module.", "")] = v
        elif k.startswith("0."):
            student_state[k.replace("0.", "")] = v
    student.load_state_dict(student_state, strict=False)
    student.to(device)
    student.eval()

    classifier = torch.nn.Linear(student.config.embed_dim, num_classes).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        for clips, _, labels in loader:
            # clips: [B, T, H, W, C]
            clips = clips.permute(0, 4, 1, 2, 3).float() / 255.0
            clips = torch.nn.functional.interpolate(
                clips, size=(16, 224, 224), mode="trilinear", align_corners=False
            )
            clips = clips.to(device)
            labels = labels.to(device)

            feats = extract_features(student, clips)
            logits = classifier(feats)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

    # Eval on train split as minimal check
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, _, labels in loader:
            clips = clips.permute(0, 4, 1, 2, 3).float() / 255.0
            clips = torch.nn.functional.interpolate(
                clips, size=(16, 224, 224), mode="trilinear", align_corners=False
            )
            clips = clips.to(device)
            labels = labels.to(device)

            feats = extract_features(student, clips)
            logits = classifier(feats)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    acc = correct / max(total, 1)
    print(f"UCF101 linear probe top-1 (train split): {acc:.4f}")


if __name__ == "__main__":
    main()
