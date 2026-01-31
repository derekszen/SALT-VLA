from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.ssv2_dataset import SSV2Config, SSV2CachedDataset
from src.models.st_videomae_student import StudentConfig, STVideoMAEStudent
from src.models.predictor import PredictorConfig, JEPAPredictor
from src.train.optim import build_optimizer
from src.utils.dist import init_distributed, is_main_process
from src.utils.checkpoint import save_checkpoint, write_jsonl


def sample_tube_mask(
    batch_size: int,
    grid_t: int,
    grid_s: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    num_mask = int(round(mask_ratio * grid_s))
    mask = torch.zeros(batch_size, grid_t, grid_s, dtype=torch.bool, device=device)
    for b in range(batch_size):
        idx = torch.randperm(grid_s, device=device)[:num_mask]
        mask[b, :, idx] = True
    return mask.view(batch_size, grid_t * grid_s)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssv2"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--log-dir", type=Path, default=Path("run_logs"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-samples", type=int, default=32)
    parser.add_argument("--overfit-iters", type=int, default=200)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if not args.cache_dir.exists():
        raise FileNotFoundError(
            f"Cache dir not found: {args.cache_dir}. Build it with scripts/build_cache_ssv2.sh"
        )

    ssv2_cfg = SSV2Config(
        data_root=args.data_root,
        split=args.split,
        num_frames=16,
        seed=0,
        sample_mode="random",
    )
    dataset = SSV2CachedDataset(ssv2_cfg, cache_dir=args.cache_dir)
    if args.overfit:
        dataset.video_ids = dataset.video_ids[: args.overfit_samples]
        dataset.items = dataset.items[: args.overfit_samples]

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=not args.overfit)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and not args.overfit),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    student_cfg = StudentConfig()
    student = STVideoMAEStudent(student_cfg).to(device)
    predictor_cfg = PredictorConfig(num_tokens=student.num_tokens, dim=student_cfg.embed_dim)
    predictor = JEPAPredictor(predictor_cfg).to(device)

    if world_size > 1:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[local_rank])
        predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[local_rank])

    params = list(student.parameters()) + list(predictor.parameters())
    optimizer = build_optimizer(torch.nn.ModuleList([student, predictor]), args.lr, args.weight_decay)

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    log_path = args.log_dir / "pretrain.jsonl"
    start_time = time.time()

    first_loss = None
    last_loss = None
    step = 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            video, target = batch
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()

            mask = sample_tube_mask(
                batch_size=video.shape[0],
                grid_t=student_cfg.num_frames // student_cfg.tubelet_size,
                grid_s=(student_cfg.img_size // student_cfg.patch_size) ** 2,
                mask_ratio=args.mask_ratio,
                device=device,
            )

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
                tokens = student(video)
                pred_full = predictor(tokens, mask)

                pred_masked = pred_full[mask].view(video.shape[0], -1, student_cfg.embed_dim)
                tgt_masked = target[mask].view(video.shape[0], -1, student_cfg.embed_dim)
                loss = cosine_loss(pred_masked, tgt_masked)

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if first_loss is None:
                first_loss = float(loss.item())
            last_loss = float(loss.item())

            if is_main_process():
                write_jsonl(
                    log_path,
                    {
                        "step": step,
                        "epoch": epoch,
                        "loss": float(loss.item()),
                        "time": time.time() - start_time,
                    },
                )

            if is_main_process() and args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    args.ckpt_dir / f"checkpoint_step{step}.pt",
                    torch.nn.ModuleList([student, predictor]),
                    optimizer,
                    step,
                )

            step += 1
            if args.overfit and step >= args.overfit_iters:
                break
        if args.overfit and step >= args.overfit_iters:
            break

    if args.overfit and first_loss is not None and last_loss is not None:
        if last_loss > 0.7 * first_loss:
            raise AssertionError(
                f"Overfit check failed: first_loss={first_loss:.4f} last_loss={last_loss:.4f}"
            )


if __name__ == "__main__":
    main()
