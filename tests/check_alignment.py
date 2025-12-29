from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.loader import SSv2Dataset
from src.models.salt import SALTModel


def _load_single_video(dataset: SSv2Dataset) -> tuple[torch.Tensor, int]:
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is not None:
            return sample, idx
    raise RuntimeError("No valid video samples found in the dataset.")


def _expand_patch_mask(
    patch_mask: torch.Tensor, tubelet_size: int, patch_size: int
) -> torch.Tensor:
    pixel_mask = patch_mask.repeat_interleave(tubelet_size, dim=1)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=2)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=3)
    return pixel_mask


def main() -> None:
    torch.set_float32_matmul_precision("high")

    repo_root = Path(__file__).resolve().parents[1]
    dataset = SSv2Dataset(root_dir=repo_root / "ssv2", split="train")
    video, idx = _load_single_video(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SALTModel()
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)

    video = video.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            masked_video, mask_tokens = model._mask_video(video)
            teacher_latents = model._teacher_tokens(video)
            student_tokens = model.student(masked_video)
            student_pred = model.proj(student_tokens)

    if teacher_latents.shape[1] == student_pred.shape[1] - 1:
        student_pred = student_pred[:, 1:]
        mask_tokens = mask_tokens[:, 1:]
    if mask_tokens.shape[1] != teacher_latents.shape[1]:
        raise ValueError(
            f"Teacher tokens mismatch: mask={mask_tokens.shape[1]} teacher={teacher_latents.shape[1]}"
        )
    if mask_tokens.shape[1] != student_pred.shape[1]:
        raise ValueError(
            f"Student tokens mismatch: mask={mask_tokens.shape[1]} student={student_pred.shape[1]}"
        )

    patch_mask = model.tokens_to_patch_mask(mask_tokens).detach().cpu().float()
    pixel_mask = _expand_patch_mask(
        patch_mask, model.tubelet_size, model.patch_size
    ).bool()

    # Downsample pixel mask to patch grid and confirm alignment with token mask.
    pooled = F.avg_pool3d(
        pixel_mask.float(),
        kernel_size=(model.tubelet_size, model.patch_size, model.patch_size),
        stride=(model.tubelet_size, model.patch_size, model.patch_size),
    )
    if not torch.allclose(pooled, patch_mask, atol=1e-6):
        raise ValueError("Patch mask does not align with pixel mask.")

    masked_pixels = masked_video[0].permute(1, 2, 3, 0)[pixel_mask[0]]
    masked_values = masked_pixels.abs().mean().item()
    print(f"[check] sample_id={idx} masked_mean_abs={masked_values:.6f}")
    if masked_values > 1e-3:
        raise ValueError("Masked pixels are not zeroed out as expected.")

    # Save a heatmap for the first tubelet slice.
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for mask visualization.") from exc

    heatmap = pixel_mask[0][0].float().numpy()
    out_path = repo_root / "tests" / "alignment_mask_t0.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="magma")
    plt.axis("off")
    plt.title("Mask Heatmap (t=0)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[check] saved heatmap: {out_path}")


if __name__ == "__main__":
    main()
