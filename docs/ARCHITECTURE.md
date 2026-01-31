# ST-JEPA Architecture (v1)

This project implements a SALT-style JEPA pretraining pipeline using cached VideoMAE-H teacher targets.
For full constraints and milestone criteria, see `AGENTS.md`.

## Core contracts
- Input: video tensor [B, 3, 16, 224, 224].
- Tubelet embedding: Conv3D with kernel=(2,16,16), stride=(2,16,16), producing N=1568 tokens (no CLS).
- Teacher: VideoMAE-H (D_teacher=1280), frozen.
- Cached targets: float16, shape [N, 384], written once and reused during training.
- Student: D=384, depth=12, heads=6; hybrid attention (8 factorized + 4 joint).
- Masking: tube masking over spatial indices, shared across time.
- Loss: cosine on masked tokens only.

## Module layout
- `src/data/`: SSv2 decode + deterministic frame sampling + transforms.
- `src/models/`: teacher wrapper, factorized attention blocks, student backbone, predictor head.
- `src/cache/`: zarr cache writer + reader.
- `src/train/`: JEPA pretraining loop with DDP + AMP.
- `src/eval/`: UCF101 linear probe and retrieval eval (MSR-VTT/MSVD).
- `src/utils/`: shapes, dist, checkpoint, logging helpers.

## Training rule
- Teacher is never run during student training; training reads cached targets only.
