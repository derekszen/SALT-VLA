# AGENTS.md -- ST-JEPA (SALT-style, frozen teacher targets) project

## Goal (one sentence)
Train a hybrid ST-Transformer student (factorized early + joint tail) to predict frozen VideoMAE-H teacher latents (cached) on SSv2, then evaluate with lightweight downstream probes.

---

## Non-negotiables (to keep implementation fast + stable)
1) Do NOT re-implement tubelets.
   - Use VideoMAE-style Conv3D tubelet embedding (kernel=(tubelet_size, patch, patch), stride same).
2) Keep a dense, fixed token grid and a stable token ordering:
   - Index: i = t * S + s, where S=(H/patch)*(W/patch), t in [0..T'-1], s in [0..S-1].
3) Use tube masking over spatial indices for v1.
   - Avoid variable per-frame visible counts (no irregular masking in v1).
4) Cache teacher targets to disk (float16) and NEVER run teacher during student training.

---

## Reference shapes (dimension contract)
Assume: num_frames=16, tubelet_size=2 => T'=8, image=224, patch=16 => H'=W'=14, S=196, N=1568.
- Teacher VideoMAE-H hidden size: D_teacher=1280.
- Cache target dim (recommended): D_target=384.
- Cache per sample: z_T shape [N, D_target] float16.
- Student output: z_S shape [B, N, D_target] float16/float32.

If the teacher/student models include a CLS token:
- Drop it for caching and loss: use tokens[:, 1:, :] everywhere.

---

## Architecture specification (v1)
### Teacher
- Frozen HF transformers model: "MCG-NJU/videomae-huge-finetuned-kinetics"
- Extract last_hidden_state (drop CLS), shape [B, N, 1280].
- Apply a fixed projection matrix W (1280 -> 384) during caching only.
  - W must be deterministic and saved to disk (e.g., numpy RNG seed 0).
  - Store z_T = (z_teacher @ W) as float16.

### Student (fastest)
- Base scaffold: VideoMAE-S-sized transformer (D=384, depth=12, heads=6).
- Hybrid blocks:
  - Blocks 0..7: factorized attention (Temporal then Spatial) using SAME qkv/proj parameter shapes as joint attention.
  - Blocks 8..11: standard joint attention.
- Predictor head:
  - 4-layer joint-attention transformer operating on full token grid with mask tokens inserted.
  - Output D_target=384.

### Masking (tube masking)
- Sample mask_spatial subset of {0..S-1}, |mask_spatial| = round(mask_ratio*S).
- Mask indices in token space: M = { t*S + s | s in mask_spatial, t in [0..T'-1] }.
- Loss computed only on M.

### Loss
- Normalize z_T and z_S on dim=-1 and use mean(1 - cosine).

---

## Repo layout (expected)
.
├─ src/
│  ├─ data/
│  │  ├─ ssv2_dataset.py
│  │  ├─ video_decode.py
│  │  └─ transforms.py
│  ├─ models/
│  │  ├─ teacher_videomae.py
│  │  ├─ st_videomae_student.py
│  │  ├─ attention_factorized.py
│  │  └─ predictor.py
│  ├─ cache/
│  │  ├─ build_teacher_cache.py
│  │  └─ cache_format.py
│  ├─ train/
│  │  ├─ pretrain_jepa.py
│  │  └─ optim.py
│  ├─ eval/
│  │  ├─ ucf101_linear_probe.py
│  │  └─ retrieval_msrvt_msvd.py
│  └─ utils/
│     ├─ dist.py
│     ├─ checkpoint.py
│     └─ shapes.py
├─ configs/
│  ├─ pretrain_ssv2.yaml
│  ├─ cache_ssv2.yaml
│  └─ eval_ucf101.yaml
└─ tests/
   ├─ test_shapes_teacher.py
   ├─ test_shapes_student.py
   └─ test_cache_roundtrip.py

---

## Dependencies (pin for reproducibility)
- Python 3.10+
- torch 2.1+ (CUDA)
- torchvision
- transformers (for VideoMAE-H teacher + optional student init)
- accelerate OR pure torchrun (DDP)
- decord OR pytorchvideo (video decode)
- einops
- numpy
- tqdm
- safetensors (optional)
- zarr or webdataset (for caching), prefer zarr for large contiguous arrays

---

## Data acquisition instructions (agent must follow licenses)
### SSv2
If SSv2 not present:
1) Register / download from Qualcomm SSv2 hosting page:
   https://www.qualcomm.com/developer/software/something-something-v-2-dataset
2) Expect webm VP9 videos numbered 1..220847 plus annotations JSONs.

Agent: do NOT attempt to scrape SSv2 from random mirrors.

### MSR-VTT / MSVD (retrieval eval)
Preferred (fast, direct zips on Hugging Face):
- MSR-VTT videos zip:
  https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/MSRVTT_Videos.zip
- MSVD videos zip:
  https://huggingface.co/datasets/friedrichor/MSVD/resolve/main/MSVD_Videos.zip

---

## Milestones + PASS criteria (must implement in order)

### M0 -- Environment boots
PASS when:
- `python -c "import torch; print(torch.cuda.is_available())"` succeeds.
- `python -c "import transformers, einops, decord"` succeeds.

### M1 -- SSv2 dataloader returns correct tensors
Implement: `src/data/ssv2_dataset.py`
PASS when:
- A batch returns `video` tensor shaped [B, C, num_frames, H, W] with C=3.
- Deterministic sampling: with fixed seed and same index, frame IDs are identical.

### M2 -- Teacher forward + projection produces correct target shapes
Implement: `src/models/teacher_videomae.py`
PASS when:
- With random input video, teacher outputs:
  - raw: [B, N(+1), 1280] or [B, N, 1280]
  - projected: [B, N, 384]
- N matches computed N=1568 for 224 and 16 frames (tubelet=2, patch=16).
- Unit test asserts exact shape and dtype.

### M3 -- Cache builder writes + reads roundtrip
Implement: `src/cache/build_teacher_cache.py`
Format recommendation:
- zarr array: `targets.zarr/targets` with shape [num_samples, N, 384] dtype float16
- a parallel `meta.jsonl` with index->video_id mapping
PASS when:
- Cache build on 64 samples completes.
- Roundtrip test: max_abs_diff(read_back, written) == 0 (byte exact) for float16.

### M4 -- Student ST-Transformer runs and matches D_target
Implement: `src/models/st_videomae_student.py`, `src/models/attention_factorized.py`
PASS when:
- Student forward outputs `z_S` shape [B, N, 384].
- Factorized blocks accept the dense grid; reshape correctness verified.
- Hybrid config (8 factorized + 4 joint) produces finite outputs (no NaNs).

### M5 -- End-to-end JEPA training step works
Implement: `src/train/pretrain_jepa.py`
PASS when:
- Single training step runs (forward, loss, backward, optimizer step) on 1 GPU.
- Overfit test: on a tiny subset (e.g., 32 clips), loss decreases by >= 30% within 200 iterations.

### M6 -- Multi-GPU scaling
PASS when:
- `torchrun --nproc_per_node=8 ...` runs for 200 steps without deadlock.
- Loss curves match single-GPU within tolerance (allow minor noise).

### M7 -- Downstream eval scripts exist (minimal)
PASS when:
- UCF101 linear probe script runs end-to-end and reports top-1 accuracy.
- Retrieval script can compute embeddings for MSR-VTT and MSVD and report Recall@K given a CLIP text encoder + trained projection head (even if numbers are low initially).

---

## Hardware usage recommendations
- Local single GPU: run M0--M5 and build a small cache shard (first 1k samples).
- 8x V100 or 8x A800: build full cache + run full pretraining.

---

## Ops notes
- SSv2 expected at `/mnt/ssv2` for fastest local IO.
- Dataloader stability: use `pin_memory=True`, `prefetch_factor=2`, and `num_workers` in the 8--12 range on 16-core CPUs.
- If workers crash with `terminate called without an active exception`, prefer `forkserver` multiprocessing context.
- Preflight sanity: run a single batch and verify shape `(B, 3, 16, 224, 224)` before long runs.

---

## Notes on "zero-shot" claims
This pipeline is video-only pretraining. True zero-shot video<->text requires some video-text alignment stage.
Agent: implement retrieval eval as:
- Freeze video encoder
- Train a small projection head on MSR-VTT train split with frozen CLIP text embeddings
- Evaluate on MSR-VTT 1k-A and MSVD splits

---

## Logging
- Write metrics to stdout + a JSONL file.
- Optional: wandb if credentials exist; otherwise disabled by default.
