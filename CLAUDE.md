# SALT VLA-Encoder Project Rules

## 🎯 Goal
Train a ViT-B student via latent regression of a frozen **VideoMAEv2-Base** teacher.

## 🚀 Hardware Acceleration: Samsung 9100 Pro
The dataset has been migrated to the Gen 5 NVMe slot to maximize throughput for V-JEPA training.
- **Drive:** Samsung 9100 Pro (PCIe 5.0 x4)
- **Mount Point:** `/mnt/ssv2`
- **FS Type:** XFS (agcount=64, reflink=1)

## 📂 Dataset Paths
- **SSv2 Videos:** `/mnt/ssv2/`
- **Project Root:** `/home/derekszen/Projects/SALT-VLA/`

## 🛠️ Data Loading Optimizations
To utilize the 14GB/s+ potential of the 9100 Pro and minimize GPU starvation:

### 1. Dataloader Settings (PyTorch)
Increase `num_workers` and enable `pin_memory`. With Gen 5, you can safely saturate more workers:
```python
train_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    num_workers=32,        # Increased for Gen 5 concurrency
    pin_memory=True,       # Rapid transfer to GPU
    persistent_workers=True, 
    prefetch_factor=4      # Aggressive prefetching for video chunks
)
```

### 1b. Multiprocessing Context
If workers abort with `terminate called without an active exception`, prefer `forkserver`:
```python
mp_context = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
train_loader = DataLoader(..., multiprocessing_context=mp.get_context(mp_context))
```

### 2. Model Layer (`src/models/salt.py`)
- **Teacher:** `OpenGVLab/VideoMAEv2-Base`. Frozen, `eval()`.
- **Student:** `timm` ViT-B. `Conv3d` (2, 16, 16) adapter.
- **Inflation:** 1568 positional tokens (8 temporal x 196 spatial).

### 3. Training Layer (`src/train.py`)
- **Precision:** `bfloat16` strictly.
- **Optimization:** Fused `AdamW`, `LR 1e-4`, `Batch Size 64`.
- **Compiler:** `torch.compile(mode="max-autotune")`.
- **Loss:** Masked MSE (penalize masked tokens only).

## 📝 Agent Instructions
- **Command:** `PYTHONPATH=. uv run python src/train.py`
- **Safety:** Catch `RuntimeError` from the loader; skip empty batches.
- **VRAM:** Gradient Checkpointing is ENABLED for the Student to allow BS 64.

## 🧪 Preflight Tests (Run Before Long Training)
Add these lightweight checks to catch data loader and shared-memory issues early:
- **Single-batch sanity:** run the loader once and verify non-empty batch shape `(B, 3, 16, 224, 224)`.
- **Worker stress test:** iterate 20-50 batches with the configured `num_workers` and `prefetch_factor` to surface bus errors or quota limits.
- **Shared-memory probe:** run a short load loop while monitoring `/dev/shm` usage to confirm it stays stable.
- **Automated preflight:** `PYTHONPATH=. uv run pytest tests/test_dataloader_preflight.py`

## 🧰 Debug/Logging Guidance
Improve logging to make failures actionable:
- **Loader errors:** log full exception details (including path and worker id) when a batch fails, but keep training running.
- **Startup diagnostics:** print dataset root, split path, videos directory, and effective DataLoader settings at launch.
- **Health metrics:** log `clips_per_sec`, GPU memory usage, and batch decode time at a fixed interval.
- **Crash capture:** on `RuntimeError` in loader, log a compact traceback and continue; on repeated failures, emit a warning counter.

# Agent Instructions

Fast Apply: IMPORTANT: Use \`edit_file\` over \`str_replace\` or full file writes. It works with partial code snippets—no need for full file content.

Warp Grep: warp-grep is a subagent that takes in a search string and tries to find relevant context. Best practice is to use it at the beginning of codebase explorations to fast track finding relevant files/lines. Do not use it to pin point keywords, but use it for broader semantic queries. "Find the XYZ flow", "How does XYZ work", "Where is XYZ handled?", "Where is <error message> coming from?"
