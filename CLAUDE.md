# SALT VLA-Encoder Project Rules

## 🎯 Goal
Train a ViT-B student via latent regression of a frozen **VideoMAE v1-Huge** teacher.

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
- **Teacher:** `MCG-NJU/videomae-huge`. Frozen, `eval()`.
- **Student:** `timm` ViT-B. `Conv3d` (2, 16, 16) adapter.
- **Inflation:** 1568 positional tokens (8 temporal x 196 spatial).

### 3. Training Layer (`src/train.py`)
- **Precision:** `bfloat16` strictly.
- **Optimization:** Fused `AdamW`
- **Compiler:** `torch.compile(mode="max-autotune")`.
- **Loss:** Masked MSE (penalize masked tokens only).

#### Validated Hyperparameters (as of 2026-01-04)
Based on extensive testing with ViT-Tiny on SSv2 (168K samples, cached latents mode):

**Stable Configuration:**
- `batch_size`: 32 (validated safe, BS=64 causes bus errors)
- `num_workers`: 8-12 (8 conservative, 12 aggressive for 16-core CPU)
- `prefetch_factor`: 2 (reduced from 4 for stability)

**Learning Rate Schedule (CRITICAL):**
- `lr`: **3e-4** (ViT sweet spot, 1e-4 too low → learning stalls)
- `min_lr`: **3e-5** (10% of peak LR, prevents complete decay)
- `warmup_steps`: **200-500** (~1-5% of total steps, proportional to epochs)
- `grad_clip`: **1.0** (gradient clipping for stability at higher LR)
- `weight_decay`: 0.05

**Training Parameters:**
- `mask_ratio`: 0.75 (75% of patches masked)
- `epochs`: 3-10 (3 for rapid prototyping, 10 for convergence)

**Observations:**
- Loss plateau at ~12.5 with LR=1e-4 → Fixed with LR=3e-4 + min_lr floor
- With improved hyperparameters: loss drops from ~15.6 → ~11.3 (27% reduction)
- Throughput: ~560-600 clips/s with cached latents
- GPU memory: 0.83 GB (very efficient with gradient checkpointing)

## 🎛️ Tunable Hyperparameters Reference

### Critical (High Impact on Learning)

| Parameter | Default | Range | Description | Notes |
|-----------|---------|-------|-------------|-------|
| `lr` | 3e-4 | 1e-4 to 1e-3 | Base learning rate | Too low → stalls; too high → instability |
| `min_lr` | 3e-5 | 1e-6 to 1e-4 | Minimum LR floor | Should be 5-10% of peak LR |
| `warmup_steps` | 200-500 | 50 to 1000 | Linear warmup duration | ~1-5% of total steps |
| `mask_ratio` | 0.75 | 0.5 to 0.9 | Fraction of patches masked | Higher = harder task |

### Important (Moderate Impact)

| Parameter | Default | Range | Description | Notes |
|-----------|---------|-------|-------------|-------|
| `batch_size` | 32 | 16 to 32 | Videos per batch | BS=64 unstable, BS=32 max safe |
| `weight_decay` | 0.05 | 0.01 to 0.1 | AdamW weight decay | Regularization strength |
| `grad_clip` | 1.0 | 0.5 to 5.0 | Gradient clipping threshold | Prevents exploding gradients |
| `student_model_name` | vit_tiny | vit_{tiny,small,base} | Student architecture | Tiny fastest, Base best quality |

### Dataloader (Affects Throughput)

| Parameter | Default | Range | Description | Notes |
|-----------|---------|-------|-------------|-------|
| `num_workers` | 8 | 4 to 12 | Parallel data loading | 8 conservative, 12 for 16-core |
| `prefetch_factor` | 2 | 2 to 4 | Batches to prefetch | Higher = more memory usage |

### Training Duration

| Parameter | Default | Range | Description | Notes |
|-----------|---------|-------|-------------|-------|
| `epochs` | 10 | 1 to 50 | Full passes over dataset | 3 for prototyping, 10-20 for convergence |
| `max_steps` | None | Any | Override with step limit | Useful for debugging |

## 📊 Experimental Results

See `EXPERIMENTS.md` for detailed training logs and ablation studies.

### Recent Findings (2026-01-04)

**Experiment: LR Schedule Optimization**
- **Problem:** Loss plateaued at ~12.5 with LR=1e-4, no improvement after 500 steps
- **Hypothesis:** LR too low + aggressive cosine decay → learning stops
- **Solution:** LR=3e-4 + min_lr=3e-5 (10% floor) + longer warmup
- **Result:** ✅ Loss dropped to ~11.3 (1.2 lower), continued learning past 2K steps
- **Impact:** 27% loss reduction, model actually learns video semantics

**Test Coverage (2026-01-04)**
- Created 44 unit tests covering model components and training logic
- Tests validate: PatchEmbed3D, positional embedding inflation, masked MSE, LR schedule
- All tests passing with new hyperparameters

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
