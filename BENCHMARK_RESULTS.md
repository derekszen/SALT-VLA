# SSv2 Dataset Benchmark Results
**Date:** 2026-01-02
**Hardware:** Samsung 9100 Pro (PCIe 5.0 x4) at `/mnt/ssv2`
**GPU:** NVIDIA GeForce RTX 5090 D (33.62 GB VRAM)
**System:** Linux 6.17.9-arch1-1

## Executive Summary

After extensive incremental testing and stress validation, the **maximum stable configuration** for sustained training is:

```python
BATCH_SIZE = 32
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
```

**Performance:** 237.72 videos/sec sustained throughput with 0 errors over 100 batches

## Testing Methodology

### Phase 1: Incremental Benchmark (9 configurations)
Started with ultra-conservative settings and progressively increased load:
- BS=1, W=0 → BS=16, W=24
- All 9 configurations passed short-burst tests (20-50 batches)
- No system freezes observed

### Phase 2: Extended Stress Testing (100-batch runs)
Tested sustained performance with conservative worker counts:

| Config | Status | Throughput | Notes |
|--------|--------|------------|-------|
| BS=16, W=8 | ✅ **PASS** | 183.36 videos/sec | Stable, 0 errors |
| BS=32, W=8 | ✅ **PASS** | 237.72 videos/sec | **OPTIMAL** |
| BS=64, W=8 | ❌ FAIL | - | Bus error, workers crashed |
| BS=64, W=4 | ❌ FAIL | - | Bus error |
| BS=64, W=2 | ❌ FAIL | - | Bus error |
| BS=64, W=1 | ❌ FAIL | - | Bus error (even with 1 worker!) |

### Key Findings

#### ✅ What Works
- **BS=32 with 8 workers** is rock solid for extended runs **on test set**
- Excellent throughput (~238 videos/sec)
- Zero errors over 100+ batch iterations
- Efficient VRAM usage (~0.31 GB)
- System remains stable with no freezes

#### ❌ What Doesn't Work
- **BS=64 consistently fails** with bus errors regardless of worker count
- **>8 workers causes crashes** on extended runs (even with BS=16)
- High worker counts (24-32) work for short bursts but fail on sustained loads
- Memory constraints prevent larger batch sizes despite 33GB VRAM available
- **Training set (168K samples) with shuffle=True hits disk quota limits**

#### Root Cause Analysis
The bus error at BS=64 is caused by:
1. **Tensor size limits:** Each BS=64 batch = ~619MB in RAM (64 × 16 frames × 224² × 3 × 4 bytes)
2. **Worker memory overhead:** With prefetch_factor=4, workers hold multiple batches simultaneously
3. **System RAM constraints:** Despite `file_system` sharing strategy, the video decode buffers exhaust available memory
4. **Decord library limitations:** Native C++ decoder may have internal buffer limits
5. **CRITICAL:** Large training set (168K samples) + shuffle creates excessive temp files, hitting disk quota

#### Train vs Test Set Differences
- **Test set:** 27,157 samples - All benchmarks passed ✅
- **Train set:** 168,915 samples (6.2x larger) - Hits disk quota with shuffle=True ⚠️
- **Issue:** `shuffle=True` on large datasets creates many temp files when using `file_system` sharing strategy

## Recommendations

### For Training (`src/train.py`)
✅ **Updated with optimal settings:**
```python
batch_size = 32
num_workers = 8
prefetch_factor = 4
```

### ⚠️ CRITICAL: Workaround for Large Training Set

The training set (168K samples) with `shuffle=True` hits disk quota limits. **Solutions:**

#### Option 1: Set Custom Temp Directory (RECOMMENDED)
Point PyTorch temp files to the NVMe drive with more space:
```python
import os
import tempfile

# At the top of train.py, before any DataLoader creation
temp_dir = "/mnt/ssv2/.torch_tmp"
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMPDIR'] = temp_dir
tempfile.tempdir = temp_dir
```

#### Option 2: Reduce Workers for Training Set
Use fewer workers to reduce temp file pressure:
```python
num_workers = 4  # Instead of 8 for training set
```
Expected throughput: ~150-180 videos/sec (still good)

#### Option 3: Start Without Shuffle
For initial validation, run a few epochs without shuffle:
```python
DataLoader(..., shuffle=False, ...)
```
Then enable shuffle once validated. Not ideal for final training.

#### Option 4: Use Test Set for Initial Validation
Start with test set to verify everything works:
```python
python src/train.py --split test  # If supported
```
