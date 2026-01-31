"""Test that temp directory fix prevents disk quota errors on training set."""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import tempfile
import time
import torch
import multiprocessing as mp
from torch.utils.data import DataLoader
from src.data.loader import SSv2Dataset, collate_drop_none

# Apply the same fix as in train.py
TORCH_TMP_DIR = "/mnt/ssv2/.torch_tmp"
os.makedirs(TORCH_TMP_DIR, exist_ok=True)
os.environ['TMPDIR'] = TORCH_TMP_DIR
tempfile.tempdir = TORCH_TMP_DIR

print(f"🔧 Temp directory configured: {TORCH_TMP_DIR}")
print(f"   TMPDIR env var: {os.environ.get('TMPDIR')}")

# Use TRAINING set (168K samples) with shuffle=True (the problematic case)
dataset = SSv2Dataset("/mnt/ssv2", split="train")
print(f"📊 Dataset: {len(dataset)} samples (training set)")

mp_context = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,  # This is what caused disk quota issues
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_drop_none,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=4,
    multiprocessing_context=mp.get_context(mp_context)
)

print(f"🚀 Starting 50-batch validation with training set + shuffle=True...")
print(f"   (This previously caused 'disk quota exceeded' errors)")

start = time.time()
batch_times = []
error_count = 0

try:
    for i, batch in enumerate(dataloader):
        if i >= 50:
            break

        batch_start = time.time()

        if batch is not None and torch.cuda.is_available():
            batch = batch.cuda(non_blocking=True)
            torch.cuda.synchronize()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if i % 10 == 0:
            elapsed = time.time() - start
            fps = ((i + 1) * 32) / elapsed
            print(f"   [{i:2d}/50] FPS: {fps:6.2f} | Batch: {batch_time*1000:.1f} ms")

    total_time = time.time() - start
    fps = (50 * 32) / total_time

    print(f"\n✅ SUCCESS - Temp dir fix works!")
    print(f"   • 50 batches completed with training set + shuffle")
    print(f"   • Throughput: {fps:.2f} videos/sec")
    print(f"   • No disk quota errors!")
    print(f"\n📁 Check temp directory:")
    print(f"   ls -lah {TORCH_TMP_DIR}")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
