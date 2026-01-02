"""
Final validation of training configuration before long run.
Tests the exact DataLoader settings that will be used in src/train.py.
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import time
import torch
import multiprocessing as mp
from torch.utils.data import DataLoader
from src.data.loader import SSv2Dataset, collate_drop_none

def validate_training_config():
    """Validate the exact configuration used in train.py."""

    print("="*70)
    print("🔍 TRAINING CONFIGURATION VALIDATION")
    print("="*70)

    # Exact settings from train.py
    data_root = "/mnt/ssv2"
    split = "train"
    batch_size = 32
    num_workers = 8
    prefetch_factor = 4

    print(f"\n📊 Configuration:")
    print(f"   • Data Root: {data_root}")
    print(f"   • Split: {split}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Num Workers: {num_workers}")
    print(f"   • Prefetch Factor: {prefetch_factor}")

    # Check dataset exists
    if not os.path.exists(data_root):
        print(f"\n❌ ERROR: Dataset not found at {data_root}")
        return False

    # Create dataset
    print(f"\n📂 Loading dataset...")
    try:
        dataset = SSv2Dataset(data_root, split=split)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

    # Create dataloader with exact train.py settings
    mp_context = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

    print(f"\n🔧 Creating DataLoader...")
    print(f"   • Multiprocessing context: {mp_context}")
    print(f"   • Persistent workers: True")
    print(f"   • Pin memory: True")
    print(f"   • Drop last: True")

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_drop_none,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=mp.get_context(mp_context)
        )
    except Exception as e:
        print(f"❌ Failed to create DataLoader: {e}")
        return False

    print(f"✅ DataLoader created successfully")

    # Test first batch (preflight check like in train.py)
    print(f"\n🧪 Preflight check (first batch)...")
    try:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)

        if batch is None or batch.numel() == 0:
            print("⚠️  Warning: Empty batch from loader")
            return False

        expected_shape = (batch_size, 3, 16, 224, 224)
        if batch.shape != expected_shape:
            print(f"❌ Shape mismatch: got {batch.shape}, expected {expected_shape}")
            return False

        print(f"✅ First batch shape: {tuple(batch.shape)} - CORRECT")

    except StopIteration:
        print("❌ DataLoader produced no batches")
        return False
    except RuntimeError as e:
        print(f"❌ RuntimeError in loader: {e}")
        return False

    # Run 50-batch validation test
    print(f"\n🚀 Running 50-batch validation test...")
    start_time = time.time()
    batch_times = []
    error_count = 0

    for i in range(50):
        try:
            batch_start = time.time()
            batch = next(dataloader_iter)

            if batch is None:
                continue

            # Transfer to GPU (simulating training)
            if torch.cuda.is_available():
                batch = batch.cuda(non_blocking=True)
                torch.cuda.synchronize()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if i % 10 == 0:
                elapsed = time.time() - start_time
                fps = ((i + 1) * batch_size) / elapsed
                vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"   [{i:2d}/50] FPS: {fps:6.2f} | VRAM: {vram:.2f} GB | "
                      f"Batch: {batch_time*1000:.1f} ms | Errors: {error_count}")

        except StopIteration:
            print(f"⚠️  DataLoader exhausted at batch {i}")
            break
        except RuntimeError as e:
            error_count += 1
            print(f"⚠️  Batch {i} error: {e}")
            if error_count > 5:
                print(f"❌ Too many errors, stopping")
                return False

    # Calculate results
    total_time = time.time() - start_time
    total_videos = len(batch_times) * batch_size
    throughput = total_videos / total_time
    avg_batch_time = sum(batch_times) / len(batch_times) * 1000 if batch_times else 0

    print(f"\n{'='*70}")
    print(f"✅ VALIDATION RESULTS:")
    print(f"   • Batches processed: {len(batch_times)}/50")
    print(f"   • Total videos: {total_videos}")
    print(f"   • Total time: {total_time:.2f} seconds")
    print(f"   • Throughput: {throughput:.2f} videos/sec")
    print(f"   • Avg batch time: {avg_batch_time:.1f} ms")
    print(f"   • Errors: {error_count}")
    print(f"{'='*70}")

    if error_count == 0 and len(batch_times) >= 45:
        print(f"\n🎉 VALIDATION PASSED - Ready for training!")
        print(f"\n💡 To start training, run:")
        print(f"   PYTHONPATH=. uv run python src/train.py")
        return True
    else:
        print(f"\n⚠️  VALIDATION FAILED - Check configuration")
        return False

if __name__ == "__main__":
    import sys

    print("\n🎯 This script validates the exact DataLoader configuration")
    print("   that will be used for training in src/train.py\n")

    success = validate_training_config()

    if success:
        print(f"\n✅ All systems go! Configuration is stable and ready.")
        sys.exit(0)
    else:
        print(f"\n❌ Configuration needs adjustment before training.")
        sys.exit(1)
