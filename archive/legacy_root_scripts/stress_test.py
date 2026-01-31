import torch.multiprocessing
# CRITICAL FIX: Bypass /dev/shm limit
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import json
import time
import torch
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu

class SSv2Dataset(Dataset):
    def __init__(self, data_root, annotation_file, num_frames=16, resolution=224):
        self.data_root = data_root
        self.num_frames = num_frames
        self.resolution = resolution

        print(f"📂 Loading metadata from {annotation_file}...")
        with open(annotation_file, "r") as f:
            self.metadata = json.load(f)
        print(f"✅ Loaded {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        vid_id = item['id']
        video_path = os.path.join(self.data_root, f"{vid_id}.webm")

        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=self.resolution, height=self.resolution, num_threads=1)
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            video_data = vr.get_batch(indices).asnumpy()
            buffer = torch.from_numpy(video_data).permute(3, 0, 1, 2).float() / 255.0
            label = item.get('label', 'unknown')
            return buffer, label
        except Exception as e:
            print(f"⚠️ Error loading {video_path}: {e}")
            return torch.zeros((3, self.num_frames, self.resolution, self.resolution)), "error"

def stress_test(batch_size, num_workers, prefetch_factor, num_batches=100):
    """Run extended stress test."""
    ROOT = "/mnt/ssv2"
    JSON = "/mnt/ssv2/test.json"

    print(f"\n{'='*70}")
    print(f"🔥 STRESS TEST: BS={batch_size} | Workers={num_workers} | Batches={num_batches}")
    print(f"{'='*70}")

    if not os.path.exists(ROOT) or not os.path.exists(JSON):
        print(f"❌ ERROR: Dataset not found")
        return None

    ds = SSv2Dataset(ROOT, JSON, resolution=224)
    mp_context = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

    try:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            multiprocessing_context=mp.get_context(mp_context) if num_workers > 0 else None
        )

        start = time.time()
        batch_times = []
        error_count = 0

        print(f"🚀 Starting {num_batches}-batch stress test...")

        for i, (batch, labels) in enumerate(dl):
            if i >= num_batches:
                break

            batch_start = time.time()

            try:
                if torch.cuda.is_available():
                    batch = batch.cuda(non_blocking=True)
                    torch.cuda.synchronize()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                # Report every 10 batches
                if i % 10 == 0 or i < 5:
                    elapsed = time.time() - start
                    current_fps = ((i + 1) * batch_size) / elapsed if elapsed > 0 else 0
                    gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                    print(f"   [{i:3d}/{num_batches}] Shape {batch.shape} | "
                          f"FPS: {current_fps:6.2f} | "
                          f"VRAM: {gpu_mem_gb:.2f} GB | "
                          f"Batch: {batch_time*1000:.1f} ms | "
                          f"Errors: {error_count}")

            except Exception as e:
                error_count += 1
                print(f"⚠️ Batch {i} failed: {e}")
                if error_count > 10:
                    print(f"❌ Too many errors, aborting")
                    return None

        total_time = time.time() - start
        total_videos = len(batch_times) * batch_size
        fps = total_videos / total_time if total_time > 0 else 0
        avg_batch_time = np.mean(batch_times) * 1000 if batch_times else 0
        std_batch_time = np.std(batch_times) * 1000 if batch_times else 0

        print(f"\n{'='*70}")
        print(f"✅ STRESS TEST RESULTS:")
        print(f"   • Total Videos: {total_videos}")
        print(f"   • Total Time: {total_time:.2f} seconds")
        print(f"   • Throughput: {fps:.2f} videos/sec")
        print(f"   • Batch Time: {avg_batch_time:.1f} ± {std_batch_time:.1f} ms")
        print(f"   • Errors: {error_count}")
        print(f"   • Status: {'✅ PASS' if error_count < 10 else '❌ FAIL'}")
        print(f"{'='*70}")

        return {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'fps': fps,
            'error_count': error_count,
            'success': error_count < 10
        }

    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print(f"🚀 EXTENDED STRESS TEST - SSv2 Dataset")
    print(f"   Purpose: Test higher batch sizes for longer duration")
    print(f"   Target: Find max stable BS for training (goal: BS=64)")

    # System info
    print(f"\n📊 SYSTEM:")
    print(f"   • GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"   • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "")

    # Test configurations - focus on sustainable worker counts
    configs = [
        # Conservative workers, moderate batch size
        {'batch_size': 16, 'num_workers': 8, 'prefetch_factor': 4, 'num_batches': 100},

        # Increase batch size with same workers
        {'batch_size': 32, 'num_workers': 8, 'prefetch_factor': 4, 'num_batches': 100},

        # Target batch size with conservative workers
        {'batch_size': 64, 'num_workers': 8, 'prefetch_factor': 4, 'num_batches': 100},

        # Try moderate worker increase if BS=64 works
        {'batch_size': 64, 'num_workers': 12, 'prefetch_factor': 4, 'num_batches': 100},

        # Maximum attempt - BS=64 with 16 workers
        {'batch_size': 64, 'num_workers': 16, 'prefetch_factor': 4, 'num_batches': 100},
    ]

    results = []

    for idx, config in enumerate(configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"# STRESS TEST {idx}/{len(configs)}")
        print(f"{'#'*70}")

        result = stress_test(**config)

        if result is None or not result['success']:
            print(f"\n⚠️ Configuration UNSTABLE or FAILED")
            print(f"   Reverting to last stable: {results[-1] if results else 'None'}")
            break

        results.append(result)

        # Longer pause between stress tests
        print(f"\n⏸️  Cooling down 5 seconds...")
        time.sleep(5)

    # Summary
    print(f"\n\n{'='*70}")
    print(f"📊 STRESS TEST SUMMARY")
    print(f"{'='*70}")

    if results:
        print(f"\n{'Config':<20} {'FPS':<10} {'Errors':<10} {'Status'}")
        print(f"{'-'*70}")
        for r in results:
            status = "✅ STABLE" if r['error_count'] == 0 else f"⚠️ {r['error_count']} errors"
            print(f"BS={r['batch_size']:2d} W={r['num_workers']:2d}          {r['fps']:<10.2f} {r['error_count']:<10} {status}")

        best = max(results, key=lambda x: x['batch_size'] if x['error_count'] == 0 else 0)

        print(f"\n🏆 MAXIMUM STABLE BATCH SIZE: {best['batch_size']}")
        print(f"   • Workers: {best['num_workers']}")
        print(f"   • Throughput: {best['fps']:.2f} videos/sec")
        print(f"   • Error Rate: {best['error_count']} errors in 100 batches")

        print(f"\n💡 FINAL RECOMMENDATION FOR src/train.py:")
        print(f"   BATCH_SIZE = {best['batch_size']}")
        print(f"   NUM_WORKERS = {best['num_workers']}")
        print(f"   PREFETCH_FACTOR = 4")

        if best['batch_size'] >= 64:
            print(f"\n✅ ACHIEVED TARGET: BS=64 is stable!")
        else:
            print(f"\n⚠️ Could not reach BS=64. Max stable: BS={best['batch_size']}")
            print(f"   This may be due to system RAM or other constraints.")
    else:
        print(f"\n❌ All stress tests failed")
