"""Comprehensive DataLoader optimization testing.

Tests multiple optimization strategies:
1. GPU-accelerated video decoding (biggest potential win)
2. Different worker counts with default sharing
3. CPU core pinning
4. Spawn vs forkserver contexts
"""

import os
import json
import time
import torch
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu, gpu

# NO file_system sharing - use default to avoid disk quota issues


class SSv2Dataset(Dataset):
    """Base dataset with CPU decoding."""
    def __init__(self, data_root, annotation_file, num_frames=16, resolution=224):
        self.data_root = data_root
        self.num_frames = num_frames
        self.resolution = resolution

        with open(annotation_file, "r") as f:
            self.metadata = json.load(f)

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
            return torch.zeros((3, self.num_frames, self.resolution, self.resolution)), "error"


class SSv2DatasetGPU(Dataset):
    """Dataset with GPU-accelerated decoding."""
    def __init__(self, data_root, annotation_file, num_frames=16, resolution=224, gpu_id=0):
        self.data_root = data_root
        self.num_frames = num_frames
        self.resolution = resolution
        self.gpu_id = gpu_id

        with open(annotation_file, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        vid_id = item['id']
        video_path = os.path.join(self.data_root, f"{vid_id}.webm")

        try:
            # GPU-accelerated decode
            vr = VideoReader(video_path, ctx=gpu(self.gpu_id), width=self.resolution, height=self.resolution)
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            video_data = vr.get_batch(indices).asnumpy()  # Still returns numpy, but decode on GPU
            buffer = torch.from_numpy(video_data).permute(3, 0, 1, 2).float() / 255.0
            label = item.get('label', 'unknown')
            return buffer, label
        except Exception as e:
            return torch.zeros((3, self.num_frames, self.resolution, self.resolution)), "error"


def worker_init_with_cpu_pinning(worker_id):
    """Pin worker to specific CPU cores to reduce context switching."""
    try:
        # Pin to cores worker_id and worker_id+8 (assuming SMT/HT)
        cores = {worker_id, worker_id + 8}
        os.sched_setaffinity(0, cores)
    except:
        pass  # Silently fail if not supported


def run_optimization_test(
    name,
    dataset_class,
    batch_size=32,
    num_workers=8,
    mp_context_name="forkserver",
    pin_cpus=False,
    num_batches=50,
    **dataset_kwargs
):
    """Run a single optimization test configuration."""
    print(f"\n{'='*70}")
    print(f"🧪 TEST: {name}")
    print(f"   BS={batch_size} | Workers={num_workers} | Context={mp_context_name}")
    print(f"   CPU Pinning={pin_cpus} | Dataset={dataset_class.__name__}")
    print(f"{'='*70}")

    ROOT = "/mnt/ssv2"
    JSON = "/mnt/ssv2/test.json"  # Using test set to avoid disk quota

    try:
        ds = dataset_class(ROOT, JSON, **dataset_kwargs)

        mp_context = mp.get_context(mp_context_name)
        worker_init_fn = worker_init_with_cpu_pinning if pin_cpus else None

        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,  # No shuffle to avoid temp file issues
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
            multiprocessing_context=mp_context if num_workers > 0 else None,
            worker_init_fn=worker_init_fn
        )

        start = time.time()
        batch_times = []
        error_count = 0

        for i, (batch, _) in enumerate(dl):
            if i >= num_batches:
                break

            batch_start = time.time()

            try:
                if torch.cuda.is_available():
                    batch = batch.cuda(non_blocking=True)
                    torch.cuda.synchronize()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                if i % 10 == 0:
                    elapsed = time.time() - start
                    fps = ((i + 1) * batch_size) / elapsed
                    print(f"   [{i:2d}/{num_batches}] FPS: {fps:6.2f} | Batch: {batch_time*1000:.1f} ms")

            except Exception as e:
                error_count += 1
                if error_count > 5:
                    print(f"❌ Too many errors, aborting")
                    return None

        total_time = time.time() - start
        total_videos = len(batch_times) * batch_size
        fps = total_videos / total_time
        avg_batch = np.mean(batch_times) * 1000

        print(f"\n✅ Results:")
        print(f"   • Throughput: {fps:.2f} videos/sec")
        print(f"   • Avg batch time: {avg_batch:.1f} ms")
        print(f"   • Errors: {error_count}")

        return {
            'name': name,
            'fps': fps,
            'avg_batch_ms': avg_batch,
            'errors': error_count,
            'success': error_count == 0
        }

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🚀 COMPREHENSIVE DATALOADER OPTIMIZATION TESTS")
    print("="*70)

    # Baseline from previous testing
    baseline_fps = 237.72

    results = []

    # Test 1: GPU-Accelerated Decoding (HIGHEST POTENTIAL)
    print("\n\n## Test 1: GPU-Accelerated Video Decoding")
    print("Expected gain: 50-150% if decoding is the bottleneck")

    result = run_optimization_test(
        "GPU Decode BS=32 W=8",
        SSv2DatasetGPU,
        batch_size=32,
        num_workers=8,
        mp_context_name="spawn",  # Spawn is safer for GPU context
        gpu_id=0
    )
    if result and result['success']:
        results.append(result)
        gain = ((result['fps'] - baseline_fps) / baseline_fps) * 100
        print(f"\n   💡 Gain over baseline: {gain:+.1f}%")

        # If GPU decode works well, try higher batch size
        if result['fps'] > baseline_fps * 1.2:
            print("\n   🎯 GPU decode shows improvement! Testing higher settings...")
            result2 = run_optimization_test(
                "GPU Decode BS=64 W=8",
                SSv2DatasetGPU,
                batch_size=64,
                num_workers=8,
                mp_context_name="spawn",
                gpu_id=0
            )
            if result2 and result2['success']:
                results.append(result2)

    time.sleep(2)

    # Test 2: CPU Pinning with CPU Decode
    print("\n\n## Test 2: CPU Core Pinning")
    print("Expected gain: 5-15% from reduced context switching")

    result = run_optimization_test(
        "CPU Pinning BS=32 W=8",
        SSv2Dataset,
        batch_size=32,
        num_workers=8,
        mp_context_name="forkserver",
        pin_cpus=True
    )
    if result and result['success']:
        results.append(result)
        gain = ((result['fps'] - baseline_fps) / baseline_fps) * 100
        print(f"\n   💡 Gain over baseline: {gain:+.1f}%")

    time.sleep(2)

    # Test 3: Higher Worker Count with Default Sharing
    print("\n\n## Test 3: Higher Worker Count (Default Sharing)")
    print("Expected gain: 20-40% if workers don't crash")

    for num_workers in [12, 16]:
        result = run_optimization_test(
            f"CPU Decode BS=32 W={num_workers}",
            SSv2Dataset,
            batch_size=32,
            num_workers=num_workers,
            mp_context_name="spawn",  # Spawn might be more stable
            num_batches=50
        )
        if result and result['success']:
            results.append(result)
            gain = ((result['fps'] - baseline_fps) / baseline_fps) * 100
            print(f"\n   💡 Gain over baseline: {gain:+.1f}%")
        else:
            print(f"\n   ⚠️ W={num_workers} failed, stopping worker count tests")
            break
        time.sleep(2)

    # Test 4: Spawn vs Forkserver
    print("\n\n## Test 4: Spawn vs Forkserver Context")
    print("Expected gain: 0-5% (diagnostic test)")

    result = run_optimization_test(
        "Spawn Context BS=32 W=8",
        SSv2Dataset,
        batch_size=32,
        num_workers=8,
        mp_context_name="spawn"
    )
    if result and result['success']:
        results.append(result)
        gain = ((result['fps'] - baseline_fps) / baseline_fps) * 100
        print(f"\n   💡 Gain over baseline: {gain:+.1f}%")

    # Final Summary
    print(f"\n\n{'='*70}")
    print(f"📊 OPTIMIZATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nBaseline: {baseline_fps:.2f} videos/sec (BS=32 W=8 CPU decode, file_system sharing)")
    print(f"\n{'Configuration':<35} {'FPS':<12} {'Gain':<10} {'Status'}")
    print(f"{'-'*70}")

    for r in results:
        gain = ((r['fps'] - baseline_fps) / baseline_fps) * 100
        status = "✅" if r['errors'] == 0 else "⚠️"
        print(f"{r['name']:<35} {r['fps']:<12.2f} {gain:+6.1f}%    {status}")

    if results:
        best = max(results, key=lambda x: x['fps'])
        print(f"\n🏆 BEST CONFIGURATION: {best['name']}")
        print(f"   • Throughput: {best['fps']:.2f} videos/sec")
        print(f"   • Improvement: {((best['fps'] - baseline_fps) / baseline_fps) * 100:+.1f}% over baseline")
        print(f"   • Batch time: {best['avg_batch_ms']:.1f} ms")

        print(f"\n💡 RECOMMENDATION:")
        print(f"   Update src/train.py with settings from: {best['name']}")
    else:
        print(f"\n❌ No optimizations succeeded")
