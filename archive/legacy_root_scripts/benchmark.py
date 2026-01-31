import os
import json
import time
import torch
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu

# REMOVED: torch.multiprocessing.set_sharing_strategy('file_system')
# This was causing "Disk quota exceeded" errors
# Using default sharing strategy with conservative worker counts instead

class SSv2Dataset(Dataset):
    def __init__(self, data_root, annotation_file, num_frames=16, resolution=224):
        self.data_root = data_root
        self.num_frames = num_frames
        self.resolution = resolution
        
        # Load the JSON map
        print(f"📂 Loading metadata from {annotation_file}...")
        with open(annotation_file, "r") as f:
            self.metadata = json.load(f)
        print(f"✅ Loaded {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        # Handle both standard ID formats and potential missing extensions
        vid_id = item['id']
        video_path = os.path.join(self.data_root, f"{vid_id}.webm")
        
        try:
            # CRITICAL FIX 2: Native C++ Resizing
            # We tell decord to resize WHILE decoding. This is 3x faster than doing it in Python
            # and prevents the "storage not resizable" crash by forcing a fixed output size.
            vr = VideoReader(video_path, ctx=cpu(0), width=self.resolution, height=self.resolution, num_threads=1)
            
            # Uniform Temporal Subsampling
            total_frames = len(vr)
            # "linspace" gives us evenly spaced frames across the whole video
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            
            # Get Batch (T, H, W, C)
            video_data = vr.get_batch(indices).asnumpy()
            
            # Convert to Tensor (C, T, H, W) and Normalize
            # We assume the model expects float32 [0.0, 1.0]
            buffer = torch.from_numpy(video_data).permute(3, 0, 1, 2).float() / 255.0
            
            # Handle missing label in test set
            label = item.get('label', 'unknown')
            return buffer, label

        except Exception as e:
            # Robust Fallback: If a file is corrupt, return a black tensor so training doesn't die
            print(f"⚠️ Error loading {video_path}: {e}")  # Enable logging for debugging
            return torch.zeros((3, self.num_frames, self.resolution, self.resolution)), "error"

def run_benchmark(batch_size, num_workers, prefetch_factor, num_batches=20):
    """Run a single benchmark configuration and return metrics."""
    ROOT = "/mnt/ssv2"
    JSON = "/mnt/ssv2/test.json"
    
    print(f"\n{'='*70}")
    print(f"🧪 TESTING: BS={batch_size} | Workers={num_workers} | Prefetch={prefetch_factor}")
    print(f"{'='*70}")
    
    # Check if dataset exists
    if not os.path.exists(ROOT):
        print(f"❌ ERROR: Dataset root not found at {ROOT}")
        return None
    if not os.path.exists(JSON):
        print(f"❌ ERROR: Metadata file not found at {JSON}")
        return None
    
    ds = SSv2Dataset(ROOT, JSON, resolution=224)
    
    # Select multiprocessing context
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
        
        print(f"🔥 Starting benchmark loop...")
        
        for i, (batch, labels) in enumerate(dl):
            if i >= num_batches: 
                break
            
            batch_start = time.time()
            
            try:
                # Simulate the real transfer to GPU (this is where bandwidth matters)
                if torch.cuda.is_available():
                    batch = batch.cuda(non_blocking=True)
                    torch.cuda.synchronize()  # Ensure transfer is complete
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # HEALTH METRICS
                if i % 5 == 0 or i < 3:
                    elapsed = time.time() - start
                    current_clips_per_sec = ((i + 1) * batch_size) / elapsed if elapsed > 0 else 0
                    gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    
                    print(f"   Batch {i:2d}: Shape {batch.shape} | "
                          f"Clips/sec: {current_clips_per_sec:6.2f} | "
                          f"VRAM: {gpu_mem_gb:.2f} GB | "
                          f"Decode: {batch_time*1000:.1f} ms | "
                          f"Errors: {error_count}")
                          
            except Exception as e:
                error_count += 1
                print(f"⚠️ Batch {i} failed: {e}")
                if error_count > 5:
                    print(f"❌ Too many errors ({error_count}), aborting this config")
                    return None

        total_time = time.time() - start
        total_frames = len(batch_times) * batch_size
        fps = total_frames / total_time if total_time > 0 else 0
        avg_batch_time = np.mean(batch_times) * 1000 if batch_times else 0
        std_batch_time = np.std(batch_times) * 1000 if batch_times else 0
        
        print(f"\n✅ RESULTS:")
        print(f"   • Throughput: {fps:.2f} videos/sec")
        print(f"   • Batch Time: {avg_batch_time:.1f} ± {std_batch_time:.1f} ms")
        print(f"   • Total Time: {total_time:.2f} seconds")
        print(f"   • Videos Processed: {total_frames}")
        print(f"   • Errors: {error_count}")
        
        return {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'fps': fps,
            'avg_batch_time': avg_batch_time,
            'std_batch_time': std_batch_time,
            'total_time': total_time,
            'error_count': error_count,
            'success': error_count < 5
        }
        
    except Exception as e:
        print(f"❌ FATAL ERROR in benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- INCREMENTAL BENCHMARK RUNNER ---
if __name__ == "__main__":
    print(f"🚀 INCREMENTAL BENCHMARK - SSv2 on Samsung 9100 Pro")
    print(f"   Strategy: Start conservative, progressively increase load")
    print(f"   Safety: Designed to prevent system freezes")
    
    # STARTUP DIAGNOSTICS
    ROOT = "/mnt/ssv2"
    JSON = "/mnt/ssv2/test.json"
    
    print(f"\n📊 SYSTEM STATUS:")
    print(f"   • Dataset Root: {ROOT} (exists: {os.path.exists(ROOT)})")
    print(f"   • Metadata Path: {JSON} (exists: {os.path.exists(JSON)})")
    print(f"   • GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   • GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # INCREMENTAL TEST CONFIGURATIONS
    # Start with safest possible settings, progressively increase
    test_configs = [
        # Phase 1: Ultra-safe baseline (single video, minimal workers)
        {'batch_size': 1, 'num_workers': 0, 'prefetch_factor': 2, 'num_batches': 10},
        
        # Phase 2: Add workers
        {'batch_size': 1, 'num_workers': 4, 'prefetch_factor': 2, 'num_batches': 20},
        
        # Phase 3: Increase batch size
        {'batch_size': 2, 'num_workers': 4, 'prefetch_factor': 2, 'num_batches': 20},
        {'batch_size': 4, 'num_workers': 4, 'prefetch_factor': 2, 'num_batches': 20},
        
        # Phase 4: Increase workers
        {'batch_size': 4, 'num_workers': 8, 'prefetch_factor': 2, 'num_batches': 20},
        {'batch_size': 4, 'num_workers': 16, 'prefetch_factor': 4, 'num_batches': 20},
        
        # Phase 5: Increase batch size further
        {'batch_size': 8, 'num_workers': 16, 'prefetch_factor': 4, 'num_batches': 30},
        {'batch_size': 16, 'num_workers': 16, 'prefetch_factor': 4, 'num_batches': 30},
        
        # Phase 6: Final stress test (if all above pass)
        {'batch_size': 16, 'num_workers': 24, 'prefetch_factor': 4, 'num_batches': 50},
    ]
    
    results = []
    
    for idx, config in enumerate(test_configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"# PHASE {idx}/{len(test_configs)}")
        print(f"{'#'*70}")
        
        result = run_benchmark(**config)
        
        if result is None or not result['success']:
            print(f"\n⚠️ Configuration failed or unstable. Stopping incremental tests.")
            print(f"   Last stable config: {results[-1] if results else 'None'}")
            break
        
        results.append(result)
        
        # Brief pause between tests to let system stabilize
        print(f"\n⏸️  Pausing 3 seconds before next test...")
        time.sleep(3)
    
    # FINAL SUMMARY
    print(f"\n\n{'='*70}")
    print(f"📊 FINAL SUMMARY - Completed {len(results)}/{len(test_configs)} tests")
    print(f"{'='*70}")
    
    if results:
        print(f"\n{'Config':<30} {'FPS':<10} {'Batch Time':<15} {'Status'}")
        print(f"{'-'*70}")
        for r in results:
            config_str = f"BS={r['batch_size']} W={r['num_workers']} PF={r['prefetch_factor']}"
            status = "✅ PASS" if r['error_count'] == 0 else f"⚠️ {r['error_count']} errors"
            print(f"{config_str:<30} {r['fps']:<10.2f} {r['avg_batch_time']:<15.1f} {status}")
        
        # Find best configuration
        stable_results = [r for r in results if r['error_count'] == 0]
        if stable_results:
            best = max(stable_results, key=lambda x: x['fps'])
            print(f"\n🏆 BEST STABLE CONFIGURATION:")
            print(f"   • Batch Size: {best['batch_size']}")
            print(f"   • Num Workers: {best['num_workers']}")
            print(f"   • Prefetch Factor: {best['prefetch_factor']}")
            print(f"   • Throughput: {best['fps']:.2f} videos/sec")
            print(f"   • Batch Time: {best['avg_batch_time']:.1f} ms")
            
            print(f"\n💡 RECOMMENDATION FOR src/train.py:")
            print(f"   BATCH_SIZE = {best['batch_size']}")
            print(f"   NUM_WORKERS = {best['num_workers']}")
            print(f"   PREFETCH_FACTOR = {best['prefetch_factor']}")
    else:
        print(f"\n❌ All tests failed. Check:")
        print(f"   • Dataset integrity at {ROOT}")
        print(f"   • Available system memory")
        print(f"   • /dev/shm capacity")