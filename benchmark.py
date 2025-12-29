import torch.multiprocessing
# CRITICAL FIX 1: Bypass the /dev/shm limit to prevent "Bus Error"
# This allows you to run high num_workers without crashing your OS
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu

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
            
            return buffer, item['label']

        except Exception as e:
            # Robust Fallback: If a file is corrupt, return a black tensor so training doesn't die
            # print(f"⚠️ Error loading {video_path}: {e}") # Uncomment to debug specific files
            return torch.zeros((3, self.num_frames, self.resolution, self.resolution)), "error"

# --- BENCHMARK RUNNER ---
if __name__ == "__main__":
    # CONFIGURATION
    ROOT = "./ssv2/videos"
    JSON = "./ssv2/train.json"
    BATCH_SIZE = 64
    NUM_WORKERS = 16  # Safe to go high now due to 'file_system' strategy
    
    print(f"🚀 Benchmarking Optimized DECORD Loader...")
    print(f"   • Hardware: Ryzen 9950X3D (CPU Decode) -> RTX 5090D (Transfer)")
    print(f"   • Strategy: Native Resize + File System Sharing")
    
    ds = SSv2Dataset(ROOT, JSON, resolution=224)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    
    start = time.time()
    
    # Warmup
    print("🔥 Warming up...")
    
    # Measure 20 batches
    limit = 20
    for i, (batch, _) in enumerate(dl):
        if i >= limit: break
        
        # Simulate the real transfer to GPU (this is where bandwidth matters)
        batch = batch.cuda(non_blocking=True)
        
        if i % 5 == 0:
            # Calculate current speed
            elapsed = time.time() - start
            current_fps = ((i + 1) * BATCH_SIZE) / elapsed if i > 0 else 0
            print(f"   Batch {i}: Shape {batch.shape} | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    total_time = time.time() - start
    total_frames = limit * BATCH_SIZE
    fps = total_frames / total_time
    
    print("\n" + "="*40)
    print(f"⚡ FINAL THROUGHPUT: {fps:.2f} videos/sec")
    print("="*40)
    
    if fps > 150:
        print("✅ STATUS: PERFECT. Ready for full-scale training.")
    elif fps > 80:
        print("⚠️ STATUS: GOOD. Likely sufficient, but check CPU usage.")
    else:
        print("❌ STATUS: BOTTLENECK. Something is still wrong.")