import os
import json
import decord
import torch.multiprocessing as mp
from tqdm import tqdm

def verify_video(video_path):
    try:
        # num_threads=1 is the key to stopping the C++ abort
        vr = decord.VideoReader(video_path, num_threads=1)
        if len(vr) > 0:
            return None # File is fine
    except:
        pass
    return video_path

def purge_corrupt_files(json_path, video_root):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    video_paths = [os.path.join(video_root, f"{item.get('id', item.get('name'))}.webm") for item in data]
    os.makedirs(os.path.join(video_root, "corrupt"), exist_ok=True)
    
    print(f"Scanning {len(video_paths)} videos for corruption...")
    
    # Use your high core count to scan
    with mp.Pool(processes=16) as pool:
        corrupt_files = []
        for result in tqdm(pool.imap_unordered(verify_video, video_paths), total=len(video_paths)):
            if result:
                corrupt_files.append(result)
                
    print(f"Found {len(corrupt_files)} unreadable files. Moving to corrupt/...")
    for path in corrupt_files:
        if os.path.exists(path):
            filename = os.path.basename(path)
            os.rename(path, os.path.join(video_root, "corrupt", filename))
    print("Purge complete.")

if __name__ == "__main__":
    purge_corrupt_files("./ssv2/train.json", "./ssv2/videos")