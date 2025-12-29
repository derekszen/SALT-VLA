#!/bin/bash
# 1. Environment & Network
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1
REPO_ID="vitoria/something-something-v2"
# Fix: Using your existing folder name
TARGET_DIR="./ssv2" 
mkdir -p "$TARGET_DIR/videos"

echo "🚀 Step 1: Downloading via PKU IPv6 (240c) -> hf-mirror.com"
# Using 'hf' via uvx - fastest way to pull archives
uvx --with hf_transfer hf download \
    --repo-type dataset \
    --local-dir "$TARGET_DIR" \
    "$REPO_ID"

echo "🧵 Step 2: Ultra-Parallel Extraction using ripunzip + GNU Parallel"
# Use ripunzip (Rust) inside Parallel to saturate all CPU cores and NVMe IOPS
find "$TARGET_DIR" -name "*.zip" | parallel -j+0 --bar \
    "ripunzip unzip-file {} -d $TARGET_DIR/videos/ && echo '✅ Extracted: {}'"

echo "✨ Step 3: Final Verification"
COUNT=$(ls "$TARGET_DIR/videos" | wc -l)
echo "🎬 Total videos found: $COUNT (Target: ~220,847)"
