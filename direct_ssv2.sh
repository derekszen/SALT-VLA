#!/bin/bash
# 1. High-speed Optimizations for PKU
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Variables
REPO_ID="vitoria/something-something-v2"
TARGET_DIR="./ssv2_dataset"

echo "🚀 Starting Direct SSv2 Download via PKU IPv6 (240c) using 'hf' + 'uvx'..."

# 3. Clean 'hf' command (removed problematic symlink flag)
uvx --with hf_transfer hf download \
    --repo-type dataset \
    --local-dir $TARGET_DIR \
    $REPO_ID
