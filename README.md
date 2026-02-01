# ST-JEPA Video Pretraining

Train a hybrid ST-Transformer student to predict frozen VideoMAE-H teacher latents cached from SSv2.
Teacher never runs during student training; training reads cached targets only.

## Architecture
- **Teacher**: VideoMAE-H (`MCG-NJU/videomae-huge-finetuned-kinetics`), frozen
- **Targets**: cached float16 `[N, 384]` tokens (drop CLS)
- **Student**: D=384, 12 blocks (first 8 factorized T→S, last 4 joint)
- **Masking**: tube masking (shared spatial mask across time)
- **Loss**: cosine on masked tokens only

## Minimum hardware
- **Linux**
- **GPU**: 1x NVIDIA GPU (>= 16GB VRAM recommended)
- **Disk**: enough for SSv2 + cache (hundreds of GB for full set)

## Setup Linux
```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python
```

## Build a small cache
```bash
./scripts/build_cache_ssv2.sh /mnt/ssv2 /mnt/ssv2/cache_videomae_huge_384 1000
```

## Run a 4-epoch pretrain
```bash
./scripts/pretrain_ssv2.sh /mnt/ssv2 /mnt/ssv2/cache_videomae_huge_384
```

## Simplest UCF-101 inference
```bash
./scripts/ucf101_infer.sh /mnt/ucf101 checkpoints/checkpoint_step0.pt
```

## Tests
```bash
make test
# or
.venv/bin/python -m pytest
```

What tests cover (brief):
- **SSv2 dataloader**: deterministic frame sampling and tensor shapes
- **Teacher wrapper**: VideoMAE-H token/projection shapes
- **Cache format**: zarr roundtrip is byte-exact
- **Student model**: output shape + finite values

Notes:
- SSv2 download must use the official Qualcomm portal. Do not use unofficial mirrors.
- If you are in Beijing or mainland China and downloads are slow, use the official Qualcomm portal via a stable network or VPN allowed by your institution. Avoid third-party mirrors.
- UCF-101 data must be prepared under `/mnt/ucf101` with `ucfTrainTestlist/`.
- If SSv2 or UCF-101 are not present, download from official sources only.

## SSv2 download steps
1) Register and download from the official Qualcomm portal:
   https://www.qualcomm.com/developer/software/something-something-v-2-dataset
2) Place videos and split JSONs under `/mnt/ssv2` or pass `--data-root` to scripts.
3) Expected files:
   - `train.json`, `validation.json`, `test.json`
   - video files named by id, e.g., `12345.webm`
