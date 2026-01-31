# ST-JEPA (SALT-style) -- VideoMAE-H teacher targets

This repo implements a hybrid ST-Transformer student that predicts frozen VideoMAE-H latents cached from SSv2.
See `AGENTS.md` for full constraints and milestone criteria.

## Environment (M0)
```bash
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python

.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv/bin/python -c "import transformers, einops, decord; print('ok')"
```

## Build a small SSv2 cache (M3) -- 1k samples
```bash
./scripts/build_cache_ssv2.sh /mnt/ssv2 /mnt/ssv2/cache_videomae_huge_384 1000
```

## Overfit test (M5) -- 32 clips, 200 iters
```bash
./scripts/pretrain_ssv2.sh /mnt/ssv2 /mnt/ssv2/cache_videomae_huge_384 --overfit
```

## Tests
```bash
make test
# or
.venv/bin/python -m pytest
```

## Notes
- SSv2 must be downloaded from Qualcomm. Do not use unofficial mirrors.
- Teacher is never run during training; only the cached targets are used.
