# SALT-VLA Status

Last updated: 2026-02-01

Project goal
- Train a hybrid ST-Transformer student to predict frozen VideoMAE-H teacher latents (cached) on SSv2, then evaluate with lightweight probes.

Current architecture
- Teacher: HF VideoMAE-H, CLS dropped, cached targets projected 1280->384.
- Cache: zarr float16 targets [N=1568, D=384] + meta.jsonl mapping.
- Student: D=384, depth=12, heads=6; 8 factorized + 4 joint blocks; 4-layer predictor.
- Masking: tube masking over spatial indices shared across time.
- Loss: cosine on masked tokens only.

Implementation status
- Core modules, scripts, configs, and tests are implemented.
- M0–M4 are covered by unit tests.
- M5–M7 scripts exist but require full-data runs to validate.

Operational notes
- SSv2 expected at /mnt/ssv2; cache default at /mnt/ssv2/cache_videomae_huge_384.
- Use scripts/ for cache build, pretrain, and eval entrypoints.
