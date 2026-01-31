from __future__ import annotations

from pathlib import Path

import numpy as np

from src.cache.cache_format import CacheSpec, create_cache, open_cache


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    spec = CacheSpec(num_samples=2, num_tokens=1568, dim=384)
    arr = create_cache(cache_dir, spec)

    data = (np.random.randn(2, 1568, 384).astype(np.float16))
    arr[:] = data

    arr2 = open_cache(cache_dir)
    loaded = np.asarray(arr2[:])

    assert loaded.dtype == np.float16
    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)
