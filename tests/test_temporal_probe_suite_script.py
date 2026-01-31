from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.run_temporal_probe_suite as suite  # noqa: E402


def test_parse_probes_list() -> None:
    assert suite._parse_probes("a,b,c") == ["a", "b", "c"]
    assert suite._parse_probes("  a,  b ,c  ") == ["a", "b", "c"]


def test_parse_probes_empty_raises() -> None:
    with pytest.raises(ValueError):
        suite._parse_probes("")
    with pytest.raises(ValueError):
        suite._parse_probes(" , , ")

