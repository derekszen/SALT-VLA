.PHONY: test phase-a

test:
	.venv/bin/python -m pytest

phase-a:
	bash scripts/phase_a_smoke.sh
