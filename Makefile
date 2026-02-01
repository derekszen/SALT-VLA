.PHONY: test phase-a

test:
	.venv/bin/python -m pytest

phase-a:
	./scripts/phase_a_smoke.sh
