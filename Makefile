.PHONY: install lint test-unit test-integration profile smoke clean

SHELL := /bin/bash
UV := uv run

install:
	uv sync

lint:
	$(UV) ruff check home_seek/ tests/

test-unit:
	$(UV) python -m pytest tests/ --ignore=tests/integration -m "not slow" -q

test-integration:
	$(UV) python -m pytest tests/integration/ -q

profile:
	$(UV) python -m home_seek.profiling_runner --prompt "Hello" --max-tokens 20 --temperature 0

smoke:
	$(UV) python -m home_seek.profiling_runner --prompt "Hello" --max-tokens 20 --temperature 0

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf home_seek.egg-info
