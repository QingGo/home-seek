.PHONY: tests test unit-tests integration-tests lint typecheck clean

SHELL := /bin/bash
.PHONY: tests test unit-tests integration-tests lint typecheck clean

tests:
	@echo "=== Running ALL tests (unit + integration) ==="
	source .venv/bin/activate && python -m pytest tests/ -v --tb=short --durations=10

unit-tests:
	@echo "=== Running fast unit tests ==="
	source .venv/bin/activate && python -m pytest tests/ -m "not slow" -v --tb=short --durations=10

integration-tests:
	@echo "=== Running integration tests (weights required) ==="
	source .venv/bin/activate && python -m pytest tests/integration/ -v --tb=short --durations=10

lint:
	@echo "=== Linting ==="
	source .venv/bin/activate && ruff check home_seek/ tests/ --fix

typecheck:
	@echo "=== Type checking ==="
	source .venv/bin/activate && python -m mypy home_seek/ --ignore-missing-imports

clean:
	@echo "=== Cleaning ==="
	rm -rf __pycache__ .pytest_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install:
	@echo "=== Installing ==="
	uv sync

test-run: unit-tests
	@echo "Quick feedback loop complete."
