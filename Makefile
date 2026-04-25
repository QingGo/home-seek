.PHONY: tests test unit-tests integration-tests lint typecheck clean

tests:
	@echo "=== Running ALL tests (unit + integration) ==="
	HOME_SEEK_WEIGHT_DIR=weights uv run pytest tests/ -v --tb=short

unit-tests:
	@echo "=== Running unit tests (no weights required) ==="
	HOME_SEEK_WEIGHT_DIR=weights uv run pytest tests/ --ignore=tests/integration -v --tb=short

integration-tests:
	@echo "=== Running integration tests (weights required) ==="
	HOME_SEEK_WEIGHT_DIR=weights uv run pytest tests/integration/ -v --tb=short

lint:
	@echo "=== Linting ==="
	uv run ruff check home_seek/ tests/ --fix

typecheck:
	@echo "=== Type checking ==="
	uv run mypy home_seek/ --ignore-missing-imports

clean:
	@echo "=== Cleaning build artifacts ==="
	rm -rf __pycache__ .pytest_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install:
	uv sync

.PHONY: test-run
test-run: unit-tests
	@echo "Quick feedback loop complete."
