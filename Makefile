.PHONY: install lint test-unit test-integration profile profile-compare smoke clean

SHELL := /bin/bash
UV := uv run
PROFILE_DIR := artifacts
LAST_PROFILE := $(PROFILE_DIR)/last_profile.json
PREV_PROFILE := $(PROFILE_DIR)/prev_profile.json

install:
	uv sync

lint:
	$(UV) ruff check src/home_seek/ tests/

test-unit:
	$(UV) python -m pytest tests/ --ignore=tests/integration -m "not slow" -q

test-integration:
	$(UV) python -m pytest tests/integration/ -q

profile: $(PROFILE_DIR)
	$(UV) python -m home_seek.profiling_runner \
		--prompt "Hello" --max-tokens 20 --temperature 0 \
		--output $(LAST_PROFILE)
	@$(UV) python3 scripts/profile_show.py $(LAST_PROFILE)

profile-compare: $(PROFILE_DIR)
	@if [ -f $(PREV_PROFILE) ]; then $(UV) python3 scripts/profile_compare.py $(PREV_PROFILE) $(LAST_PROFILE); else echo "No previous profile at $(PREV_PROFILE). Run make profile twice."; fi

smoke:
	$(UV) python -m home_seek.profiling_runner \
		--prompt "Hello" --max-tokens 20 --temperature 0

server:
	$(UV) python -m home_seek server --port 8000

cli:
	$(UV) python -m home_seek cli --port 8000

$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf home_seek.egg-info
