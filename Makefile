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
	@python3 -c "
	import json
	with open('$(LAST_PROFILE)') as f:
		d = json.load(f)
	p = d['performance']
	print(f'\\n=== Profile saved to $(LAST_PROFILE) ===')
	print(f'  Decode throughput: {p[\"tokens_per_second\"]:.2f} t/s')
	print(f'  Peak memory: {p[\"peak_memory_gb\"]:.2f} GB')
	"

profile-compare: $(PROFILE_DIR)
	@python3 -c "
	import json, sys
	try:
		with open('$(PREV_PROFILE)') as f:
			prev = json.load(f)['performance']
	except (FileNotFoundError, json.JSONDecodeError):
		print('No previous profile found at $(PREV_PROFILE). Run make profile first.')
		sys.exit(1)
	with open('$(LAST_PROFILE)') as f:
		cur = json.load(f)['performance']
	print(f'{\"Metric\":<35} {\"Previous\":<12} {\"Current\":<12} {\"Change\":<10}')
	print('-' * 70)
	for key in ['tokens_per_second', 'ms_per_token', 'peak_memory_gb']:
		pv = prev.get(key, 0)
		cv = cur.get(key, 0)
		if pv == 0:
			ch = 'N/A'
		else:
			pct = (cv - pv) / pv * 100
			ch = f'{pct:+.1f}%'
		print(f'{key:<35} {pv:<12.4f} {cv:<12.4f} {ch:<10}')
	"

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
