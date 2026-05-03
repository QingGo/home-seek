.PHONY: install lint test-unit test-integration profile profile-light profile-compare smoke clean

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
		--rounds 5 --prompts "Hello" "What is AI?" "Write a poem" "How are you?" "Hi" --max-tokens 30 --temperature 0 \
		--output $(LAST_PROFILE) --profile-mode full
	@$(UV) python3 scripts/profile_show.py $(LAST_PROFILE)

profile-light: $(PROFILE_DIR)
	$(UV) python -m home_seek.profiling_runner \
		--rounds 5 --prompts "Hello" "What is AI?" "Write a poem" "How are you?" "Hi" --max-tokens 30 --temperature 0 \
		--output $(LAST_PROFILE) --profile-mode light
	@$(UV) python3 scripts/profile_show.py $(LAST_PROFILE)

profile-compare: $(PROFILE_DIR)
	@if [ -f $(PREV_PROFILE) ]; then $(UV) python3 scripts/profile_compare.py $(PREV_PROFILE) $(LAST_PROFILE); else echo "No previous profile at $(PREV_PROFILE). Run make profile twice."; fi

# ── Nsight Systems (stream 级 profiler, 看 CUDA stream 并行性) ─────────
PROFILE_NSYS := $(PROFILE_DIR)/nsys_trace

profile-nsys: $(PROFILE_DIR)
	@echo "=== Checking Nsight Systems ==="; \
	if ! which nsys >/dev/null 2>&1; then \
		echo "nsys not found. Install: apt-get install nsight-systems"; \
		exit 1; \
	fi; \
	nsys --version 2>&1 | head -1; \
	case "$$(nsys --version 2>&1)" in \
		*2021*) echo "WARNING: nsys 2021 has GLIBC compat issue on Ubuntu 22.04. Upgrade to 2023+.";; \
	esac; \
	echo "---"
	nsys profile -o $(PROFILE_NSYS) -t nvtx,cuda,osrt \
		--cuda-memory-usage true \
		--show-output true \
		--force-overwrite true \
		$(UV) python -m home_seek.profiling_runner \
			--rounds 2 --prompts "Hello" "Hello" --max-tokens 5 --temperature 0
	@if [ -f "$(PROFILE_NSYS).qdstrm" ]; then \
		echo "=== NSYS trace saved: $(PROFILE_NSYS).qdstrm ==="; \
		echo "View with: nsys-ui $(PROFILE_NSYS).qdstrm"; \
	elif [ -f "$(PROFILE_NSYS).nsys-rep" ]; then \
		echo "=== NSYS trace saved: $(PROFILE_NSYS).nsys-rep ==="; \
	else \
		echo "=== WARNING: nsys trace not produced ==="; \
		echo "Check GLIBC compatibility (see .agent_memory.md)"; \
	fi

# ── Nsight Compute (单个 kernel 深度分析) ───────────────────────────
PROFILE_NCU := $(PROFILE_DIR)/ncu_kernel

profile-ncu: $(PROFILE_DIR)
	@echo "=== Checking Nsight Compute ==="; \
	if ! which ncu >/dev/null 2>&1; then \
		echo "ncu not found. Install: apt-get install nsight-compute"; \
		exit 1; \
	fi; \
	ncu --version 2>&1 | head -1; \
	echo "---"
	ncu --set full -o $(PROFILE_NCU) -f \
		--kernel-name "regex:_triton_dequantize_fp4_kernel|_triton_fused_down_kernel|triton_fused_gate_up_kernel" \
		--launch-count 10 \
		$(UV) python -m home_seek.profiling_runner \
			--rounds 1 --prompt "Hello" --max-tokens 2 --temperature 0 2>&1; \
	EXIT_CODE=$$?; \
	if [ -f "$(PROFILE_NCU).ncu-rep" ]; then \
		echo "=== NCU trace saved: $(PROFILE_NCU).ncu-rep ==="; \
		echo "View with: ncu-ui $(PROFILE_NCU).ncu-rep"; \
	elif echo "$$EXIT_CODE" | grep -q "ERR_NVGPUCTRPERM" 2>/dev/null; then \
		echo "=== NCU: GPU perf counters restricted (ERR_NVGPUCTRPERM) ==="; \
		echo "This is a cloud/host-level restriction, cannot fix in container."; \
	else \
		echo "=== WARNING: ncu did not produce trace ==="; \
	fi

# ── Chrome trace (兼容性最好, 云平台可用) ─────────────────────────
profile-chrome: $(PROFILE_DIR)
	$(UV) python -m home_seek.profiling_runner \
		--rounds 2 --prompts "Hello" "Hello" --max-tokens 5 --temperature 0 \
		--profiler chrome --profiler-warmup 1 \
		--output $(PROFILE_DIR)/chrome_profile.json
	@ls -lh $(PROFILE_DIR)/chrome_profile.json; \
	for f in profile_round*.json; do if [ -f "$$f" ]; then ls -lh "$$f"; fi; done

# ── 列出所有 CUDA kernel 名 (辅助 profile-ncu) ──────────────────
profile-ncu-list: $(PROFILE_DIR)
	ncu --set full --replay-mode kernel \
		$(UV) python -m home_seek.profiling_runner \
			--rounds 1 --prompt "Hello" --max-tokens 2 --temperature 0 2>&1 | \
		grep -E "^[0-9]+\." | sed 's/^[0-9]*\. //' | sort

smoke:
	$(UV) python -m home_seek.profiling_runner \
		--rounds 5 --prompts "Hello" "What is AI?" "Write a poem" "How are you?" "Hi" --max-tokens 30 --temperature 0

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
