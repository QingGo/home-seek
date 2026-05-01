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
		--rounds 5 --prompts "Hello" "What is AI?" "Write a poem" "How are you?" "Hi" --max-tokens 30 --temperature 0 \
		--output $(LAST_PROFILE)
	@$(UV) python3 scripts/profile_show.py $(LAST_PROFILE)

profile-compare: $(PROFILE_DIR)
	@if [ -f $(PREV_PROFILE) ]; then $(UV) python3 scripts/profile_compare.py $(PREV_PROFILE) $(LAST_PROFILE); else echo "No previous profile at $(PREV_PROFILE). Run make profile twice."; fi

# ── Nsight Systems (stream 级 profiler, 看 CUDA stream 并行性) ─────────
PROFILE_NSYS := $(PROFILE_DIR)/nsys_trace

profile-nsys: $(PROFILE_DIR)
	nsys profile -o $(PROFILE_NSYS) -t nvtx,cuda,osrt \
		--gpu-metrics-devices=0 \
		--cuda-memory-usage true \
		--show-output true \
		--force-overwrite true \
		$(UV) python -m home_seek.profiling_runner \
			--rounds 2 --prompts "Hello" "Hello" --max-tokens 5 --temperature 0
	@echo "=== NSYS trace saved: $(PROFILE_NSYS).nsys-rep ==="
	@echo "View with: nsys-ui $(PROFILE_NSYS).nsys-rep"

# ── Nsight Compute (单个 kernel 深度分析) ───────────────────────────
PROFILE_NCU := $(PROFILE_DIR)/ncu_kernel

profile-ncu: $(PROFILE_DIR)
	ncu --set full -o $(PROFILE_NCU) --target-processes all --replay-mode application \
		--kernel-name "triton_fused_gate_up_kernel|_triton_fused_down_kernel|_triton_dequantize_fp4" \
		--launch-count 10 \
		$(UV) python -m home_seek.profiling_runner \
			--rounds 1 --prompt "Hello" --max-tokens 2 --temperature 0
	@echo "=== NCU trace saved: $(PROFILE_NCU).ncu-rep ==="
	@echo "View with: ncu-ui $(PROFILE_NCU).ncu-rep"

# ── Nsys + Profiler (双工具: Chrome trace + stream timeline) ─────────
profile-deep: $(PROFILE_DIR)
	nsys profile -o $(PROFILE_DIR)/deep_trace -t nvtx,cuda,osrt \
		--gpu-metrics-devices=0 --cuda-memory-usage true \
		--show-output true --force-overwrite true \
		$(UV) python -m home_seek.profiling_runner \
			--rounds 2 --prompts "Hello" "Hello" --max-tokens 5 --temperature 0 \
			--profiler chrome --profiler-warmup 1 --output $(PROFILE_DIR)/chrome_profile.json
	@echo "=== Deep profile saved ==="
	@ls -lh $(PROFILE_DIR)/deep_trace.nsys-rep $(PROFILE_DIR)/chrome_profile.json 2>/dev/null
	@echo "View nsys: nsys-ui $(PROFILE_DIR)/deep_trace.nsys-rep"

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
