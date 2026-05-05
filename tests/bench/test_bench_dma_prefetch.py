"""DMA prefetch feasibility benchmark.

Measures:
  Q1: Adjacent-layer expert overlap rate (per-token Jaccard) — the key unknown
  Q2: DMA time per token + overlap window size
  Q3: Amdahl projection with real data

Run independently (each test creates and destroys its own engine):
    pytest tests/bench/test_bench_dma_prefetch.py -m bench -v -s
"""

import gc
import time
import pytest
import torch

from home_seek.inference_engine import HomeSeekInferenceEngine


def _cleanup_engine(eng):
    """Aggressively free GPU memory held by an engine."""
    eng._layer_weight_cache.clear()
    eng._gpu_hot_experts.clear()
    eng._gpu_bf16_cache.clear()
    eng._gpu_bf16_deq_cache.clear()
    eng._compressors.clear()
    eng._indexers.clear()
    eng._hybrid_kv.clear()
    eng.layer_states.clear()
    for c in getattr(eng, '_expert_caches', {}).values():
        c.clear()
    if hasattr(eng, 'expert_cache'):
        eng.expert_cache.clear()
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ── Q1: Routing overlap ─────────────────────────────────────────────

def _run_routing_collection(prompts, max_tokens=12):
    """Run engine with routing monkey-patch and return per-token per-layer eids.

    Returns:
        routes: dict[layer_idx] → list of per-token eid lists
        total_generated: int
    """
    eng = HomeSeekInferenceEngine("weights", verbose=False)
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file="weights/tokenizer.json")

    routes: dict[int, list[list[int]]] = {}
    original_ffn = eng._forward_ffn

    def collecting_ffn(hidden_states, lw, layer_idx, input_ids=None):
        result, used = original_ffn(hidden_states, lw, layer_idx, input_ids)
        eids = getattr(eng, '_last_routed_eids', [])
        if layer_idx not in routes:
            routes[layer_idx] = []
        routes[layer_idx].append([e for e in eids if e >= 0])
        return result, used

    eng._forward_ffn = collecting_ffn

    total_generated = 0
    try:
        for prompt_text in prompts:
            input_ids = tok.encode(prompt_text)
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            generated = []
            list(eng.generate(input_ids, max_new_tokens=max_tokens, temperature=0,
                              stream_callback=lambda tid: generated.append(tid)))
            total_generated += len(generated)
    finally:
        eng._forward_ffn = original_ffn
        _cleanup_engine(eng)

    return routes, total_generated


# ── Q2/Q3: DMA timing ───────────────────────────────────────────────

def _run_dma_timing(prompt_text="How are you?", max_tokens=15):
    """Run engine with timing monkey-patches. Returns timing stats."""
    eng = HomeSeekInferenceEngine("weights", verbose=False)
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file="weights/tokenizer.json")

    original_ffn = eng._forward_ffn
    original_m1 = eng._forward_ffn_m1_triton

    dma_times_ms: list[float] = []       # M1 total per call
    overlap_windows_ms: list[float] = []  # routing_prev → m1_start
    layer_gaps_ms: list[float] = []       # routing[N] → routing[N+1]
    last_routing_t_s: float | None = None
    last_layer_idx: int = -1

    def timing_ffn(hidden_states, lw, layer_idx, input_ids=None):
        nonlocal last_routing_t_s, last_layer_idx

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result, used = original_ffn(hidden_states, lw, layer_idx, input_ids)
        torch.cuda.synchronize()

        if last_routing_t_s is not None and layer_idx > 0:
            layer_gaps_ms.append((t0 - last_routing_t_s) * 1000)
        last_routing_t_s = t0
        last_layer_idx = layer_idx
        return result, used

    def timing_m1(hidden_states, flat_hidden, flat_topk_idx, flat_topk_w, layer_idx):
        nonlocal last_routing_t_s

        if last_routing_t_s is not None:
            overlap_windows_ms.append((time.perf_counter() - last_routing_t_s) * 1000)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_m1(hidden_states, flat_hidden, flat_topk_idx, flat_topk_w, layer_idx)
        torch.cuda.synchronize()
        dma_times_ms.append((time.perf_counter() - t0) * 1000)
        return result

    eng._forward_ffn = timing_ffn
    eng._forward_ffn_m1_triton = timing_m1

    try:
        input_ids = tok.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        # Warm up page cache + GPU caches with 1-token prompt to avoid
        # legacy batched path OOM (T>1 prefill allocates large weight concat).
        short_text = "Hi"
        short_ids = tok.encode(short_text)
        short_ids = torch.tensor([short_ids], dtype=torch.long)
        list(eng.generate(short_ids, max_new_tokens=2, temperature=0))

        # Clear accumulated stats from warmup
        dma_times_ms.clear()
        overlap_windows_ms.clear()
        layer_gaps_ms.clear()

        # Measurement pass
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        generated = list(eng.generate(input_ids, max_new_tokens=max_tokens, temperature=0))
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - wall_t0
        n_tokens = len(generated)
    finally:
        eng._forward_ffn = original_ffn
        eng._forward_ffn_m1_triton = original_m1
        _cleanup_engine(eng)

    return {
        "n_tokens": n_tokens,
        "wall_s": wall_s,
        "dma_times_ms": dma_times_ms,
        "overlap_windows_ms": overlap_windows_ms,
        "layer_gaps_ms": layer_gaps_ms,
    }


# ── Benchmark tests ─────────────────────────────────────────────────

@pytest.mark.bench
class TestDmaPrefetchFeasibility:

    @pytest.fixture(autouse=True)
    def _require_weights(self, require_weights):
        pass

    # ── Q1 ──────────────────────────────────────────────────────────

    def test_q1_adjacent_routing_overlap(self):
        """Q1: Per-token expert overlap between adjacent layers.

        For each token t, compare layer N with layer N+1:
            overlap = |intersection(t)| / |union(t)|

        This is the expected hit rate for the same-as-current heuristic.
        Also compute: what if we predict ALL experts from layer N (not just
        the routed subset)? i.e., the set of all experts that ever appeared
        in layer N's routing across all tokens.
        """
        print("\n─── Q1: Per-token adjacent-layer routing overlap ───")

        prompts = ["How are you?", "What is AI?", "Write a poem", "Hello", "Hi"]
        routes, total_generated = _run_routing_collection(prompts, max_tokens=12)
        layers = sorted(routes.keys())
        n_observations = len(routes.get(layers[0], []))

        print(f"\nCollected routing for {len(layers)} layers × "
              f"{n_observations} tokens × {len(prompts)} prompts "
              f"(total {total_generated} generated tokens)")

        # ── Per-token overlap ──
        per_token_rates: list[float] = []
        for t in range(n_observations):
            for i in range(len(layers) - 1):
                l_curr, l_next = layers[i], layers[i + 1]
                if t >= len(routes.get(l_curr, [])) or t >= len(routes.get(l_next, [])):
                    continue
                s_curr = set(routes[l_curr][t])
                s_next = set(routes[l_next][t])
                if s_curr and s_next:
                    intersection = s_curr & s_next
                    union = s_curr | s_next
                    per_token_rates.append(len(intersection) / len(union))

        avg_pt = sum(per_token_rates) / len(per_token_rates) if per_token_rates else 0
        min_pt = min(per_token_rates) if per_token_rates else 0
        max_pt = max(per_token_rates) if per_token_rates else 0

        print(f"\nPer-token overlap (n={len(per_token_rates)} layer-pairs × tokens):")
        print(f"  mean: {avg_pt:.1%}")
        print(f"  min:  {min_pt:.1%}")
        print(f"  max:  {max_pt:.1%}")

        # ── Same-as-current heuristic hit rate ──
        # For prefetch: we predict layer N+1 will use the SAME experts as layer N.
        # Hit = |prediction ∩ actual| / |actual|
        accurate_predictions = 0
        total_predictions = 0
        for t in range(n_observations):
            for i in range(len(layers) - 1):
                l_curr, l_next = layers[i], layers[i + 1]
                if t >= len(routes.get(l_curr, [])) or t >= len(routes.get(l_next, [])):
                    continue
                predicted = routes[l_curr][t]
                actual = routes[l_next][t]
                if actual:
                    hits = len(set(predicted) & set(actual))
                    accurate_predictions += hits
                    total_predictions += len(actual)

        hit_rate = accurate_predictions / total_predictions if total_predictions else 0
        print(f"\nSame-as-current hit rate: {hit_rate:.1%} "
              f"({accurate_predictions}/{total_predictions})")

        # ── Global overlap (for reference) ──
        global_rates = []
        for i in range(len(layers) - 1):
            s_curr, s_next = set(), set()
            for obs in routes.get(layers[i], []):
                s_curr.update(obs)
            for obs in routes.get(layers[i + 1], []):
                s_next.update(obs)
            if s_curr and s_next:
                global_rates.append(len(s_curr & s_next) / len(s_curr | s_next))
        avg_global = sum(global_rates) / len(global_rates) if global_rates else 0
        print(f"\nGlobal overlap (reference, n={len(global_rates)} layer pairs):")
        print(f"  mean: {avg_global:.1%}")

        print("\n─── Interpretation ───")
        if hit_rate > 0.5:
            print(f"  ⭐ Hit rate {hit_rate:.0%}: prefetch viable, expected +{hit_rate*100:.0f}% DMA hidden")
        elif hit_rate > 0.25:
            print(f"  ⚠️  Hit rate {hit_rate:.0%}: marginal, only {hit_rate*100:.0f}% DMA can be hidden")
        else:
            print(f"  ✗ Hit rate {hit_rate:.0%}: same-as-current insufficient")
            print("  Need alternative prediction or multi-token lookahead")

        assert len(per_token_rates) > 0

    # ── Q2 ──────────────────────────────────────────────────────────

    def test_q2_dma_timing(self):
        """Q2: Measure DMA time per token and overlap window.

        Run a single prompt with timing patches. Reports:
          - DMA (M1 load+kernel) per token
          - Overlap window: time from routing_done[N] to dma_start[N+1]
          - Can DMA fit in window?
        """
        print("\n─── Q2: DMA timing and overlap window ───")
        stats = _run_dma_timing("How are you?", max_tokens=20)

        dma = stats["dma_times_ms"]
        windows = stats["overlap_windows_ms"]
        gaps = stats["layer_gaps_ms"]
        n_tok = stats["n_tokens"]

        if not dma:
            pytest.skip("No M1 path calls (all hot-batched?)")

        total_dma = sum(dma)
        avg_dma = total_dma / len(dma)
        dma_per_tok = total_dma / n_tok if n_tok else 0
        avg_win = sum(windows) / len(windows) if windows else 0
        avg_gap = sum(gaps) / len(gaps) if gaps else 0

        print(f"\nM1 calls: {len(dma)} across {n_tok} tokens")
        print(f"  avg M1 (load+kernel): {avg_dma:.1f} ms/call")
        print(f"  DMA per token:        {dma_per_tok:.1f} ms")
        print(f"  inter-layer gap:      {avg_gap:.1f} ms (routing[N]→routing[N+1])")
        print(f"  overlap window:       {avg_win:.1f} ms (routing[N]→dma_start[N+1])")
        print(f"  wall time:            {stats['wall_s']:.1f}s  "
              f"({n_tok/stats['wall_s']:.2f} t/s)")

        can = avg_win > avg_dma
        print(f"\n  DMA fits in window? {'✓ YES' if can else '✗ NO'}")
        if can:
            coverage = min(1.0, avg_win / avg_dma)
            print(f"  {avg_dma:.0f}ms DMA in {avg_win:.0f}ms window → "
                  f"{coverage*100:.0f}% coverage per call")
        else:
            print(f"  Window ({avg_win:.0f}ms) too small vs DMA ({avg_dma:.0f}ms)")

        assert len(dma) > 0

    # ── Q3 ──────────────────────────────────────────────────────────

    def test_q3_amdahl_projection(self):
        """Q3: Full Amdahl projection combining Q1+Q2 data.

        Runs routing collection + timing in a single pass and prints projected
        throughput improvement at various overlap rates.
        """
        print("\n─── Q3: Amdahl projection ───")

        # ── Collect routing data ──
        routes, total_gen = _run_routing_collection(
            ["How are you?", "What is AI?", "Hello"], max_tokens=12)

        layers = sorted(routes.keys())
        n_obs = len(routes.get(layers[0], []))

        # Per-token same-as-current hit rate
        hits, total = 0, 0
        for t in range(n_obs):
            for i in range(len(layers) - 1):
                lc, ln = layers[i], layers[i + 1]
                if t >= len(routes.get(lc, [])) or t >= len(routes.get(ln, [])):
                    continue
                actual = routes[ln][t]
                if actual:
                    hits += len(set(routes[lc][t]) & set(actual))
                    total += len(actual)
        p_hit = hits / total if total else 0

        # ── Collect timing data ──
        stats = _run_dma_timing("How are you?", max_tokens=15)
        dma = stats["dma_times_ms"]
        windows = stats["overlap_windows_ms"]
        n_tok = stats["n_tokens"]
        wall_s = stats["wall_s"]

        if not dma or n_tok <= 0:
            pytest.skip("Insufficient timing data")

        dma_per_tok = sum(dma) / n_tok
        wall_per_tok = wall_s * 1000 / n_tok
        avg_win = sum(windows) / len(windows) if windows else 0
        avg_dma = sum(dma) / len(dma)

        print("\nMeasured baseline:")
        print(f"  per-tok:         {wall_per_tok:.0f}ms  ({1000/wall_per_tok:.2f} t/s)")
        print(f"  DMA/tok:         {dma_per_tok:.0f}ms  ({dma_per_tok/wall_per_tok*100:.0f}%)")
        print(f"  DMA/call:        {avg_dma:.1f}ms")
        print(f"  overlap window:  {avg_win:.1f}ms")
        print(f"  routing hit P:   {p_hit:.0%}")

        # Window-constrained effective hit rate
        window_coverage = min(1.0, avg_win / avg_dma) if avg_dma > 0 else 0
        effective_p = p_hit * window_coverage

        print("\nOverlap feasibility:")
        print(f"  DMA fits in window: {avg_win/avg_dma*100:.0f}% coverage" if avg_dma > 0
              else "  no DMA data")
        print(f"  effective P:        {effective_p:.0%}")

        dma_fraction = dma_per_tok / wall_per_tok if wall_per_tok > 0 else 0
        print(f"\nAmdahl ceiling (DMA fraction = {dma_fraction:.1%}):")
        print(f"  DMA fully hidden → {(1000/(wall_per_tok - dma_per_tok)):>.2f} t/s  "
              f"({(1000/(wall_per_tok - dma_per_tok))/(1000/wall_per_tok)*100-100:+.0f}%)")

        print("\nProjection by scenario:")
        print(f"  {'P (hit)':<10} {'new t/s':>8} {'improve':>10} {'new ms':>10}")
        print(f"  {'-'*40}")
        for sp in [0.2, 0.4, 0.6, 0.8, 1.0]:
            saved = dma_per_tok * sp * window_coverage
            new_ms = wall_per_tok - saved
            new_ts = 1000 / new_ms
            improve = (new_ts / (1000 / wall_per_tok) - 1) * 100
            marker = " ← actual P" if abs(sp - p_hit) < 0.05 else ""
            print(f"  {sp:<10.0%} {new_ts:>8.2f} {improve:>+9.1f}% {new_ms:>10.0f}{marker}")

        # Project with actual P
        saved = dma_per_tok * effective_p
        actual_new_ms = wall_per_tok - saved
        actual_new_ts = 1000 / actual_new_ms
        actual_improve = (actual_new_ts / (1000 / wall_per_tok) - 1) * 100

        print(f"\nWith effective P={effective_p:.0%} (P={p_hit:.0%} × "
              f"window={window_coverage:.0%}):")
        print(f"  projected: {actual_new_ts:.2f} t/s  ({actual_improve:+.1f}%)")

        if effective_p < 0.15:
            print("\n  ✗ Effective P < 15% — prefetch not worthwhile.")
        elif effective_p < 0.30:
            print("\n  ⚠️  Effective P 15-30% — borderline. Consider alternative predictors.")
        else:
            print("\n  ✓ Effective P > 30% — DMA prefetch worthwhile.")

        assert wall_per_tok > 0
