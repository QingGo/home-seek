"""Temporal (cross-step) routing persistence + DMA window benchmark.

Measures:
  Q1: Per-layer per-token routing eids across decode steps
  Q2: Cross-step routing overlap (step S → S+1) — temporal, not cross-layer!
  Q3: "Other" window size (layer_loop_end → next step embed)
  Q4: Expert load DMA time per call
  Q5: Amdahl projection for temporal DMA prefetch

Key difference from test_bench_dma_prefetch.py: that test measured CROSS-LAYER
routing overlap (layer N→N+1). This test measures CROSS-STEP (step S→S+1),
which V21.19 found has Pearson=0.718 and Top-12 hit=55.9%.

Run:
    pytest tests/bench/test_bench_temporal_prefetch.py -m bench -v -s
"""

import gc
import time
import pytest
import torch


def _cleanup_engine(eng):
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


# ── Q1+Q2: Temporal routing collection ───────────────────────────────

def _run_temporal_routing(prompts, max_tokens=15):
    """Collect per-layer routing eids for each decode step.

    Returns:
        routes: dict[layer_idx] → list[list[int]]  (one per decode step)
        total_gen: int
    """
    from home_seek.inference_engine import HomeSeekInferenceEngine
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

    total_gen = 0
    try:
        for prompt_text in prompts:
            input_ids = tok.encode(prompt_text)
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            generated = []
            list(eng.generate(input_ids, max_new_tokens=max_tokens, temperature=0,
                              stream_callback=lambda tid: generated.append(tid)))
            total_gen += len(generated)
    finally:
        eng._forward_ffn = original_ffn
        _cleanup_engine(eng)

    return routes, total_gen


# ── Q3+Q4: Wall-clock timing of decode structure ──────────────────────

def _run_decode_timing(prompts=None, max_tokens=15):
    """Measure decode loop structure with wall-clock (no profiling tax).

    Monkey-patches _forward_layer to track:
      - layer_loop_end → next step embed (the "Other" window)
      - per-layer timestamps to reconstruct layer execution order

    Also monkey-patches _forward_ffn_m1_triton to measure DMA time.

    Returns dict with timing stats.
    """
    if prompts is None:
        prompts = ["How are you?"]

    from home_seek.inference_engine import HomeSeekInferenceEngine
    eng = HomeSeekInferenceEngine("weights", verbose=False)
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file="weights/tokenizer.json")

    # ── Timing accumulators ──
    other_window_ms: list[float] = []     # layer loop end → next step embed
    m1_dma_ms: list[float] = []           # expert load time
    m1_kernel_ms: list[float] = []        # kernel execution time
    layer_times_ms: list[float] = []      # full layer time
    ffn_times_ms: list[float] = []        # FFN only
    attn_mhc_times_ms: list[float] = []   # Attention+MHC only

    # ── Track step boundaries via _forward_layer ──
    last_layer_end: float | None = None
    cur_step_layers: int = 0

    original_layer = eng._forward_layer
    original_ffn = eng._forward_ffn
    original_m1 = eng._forward_ffn_m1_triton

    def traced_layer(h, lw, layer_idx, input_ids):
        nonlocal last_layer_end, cur_step_layers

        # h is 4D: [B, T, hc_mult, D]; decode has T=1
        T_dim = h.shape[1] if h.dim() >= 2 else 0
        is_decode = (T_dim == 1)

        if is_decode and layer_idx == 0:
            cur_step_layers = 0
            if last_layer_end is not None:
                torch.cuda.synchronize()
                gap = (time.perf_counter() - last_layer_end) * 1000
                other_window_ms.append(gap)

        # Measure full layer time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_layer(h, lw, layer_idx, input_ids)
        torch.cuda.synchronize()

        if is_decode:
            dt = (time.perf_counter() - t0) * 1000
            layer_times_ms.append(dt)
            cur_step_layers += 1

            n_layers = getattr(eng.config, 'num_hidden_layers', 43)
            if layer_idx == n_layers - 1:
                last_layer_end = time.perf_counter()

        return result

    def traced_ffn(hidden_states, lw, layer_idx, input_ids=None):
        T_dim = hidden_states.shape[1] if hidden_states.dim() >= 2 else 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result, used = original_ffn(hidden_states, lw, layer_idx, input_ids)
        torch.cuda.synchronize()
        if T_dim == 1:
            ffn_times_ms.append((time.perf_counter() - t0) * 1000)
        return result, used

    def traced_m1(hidden_states, flat_hidden, flat_topk_idx, flat_topk_w, layer_idx):
        # Measure load (DMA) phase
        t_load0 = time.perf_counter()
        eids = {}
        idx_row = flat_topk_idx[0].tolist()
        w_row = flat_topk_w[0].tolist()
        for k in range(flat_topk_idx.shape[1]):
            eid = int(idx_row[k])
            if eid >= 0:
                eids[eid] = eids.get(eid, 0.0) + float(w_row[k])

        fp4_data = {}
        for eid in eids:
            raw = eng._load_expert_fp4_raw(layer_idx, eid)
            if raw is None:
                return None
            fp4_data[eid] = raw
        t_load1 = time.perf_counter()

        # Measure kernel phase
        t_k0 = time.perf_counter()
        try:
            result = eng._fused_moe.forward_m1_fp4(flat_hidden, fp4_data, eids)
        except Exception:
            return None

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        m1_dma_ms.append((t_load1 - t_load0) * 1000)
        m1_kernel_ms.append((t1 - t_k0) * 1000)

        return result.reshape(hidden_states.shape)

    eng._forward_layer = traced_layer
    eng._forward_ffn = traced_ffn
    eng._forward_ffn_m1_triton = traced_m1

    total_generated = 0
    try:
        for prompt_text in prompts:
            input_ids = tok.encode(prompt_text)
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            generated = []

            # Prefill prep
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            list(eng.generate(input_ids, max_new_tokens=max_tokens, temperature=0,
                              stream_callback=lambda tid: generated.append(tid)))

            torch.cuda.synchronize()
            wall_s = time.perf_counter() - t_start
            n_gen = len(generated)
            total_generated += n_gen
    finally:
        eng._forward_ffn = original_ffn
        eng._forward_layer = original_layer
        eng._forward_ffn_m1_triton = original_m1
        _cleanup_engine(eng)

    import statistics
    results = {
        "n_tokens": total_generated,
        "wall_s": wall_s,
    }
    if other_window_ms:
        results["other_window_mean_ms"] = statistics.mean(other_window_ms)
        results["other_window_median_ms"] = statistics.median(other_window_ms)
        results["other_window_all"] = other_window_ms
    if m1_dma_ms:
        results["m1_dma_mean_ms"] = statistics.mean(m1_dma_ms)
        results["m1_dma_total_ms"] = sum(m1_dma_ms)
    if m1_kernel_ms:
        results["m1_kernel_mean_ms"] = statistics.mean(m1_kernel_ms)
        results["m1_kernel_total_ms"] = sum(m1_kernel_ms)
    if layer_times_ms:
        results["layer_mean_ms"] = statistics.mean(layer_times_ms)
        results["layer_total_ms"] = sum(layer_times_ms)
    if ffn_times_ms:
        results["ffn_mean_ms"] = statistics.mean(ffn_times_ms)
    if attn_mhc_times_ms:
        results["attn_mhc_mean_ms"] = statistics.mean(attn_mhc_times_ms)
    return results


# ── Benchmark tests ───────────────────────────────────────────────────

@pytest.mark.bench
class TestTemporalPrefetch:
    """Temporal (cross-step) routing persistence + DMA prefetch feasibility."""

    @pytest.fixture(autouse=True)
    def _require_weights(self, require_weights):
        pass

    def test_q1_temporal_routing_overlap(self):
        """Q1: Cross-step routing overlap — step S → step S+1, same layer.

        For each layer, compare the actual routed experts at steps S and S+1.
        Reports: Jaccard, hit rate (if we prefetch step S's eids for S+1).

        This is the key metric V21.19 identified: Pearson=0.718, Top-12=55.9%.
        Here we verify with actual routing eids (not gate scores).
        """
        print("\n─── Q1: Temporal (step S→S+1) routing overlap ───")

        prompts = ["How are you?", "What is AI?", "Write a poem", "Hello", "Hi"]
        routes, total_gen = _run_temporal_routing(prompts, max_tokens=15)

        layers = sorted(routes.keys())
        max_steps = max(len(routes[l]) for l in layers if routes[l])

        print(f"\nCollected: {len(layers)} layers, up to {max_steps} decode steps/layer")
        print(f"Total generated: {total_gen} tokens")

        # ── Full eid list overlap (step S → S+1) ──
        all_hits_same = 0
        all_total = 0
        step_pairs = 0

        for li in layers:
            steps = routes[li]
            for s in range(len(steps) - 1):
                eids_s = set(steps[s])
                eids_s1 = set(steps[s + 1])
                if eids_s1:
                    all_hits_same += len(eids_s & eids_s1)
                    all_total += len(eids_s1)
                    step_pairs += 1

        hit_rate = all_hits_same / all_total if all_total else 0

        # ── Jaccard per step pair ──
        jaccards = []
        for li in layers:
            steps = routes[li]
            for s in range(len(steps) - 1):
                eids_s = set(steps[s])
                eids_s1 = set(steps[s + 1])
                union = eids_s | eids_s1
                if union:
                    jaccards.append(len(eids_s & eids_s1) / len(union))

        import statistics
        jac_mean = statistics.mean(jaccards) if jaccards else 0

        print("\nTemporal (step S→S+1) routing overlap:")
        print(f"  Step pairs analyzed:  {step_pairs}")
        print(f"  Same-exact-eids hit:  {hit_rate:.1%}  ({all_hits_same}/{all_total})")
        print(f"  Mean Jaccard:         {jac_mean:.1%}")
        print(f"  Jaccard range:        {min(jaccards):.1%} – {max(jaccards):.1%}" if jaccards else "")

        # ── Top-K prefetch hit rates ──
        # What if we prefetch more than just the exact 6?
        print("\nTop-K prefetch hit rates (prefetch step S top-K → step S+1 top-6):")
        print(f"  {'K':>6} {'hit rate':>10} {'hits/6':>10} {'cache MB':>10}")
        print(f"  {'─'*42}")
        for k in [6, 12, 24, 48, 96]:
            k_hits = 0
            k_total = 0
            for li in layers:
                steps = routes[li]
                for s in range(len(steps) - 1):
                    eids_s = set(steps[s])
                    eids_s1 = set(steps[s + 1])
                    if eids_s1:
                        # Take top-K from step S (by index, since all have ~same weight)
                        # In practice we'd use routing weights, but eid set is same
                        pred = set(list(steps[s])[:k])  # just first k of the 6
                        k_hits += len(pred & eids_s1)
                        k_total += len(eids_s1)
            rate = k_hits / k_total if k_total else 0
            mb = k * 12.75
            predicted = 6  # we can only prefetch 6 experts max (that's what was routed)
            print(f"  {k:>6} {rate:>9.1%}  {rate * predicted:>10.2f}  {mb:>10.0f}")

        # ── Interpretation ──
        print("\n─── Interpretation ───")
        if hit_rate > 0.50:
            print(f"  ✓ Hit rate {hit_rate:.0%}: strong temporal persistence")
            print(f"    → DMA prefetch viable! Expected {hit_rate*244:.0f}ms DMA saved")
        elif hit_rate > 0.30:
            print(f"  ⚠ Hit rate {hit_rate:.0%}: moderate temporal persistence")
            print(f"    → DMA prefetch marginally viable, ~{hit_rate*244:.0f}ms saved")
        else:
            print(f"  ✗ Hit rate {hit_rate:.0%}: weak temporal persistence")
            print("    → Even full reuse barely helps")

        assert step_pairs > 0

    def test_q2_decode_timing_wallclock(self):
        """Q2: Wall-clock decode timing with NO profiling tax.

        Measures:
          - "Other" window: time from layer_loop_end to next_step_embed_start
          - Layer time, FFN time, M1 DMA time
          - This is the real window where async DMA prefetch can run
        """
        print("\n─── Q2: Wall-clock decode timing (no profiling tax) ───")

        stats = _run_decode_timing(["How are you?"], max_tokens=20)

        n_tok = stats["n_tokens"]
        wall_s = stats["wall_s"]

        print(f"\nGenerated {n_tok} tokens in {wall_s:.1f}s  "
              f"({n_tok/wall_s:.2f} t/s)")

        ow = stats.get("other_window_mean_ms", 0)
        ow_med = stats.get("other_window_median_ms", 0)
        ow_all = stats.get("other_window_all", [])

        print("\n  Other window (layer_end → next embed_start):")
        print(f"    mean:   {ow:.0f} ms")
        print(f"    median: {ow_med:.0f} ms")
        if ow_all:
            print(f"    range:  {min(ow_all):.0f} – {max(ow_all):.0f} ms")
            print(f"    values: {[f'{v:.0f}' for v in ow_all]}")

        dma_mean = stats.get("m1_dma_mean_ms", 0)
        kernel_mean = stats.get("m1_kernel_mean_ms", 0)
        layer_mean = stats.get("layer_mean_ms", 0)
        ffn_mean = stats.get("ffn_mean_ms", 0)

        print("\n  Per-call breakdown:")
        print(f"    layer:   {layer_mean:.0f} ms")
        print(f"    FFN:     {ffn_mean:.0f} ms")
        print(f"    M1 DMA:  {dma_mean:.0f} ms  (expert loading)")
        print(f"    M1 kern: {kernel_mean:.0f} ms  (Triton gate_up+down)")

        # Per-token projection
        n_layers = 43
        per_tok_layer = layer_mean * n_layers
        per_tok_dma = dma_mean * n_layers
        per_tok_kernel = kernel_mean * n_layers
        per_tok_total = per_tok_layer + ow_med

        print(f"\n  Per-token projection (×{n_layers} layers):")
        print(f"    layers:       {per_tok_layer:.0f} ms")
        print(f"    DMA:          {per_tok_dma:.0f} ms  ({per_tok_dma / per_tok_total * 100:.0f}%)")
        print(f"    kernel:       {per_tok_kernel:.0f} ms")
        print(f"    other win:    {ow_med:.0f} ms")
        print("    ────────────────────")
        print(f"    total/tok:    {per_tok_total:.0f} ms  "
              f"({1000 / per_tok_total:.2f} t/s)")

        # DMA fits in Other window?
        dma_per_call = dma_mean
        can_fit = dma_per_call < ow_med
        print(f"\n  DMA ({dma_per_call:.0f}ms) fits in Other window "
              f"({ow_med:.0f}ms)? {'✓ YES' if can_fit else '✗ NO'}")
        if can_fit:
            n_concurrent = int(ow_med / dma_per_call) if dma_per_call > 0 else 0
            print(f"    Can prefetch {n_concurrent} experts concurrently")
        print(f"    Overlap ratio: {min(1.0, ow_med / dma_per_call) * 100:.0f}%")

        assert n_tok > 0

    def test_q3_amdahl_temporal_prefetch(self):
        """Q3: Amdahl projection for temporal DMA prefetch.

        Combines temporal routing overlap with DMA timing to project
        throughput improvement at various hit rates.

        Key insight from V21.19: gate scores have Pearson=0.718 between steps,
        meaning step S routing reliably predicts step S+1 routing.
        Top-12 hit rate ≈ 55.9%. DMA is 22% of decode (244ms out of 1130ms).
        """
        print("\n─── Q3: Amdahl projection for temporal DMA prefetch ───")

        # ── Get routing overlap data ──
        prompts = ["How are you?", "What is AI?", "Write a poem", "Hello", "Hi"]
        routes, total_gen = _run_temporal_routing(prompts, max_tokens=15)

        layers = sorted(routes.keys())

        # Compute temporal hit rates for different prefetch budgets
        hit_rates = {}
        for k in [6, 12, 24, 48]:
            k_hits = 0
            k_total = 0
            for li in layers:
                steps = routes[li]
                for s in range(len(steps) - 1):
                    eids_s = set(steps[s])
                    eids_s1 = set(steps[s + 1])
                    if eids_s1:
                        pred = set(list(steps[s])[:min(k, len(steps[s]))])
                        k_hits += len(pred & eids_s1)
                        k_total += len(eids_s1)
            hit_rates[k] = k_hits / k_total if k_total else 0

        # ── Get DMA timing ──
        stats = _run_decode_timing(["How are you?"], max_tokens=20)
        dma_per_call = stats.get("m1_dma_mean_ms", 5.6)
        layer_per_call = stats.get("layer_mean_ms", 26)
        ow_mean = stats.get("other_window_mean_ms", 200)

        # Compute DMA fraction
        n_layers = 43
        total_dma_per_tok = dma_per_call * n_layers
        total_layer_per_tok = layer_per_call * n_layers
        total_per_tok = total_layer_per_tok + ow_mean
        dma_fraction = total_dma_per_tok / total_per_tok

        print(f"\n  Baseline per-token:  {total_per_tok:.0f}ms  ({1000/total_per_tok:.2f} t/s)")
        print(f"  DMA/tok:             {total_dma_per_tok:.0f}ms  ({dma_fraction:.1%})")
        print(f"  Other window:        {ow_mean:.0f}ms")
        print(f"  DMA/call:            {dma_per_call:.1f}ms")
        print(f"  Layer/call:          {layer_per_call:.1f}ms")

        # How many experts can fit in the Other window?
        max_prefetch_experts = int(ow_mean / dma_per_call) if dma_per_call > 0 else 0
        print(f"  Max concurrent DMA:  {max_prefetch_experts} experts in {ow_mean:.0f}ms window")

        # Window-constrained effective hit rate
        window_coverage = min(1.0, ow_mean / (dma_per_call * 6))
        print(f"  Window coverage:     {window_coverage:.0%}  "
              f"(can DMA {max_prefetch_experts} of 6 experts per prefetch)")

        print("\n  Temporal routing hit rates:")
        for k, r in hit_rates.items():
            prefetch_bytes = k * 12.75
            effective_r = r * window_coverage
            saved = total_dma_per_tok * effective_r
            new_ms = total_per_tok - saved
            new_ts = 1000 / new_ms
            improve = (new_ts / (1000 / total_per_tok) - 1) * 100
            marker = " ← prefetch budget" if k <= max_prefetch_experts else " ← exceeds window"
            print(f"    K={k:>3}: hit={r:.1%} → {new_ts:.2f} t/s ({improve:+.1f}%)  "
                  f"{prefetch_bytes:.0f} MB DMA{marker}")

        print("\n  === Projected improvement ===")
        for k, r in hit_rates.items():
            if k <= max_prefetch_experts + 6:
                effective_r = r * window_coverage
                saved = total_dma_per_tok * effective_r
                new_ms = total_per_tok - saved
                new_ts = 1000 / new_ms
                improve = (new_ts / (1000 / total_per_tok) - 1) * 100
                print(f"    Prefetch {k} experts → {new_ts:.2f} t/s ({improve:+.0f}%)  "
                      f"saved {saved:.0f}ms DMA")

        print(f"\n  Amdahl ceiling (DMA fully hidden): "
              f"{1000/(total_per_tok - total_dma_per_tok):.2f} t/s  "
              f"({(1000/(total_per_tok - total_dma_per_tok))/(1000/total_per_tok)*100-100:+.0f}%)")

        # ── Sanity: same-exact-eids temporal hit ──
        same_hits, same_total = 0, 0
        for li in layers:
            steps = routes[li]
            for s in range(len(steps) - 1):
                eids_s = set(steps[s])
                eids_s1 = set(steps[s + 1])
                if eids_s1:
                    same_hits += len(eids_s & eids_s1)
                    same_total += len(eids_s1)
        same_rate = same_hits / same_total if same_total else 0
        print(f"\n  Same-exact-eids temporal hit: {same_rate:.1%}")
        print("  (Prefetch step S's routed experts and check vs step S+1's routed)")

        assert total_per_tok > 0

    def test_q4_per_layer_temporal_variance(self):
        """Q4: How does temporal routing persistence vary across layers?

        Hash layers (0–2) use token-id-based routing, which may have different
        temporal characteristics than non-hash layers.
        """
        print("\n─── Q4: Per-layer temporal persistence ───")

        routes, _ = _run_temporal_routing(
            ["How are you?", "What is AI?", "Write a poem", "Hello", "Hi"],
            max_tokens=15)

        layers = sorted(routes.keys())

        from home_seek.model_config import DeepSeekV4FlashConfig
        import os
        config_path = os.path.join("weights", "config.json")
        cfg = DeepSeekV4FlashConfig.from_json(config_path)
        n_hash = cfg.num_hash_layers

        # ── Per-layer temporal hit rate ──
        print("\n  Layer temporal persistences (same-exact-eids hit):")
        print(f"  {'Layer':>6} {'hit':>8} {'Jaccard':>8} {'steps':>6}")
        print(f"  {'─'*35}")
        layer_stats = {}
        for li in layers:
            steps = routes[li]
            hits, total, jacs = 0, 0, []
            for s in range(len(steps) - 1):
                eids_s = set(steps[s])
                eids_s1 = set(steps[s + 1])
                if eids_s1:
                    hits += len(eids_s & eids_s1)
                    total += len(eids_s1)
                    union = eids_s | eids_s1
                    if union:
                        jacs.append(len(eids_s & eids_s1) / len(union))
            rate = hits / total if total else 0
            import statistics
            jac = statistics.mean(jacs) if jacs else 0
            tag = "hash" if li < n_hash else ""
            layer_stats[li] = {"rate": rate, "jac": jac, "tag": tag, "steps": len(steps)}
            print(f"  {li:>6} {rate:>7.1%} {jac:>7.1%} {len(steps):>6}  {tag}")

        # ── Aggregated by layer type ──
        hash_layers = [li for li in layers if li < n_hash]
        non_hash = [li for li in layers if li >= n_hash]

        if hash_layers:
            hash_rates = [layer_stats[l]["rate"] for l in hash_layers]
            hash_jacs = [layer_stats[l]["jac"] for l in hash_layers]
            print(f"\n  Hash layers (0–{n_hash-1}):")
            print(f"    hit rate: {statistics.mean(hash_rates):.1%}")
            print(f"    Jaccard:  {statistics.mean(hash_jacs):.1%}")
        if non_hash:
            nh_rates = [layer_stats[l]["rate"] for l in non_hash]
            nh_jacs = [layer_stats[l]["jac"] for l in non_hash]
            print(f"\n  Non-hash layers ({n_hash}–{layers[-1]}):")
            print(f"    hit rate: {statistics.mean(nh_rates):.1%}")
            print(f"    Jaccard:  {statistics.mean(nh_jacs):.1%}")
            print(f"    range:    {min(nh_rates):.1%} – {max(nh_rates):.1%}")

        # Key insight: if hash layers have low temporal persistence,
        # we should handle them separately (use token-id-based prediction)
        print("\n─── Interpretation ───")
        hash_mean = statistics.mean(hash_rates) if hash_rates else 0
        non_hash_mean = statistics.mean(nh_rates) if nh_rates else 0
        non_hash_mean_val = non_hash_mean
        if hash_layers:
            if hash_mean < non_hash_mean_val * 0.7 if non_hash_mean_val else False:
                print(f"  Hash layers ({hash_mean:.1%}) significantly less persistent")
                print(f"  → Prefetch non-hash layers primarily ({non_hash_mean_val:.1%})")
                print("  → Hash layers are <10% of total layers, low impact even if zero hit")
            else:
                print(f"  Hash layers ({hash_mean:.1%}) similar to non-hash ({non_hash_mean_val:.1%})")
                print("  → Uniform temporal prefetch strategy works")

        assert len(layer_stats) > 0
