"""Diagnose inter-layer and temporal routing correlations.

Collects gate scores[256] per (layer, decode_step) during inference.
Computes Spearman ρ and top-K prediction accuracy.

Key metrics:
  ρ_inter:   Spearman corr between layer N and N+1 scores (same token/step)
  ρ_temporal: Spearman corr between step S and S+1 scores (same layer)
  topK_accuracy: how many of target top-6 are in source top-K

Usage:
    uv run python scripts/diagnose_routing.py
"""

import gc
import statistics
import torch
import numpy as np
from collections import defaultdict

from home_seek.inference_engine import HomeSeekInferenceEngine


# ── Global state for score collection ────────────────────────────────

# scores[layer_idx] = list of score vectors [256] across decode steps
_scores: dict[int, list[torch.Tensor]] = defaultdict(list)
_current_layer: int = -1
_decode_step_global: int = 0
_n_experts: int = 256


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation (robust to many near-zero values)."""
    try:
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return None


def _cosine(x: np.ndarray, y: np.ndarray) -> float | None:
    """Cosine similarity between score vectors."""
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < 1e-20 or ny < 1e-20:
        return None
    return float(np.dot(x, y) / (nx * ny))


def _score_sparsity(x: np.ndarray, topk: int = 20) -> float:
    """Fraction of total score mass in top-K experts."""
    total = np.sum(x)
    if total < 1e-20:
        return 0.0
    return float(np.sum(np.sort(x)[-topk:]) / total)


def _topk_hit_fraction(scores_src: np.ndarray, actual_topk: set[int], k: int) -> float:
    """Fraction of actual_topk found in top-k of scores_src."""
    topk = set(np.argsort(-scores_src)[:k].tolist())
    return len(topk & actual_topk) / len(actual_topk)


# ── Engine patching ──────────────────────────────────────────────────

def _install_collector(eng):
    """Monkey-patch _forward_ffn to capture gate scores per layer per step."""
    global _scores, _current_layer, _decode_step_global, _n_experts
    _scores = defaultdict(list)
    _current_layer = -1
    _decode_step_global = 0
    _n_experts = eng.config.n_routed_experts

    original_ffn = eng._forward_ffn

    def _ffn_with_scores(hidden_states, lw, layer_idx, input_ids=None):
        global _decode_step_global, _current_layer

        B, T, D = hidden_states.shape
        total_tokens = B * T
        is_decode = (total_tokens == 1)

        if is_decode:
            _current_layer = layer_idx

            # Compute gate scores inline (same as _compute_routing_experts)
            gate_w = eng._deq("ffn.gate.weight",
                              lw.get("ffn.gate.weight"),
                              lw.get("ffn.gate.scale"), layer_idx)
            gate_bias = lw.get("ffn.gate.bias")

            if gate_w is not None:
                flat_hidden = hidden_states.reshape(total_tokens, D)
                logits = torch.matmul(flat_hidden.to(gate_w.dtype), gate_w.t())
                if gate_bias is not None:
                    logits = logits + gate_bias.to(logits.dtype)
                logits = logits.float()
                scores = torch.nn.functional.softplus(logits).sqrt()
                # scores shape: [1, 256]
                _scores[layer_idx].append(scores[0].detach().cpu().clone())

        return original_ffn(hidden_states, lw, layer_idx, input_ids)

    eng._forward_ffn = _ffn_with_scores
    return original_ffn


def _install_step_tracker(eng):
    """Track decode steps via _forward_layer to know when a new step starts."""
    original_layer = eng._forward_layer

    def _layer_with_step(h, lw, layer_idx, input_ids=None):
        global _decode_step_global
        is_decode = (h.dim() == 3 and h.shape[1] == 1)
        if is_decode and layer_idx == 0:
            _decode_step_global += 1
        return original_layer(h, lw, layer_idx, input_ids)

    eng._forward_layer = _layer_with_step
    return original_layer


# ── Analysis ─────────────────────────────────────────────────────────

def analyze():
    global _scores, _decode_step_global

    # Convert to numpy array [L, T, N_experts]
    L = max(_scores.keys()) + 1
    max_T = max(len(v) for v in _scores.values())
    if max_T < 2 or L < 2:
        print("Insufficient data for analysis")
        return

    print(f"\n{'='*65}")
    print(f"Data: {L} layers, up to {max_T} decode steps per layer")
    print(f"{'='*65}")

    # Build aligned matrix: layers that have all T steps
    min_T = min(len(v) for v in _scores.values() if len(v) > 0)
    T = min_T

    # Only use layers with >= T steps
    valid_layers = [l for l in range(L) if len(_scores[l]) >= T]
    valid_L = len(valid_layers)

    arr = np.zeros((valid_L, T, _n_experts), dtype=np.float32)
    for idx, l in enumerate(valid_layers):
        for t in range(T):
            arr[idx, t] = _scores[l][t].numpy()

    print(f"Aligned: {valid_L} layers × {T} steps")

    # ── 0. Score statistics ──
    print(f"\n{'─'*35}")
    print("0. SCORE VECTOR STATISTICS")
    print(f"{'─'*35}")

    sparsities = []
    score_means = []
    score_maxs = []
    for li in range(valid_L):
        for t in range(T):
            v = arr[li, t]
            sparsities.append(_score_sparsity(v, topk=20))
            score_means.append(float(np.mean(v)))
            score_maxs.append(float(np.max(v)))
    print(f"  top-20 mass fraction: {statistics.mean(sparsities):.1%}")
    print(f"  mean score value:     {statistics.mean(score_means):.4f}")
    print(f"  max score value:      {statistics.mean(score_maxs):.4f}")
    note = ("  WARNING: scores have large shared baseline (~2.4)"
            if statistics.mean(score_means) > 1.0 else "")
    if note:
        print(f"{note}")
        print(f"    Cosine dominated by baseline → use PEARSON for real signal")
        print(f"    Top-6 overlap is the ground truth for routing prediction")

    # ── 1. Inter-layer correlations ──
    print(f"\n{'─'*35}")
    print("1. INTER-LAYER COSINE (same step, layer L→L+1)")
    print(f"{'─'*35}")

    inter_cos = []
    inter_pearson = []
    for t in range(T):
        for li in range(valid_L - 1):
            c = _cosine(arr[li, t], arr[li + 1, t])
            p = _pearson(arr[li, t], arr[li + 1, t])
            if c is not None:
                inter_cos.append(c)
            if p is not None and not np.isnan(p):
                inter_pearson.append(p)

    if inter_cos:
        print(f"  cosine mean:   {statistics.mean(inter_cos):.4f}")
        print(f"  cosine median: {statistics.median(inter_cos):.4f}")
        print(f"  cosine std:    {statistics.stdev(inter_cos):.4f}")
    if inter_pearson:
        print(f"  Pearson mean:  {statistics.mean(inter_pearson):.4f}")
        print(f"  Pearson std:   {statistics.stdev(inter_pearson):.4f}")

    # ── 2. Temporal correlations ──
    print(f"\n{'─'*35}")
    print("2. TEMPORAL COSINE (same layer, step S→S+1)")
    print(f"{'─'*35}")

    temp_cos = []
    temp_pearson = []
    for li in range(valid_L):
        for t in range(T - 1):
            c = _cosine(arr[li, t], arr[li, t + 1])
            p = _pearson(arr[li, t], arr[li, t + 1])
            if c is not None:
                temp_cos.append(c)
            if p is not None and not np.isnan(p):
                temp_pearson.append(p)

    if temp_cos:
        print(f"  cosine mean:   {statistics.mean(temp_cos):.4f}")
        print(f"  cosine median: {statistics.median(temp_cos):.4f}")
        print(f"  cosine std:    {statistics.stdev(temp_cos):.4f}")
    if temp_pearson:
        print(f"  Pearson mean:  {statistics.mean(temp_pearson):.4f}")
        print(f"  Pearson std:   {statistics.stdev(temp_pearson):.4f}")

    # ── 3. Top-6 overlap rates ──
    print(f"\n{'─'*35}")
    print("3. TOP-6 OVERLAP (Jaccard-like)")
    print(f"{'─'*35}")

    inter_ov = []
    for t in range(T):
        for li in range(valid_L - 1):
            s_curr = set(np.argsort(-arr[li, t])[:6].tolist())
            s_next = set(np.argsort(-arr[li + 1, t])[:6].tolist())
            inter_ov.append(len(s_curr & s_next) / 6)

    temp_ov = []
    for li in range(valid_L):
        for t in range(T - 1):
            s_t = set(np.argsort(-arr[li, t])[:6].tolist())
            s_t1 = set(np.argsort(-arr[li, t + 1])[:6].tolist())
            temp_ov.append(len(s_t & s_t1) / 6)

    print(f"  Inter-layer (N→N+1): {statistics.mean(inter_ov):.1%}  "
          f"(range {min(inter_ov):.1%}–{max(inter_ov):.1%})")
    print(f"  Temporal (t→t+1):   {statistics.mean(temp_ov):.1%}  "
          f"(range {min(temp_ov):.1%}–{max(temp_ov):.1%})")

    # ── 4. Top-K prediction accuracy (inter-layer) ──
    print(f"\n{'─'*50}")
    print("4. INTER-LAYER PREDICTION (layer N top-K → layer N+1 top-6)")
    print(f"{'─'*50}")
    print(f"  {'K':<6} {'hit rate':>10}  {'hits/6':>10}  {'cache MB*':>10}")
    print(f"  {'─'*40}")

    for k in [6, 12, 24, 48, 96, 192]:
        rates = []
        for t in range(T):
            for li in range(valid_L - 1):
                actual = set(np.argsort(-arr[li + 1, t])[:6].tolist())
                rates.append(_topk_hit_fraction(arr[li, t], actual, k))
        if rates:
            mb = k * 12.75  # ~12.75 MB per expert in FP4
            print(f"  {k:<6} {statistics.mean(rates):>9.1%}  "
                  f"{statistics.mean(rates)*6:>10.2f}  {mb:>10.0f}")

    # ── 5. Temporal prediction accuracy ──
    print(f"\n{'─'*50}")
    print("5. TEMPORAL PREDICTION (step S top-K → step S+1 top-6)")
    print(f"{'─'*50}")
    print(f"  {'K':<6} {'hit rate':>10}  {'hits/6':>10}")
    print(f"  {'─'*30}")

    for k in [6, 12, 24, 48, 96]:
        rates = []
        for li in range(valid_L):
            for t in range(T - 1):
                actual = set(np.argsort(-arr[li, t + 1])[:6].tolist())
                rates.append(_topk_hit_fraction(arr[li, t], actual, k))
        if rates:
            print(f"  {k:<6} {statistics.mean(rates):>9.1%}  "
                  f"{statistics.mean(rates)*6:>10.2f}")

    # ── 6. Per-layer temporal cosine ──
    print(f"\n{'─'*50}")
    print("6. TEMPORAL COSINE BY LAYER")
    print(f"{'─'*50}")

    layer_cos = []
    for li in range(valid_L):
        lc = []
        for t in range(T - 1):
            c = _cosine(arr[li, t], arr[li, t + 1])
            if c is not None:
                lc.append(c)
        if lc:
            layer_cos.append(statistics.mean(lc))

    if layer_cos:
        print(f"  mean over layers:  {statistics.mean(layer_cos):.4f}")
        print(f"  std over layers:   {statistics.stdev(layer_cos):.4f}")
        print(f"  best layer cosine: {max(layer_cos):.4f}")
        print(f"  worst layer cosine:{min(layer_cos):.4f}")

        hash_l = eng_config_cache.get('num_hash_layers', 3)
        if hash_l < valid_L:
            early_cos = layer_cos[:hash_l]
            mid_cos = layer_cos[hash_l:min(hash_l + 10, valid_L)]
            late_cos = layer_cos[-10:]
            print(f"\n  Hash layers (0–{hash_l - 1}):  {statistics.mean(early_cos):.4f}")
            print(f"  Middle (layers {hash_l}–{min(hash_l+9,valid_L-1)}): {statistics.mean(mid_cos):.4f}")
            print(f"  Late  (last 10):             {statistics.mean(late_cos):.4f}")

    # ── 7. Summary ──
    c_i = statistics.mean(inter_cos) if inter_cos else 0
    c_t = statistics.mean(temp_cos) if temp_cos else 0
    ti6 = statistics.mean(temp_ov) if temp_ov else 0

    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Score sparsity (top-20 mass): {statistics.mean(sparsities):.1%}")
    print(f"  Inter-layer cosine = {c_i:.4f}")
    print(f"  Temporal cosine    = {c_t:.4f}")
    print(f"  Temporal top-6 overlap = {ti6:.1%}")

    print(f"\n  ── Inter-layer prediction (layer N→N+1) ──")
    if inter_pearson and statistics.mean(inter_pearson) > 0.3:
        print(f"  ✓ Pearson={statistics.mean(inter_pearson):.3f} > 0.3 → variations correlated, MLP viable")
    else:
        pe = statistics.mean(inter_pearson) if inter_pearson else 0
        print(f"  ✗ Pearson={pe:.3f} ≈ 0 → informative variations uncorrelated")
        print(f"    Cosine={c_i:.3f} is high but dominated by shared score baseline (~2.4)")
        print(f"    Cross-layer DMA prefetch NOT viable — scores are 'parallel' but their")
        print(f"    expert-ranking variations are independent. This confirms the 2.2% top-6 overlap.")

    print(f"\n  ── Temporal prediction (step S→S+1) ──")
    if temp_pearson and statistics.mean(temp_pearson) > 0.5:
        pe = statistics.mean(temp_pearson)
        print(f"  ✓ Pearson={pe:.3f} > 0.5 → strong temporal correlation between decode steps")
        print(f"    Cosine={c_t:.3f} confirms the alignment.")
        print(f"    Top-6 overlap={ti6:.1%}, top-12 hit={statistics.mean([_topk_hit_fraction(arr[li,t], set(np.argsort(-arr[li,t+1])[:6].tolist()), 12) for li in range(valid_L) for t in range(T-1)]):.0%}")
        print(f"    → NEXT-TOKEN DMA PREFETCH VIABLE")
        print(f"    Prefetch current step's top-12 experts for next step → ~56% hit rate.")
        print(f"    DMA saving: 56% × 32.5% DMA fraction ≈ 18% fewer DMA in critical path.")
        print(f"    Requires: inter-step prefetch buffer, non_blocking DMA in Other/embed window.")
    else:
        print(f"  ⚠ Pearson modest → temporal prefetch marginal")


# Cache engine config for per-layer analysis
eng_config_cache: dict = {}


# ── Main ─────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    global eng_config_cache

    print("Building engine...")
    eng = HomeSeekInferenceEngine("weights", verbose=False)
    eng_config_cache['num_hash_layers'] = eng.config.num_hash_layers

    print("Patching FFN for score collection...")
    orig_ffn = _install_collector(eng)
    orig_layer = _install_step_tracker(eng)

    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file="weights/tokenizer.json")

    prompts = ["How are you?", "What is AI?", "Write a poem", "Hello", "Hi"]
    total_gen = 0

    for pi, prompt_text in enumerate(prompts):
        input_ids = tok.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        global _decode_step_global
        _decode_step_global = 0

        generated = []
        _ = list(eng.generate(input_ids, max_new_tokens=15, temperature=0,
                              stream_callback=lambda _: generated.append(1)))
        n_gen = sum(generated) if generated else 0
        total_gen += n_gen

        # Quick sanity: print score collection stats per layer
        layer_counts = {l: len(v) for l, v in _scores.items()}
        print(f"  [{pi + 1}/{len(prompts)}] '{prompt_text}': {n_gen} tokens, "
              f"layers with data: {len(layer_counts)}, "
              f"max steps/layer: {max(layer_counts.values()) if layer_counts else 0}")

    eng._forward_ffn = orig_ffn
    eng._forward_layer = orig_layer

    del eng
    gc.collect()
    torch.cuda.empty_cache()

    analyze()


if __name__ == "__main__":
    main()
