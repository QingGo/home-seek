"""Measure genuine wall-clock time of M1 kernel path — zero profiling tax.

No torch.cuda.synchronize(), no events, no trace — just time.perf_counter()
around the actual code paths. CUDA kernel launches are asynchronous so we
need a single sync at the end of the measured region.

Usage:
    python scripts/m1_wallclock.py
"""

import time
import torch

from home_seek.inference_engine import HomeSeekInferenceEngine


# ── accumulators ───────────────────────────────────────────────
_m1_total_ms = 0.0
_m1_count = 0
_m1_dma_ms = 0.0
_m1_kernel_ms = 0.0
_ffn_total_ms = 0.0
_ffn_count = 0
_layer_total_ms = 0.0
_layer_count = 0
_decode_total_ms = 0.0
_decode_count = 0
_m1_kernel_only_ms = 0.0
_m1_kernel_count = 0


def _patch_engine(eng):
    """Monkey-patch to add wall-clock measurements. No extra CUDA syncs."""

    # ── _forward_ffn_m1_triton ──
    original_m1 = eng._forward_ffn_m1_triton

    def traced_m1(hidden_states, flat_hidden, flat_topk_idx, flat_topk_w, layer_idx):
        global _m1_total_ms, _m1_count, _m1_dma_ms, _m1_kernel_ms, _m1_kernel_only_ms, _m1_kernel_count
        t0 = time.perf_counter()

        # Measure DMA (load) separately
        t_dma0 = time.perf_counter()
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
        t_dma1 = time.perf_counter()

        # Measure fused_moe kernel launch + execution
        t_k0 = time.perf_counter()
        try:
            result = eng._fused_moe.forward_m1_fp4(flat_hidden, fp4_data, eids)
        except Exception:
            return None

        # Sync to capture actual GPU time
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        _m1_total_ms += (t1 - t0) * 1000
        _m1_dma_ms += (t_dma1 - t_dma0) * 1000
        _m1_kernel_ms += (t1 - t_k0) * 1000
        _m1_count += 1

        return result.reshape(hidden_states.shape)

    eng._forward_ffn_m1_triton = traced_m1

    # ── _forward_ffn ──
    original_ffn = eng._forward_ffn

    def traced_ffn(hidden_states, lw, layer_idx, input_ids=None):
        global _ffn_total_ms, _ffn_count
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        t0s = time.perf_counter()
        result = original_ffn(hidden_states, lw, layer_idx, input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ffn_total_ms += (t1 - t0) * 1000
        _ffn_count += 1
        return result

    eng._forward_ffn = traced_ffn

    # ── _forward_layer ──
    original_layer = eng._forward_layer

    def traced_layer(h, lw, layer_idx, input_ids):
        global _layer_total_ms, _layer_count
        t0 = time.perf_counter()
        result = original_layer(h, lw, layer_idx, input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _layer_total_ms += (t1 - t0) * 1000
        _layer_count += 1
        return result

    eng._forward_layer = traced_layer

    return original_ffn, original_layer, original_m1


@torch.no_grad()
def main():
    global _m1_total_ms, _m1_count, _m1_dma_ms, _m1_kernel_ms
    global _ffn_total_ms, _ffn_count, _layer_total_ms, _layer_count
    global _decode_total_ms, _decode_count

    print("=== Wall-Clock M1 Measurement (no profiling tax) ===")
    print("Initializing engine...")
    eng = HomeSeekInferenceEngine("weights", verbose=False)
    _patch_engine(eng)

    # Use a single standard prompt, generate a modest number of tokens
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file="weights/tokenizer.json")
    prompt_text = "How are you?"
    input_ids = tok.encode(prompt_text)
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    print(f"Prompt: '{prompt_text}' ({len(input_ids[0])} tokens)")
    print("Running inference (max 30 new tokens)...")

    # We need to measure the entire decode loop, so we wrap generate() callback
    decode_start = None
    token_count = 0
    decode_wall_ms = 0.0

    def stream_cb(token_id):
        nonlocal decode_start, token_count, decode_wall_ms
        if decode_start is not None:
            torch.cuda.synchronize()
            decode_wall_ms += (time.perf_counter() - decode_start) * 1000
            token_count += 1
        decode_start = time.perf_counter()

    prefill_start = time.perf_counter()
    generated = list(eng.generate(input_ids, max_new_tokens=30, temperature=0,
                                  stream_callback=stream_cb))
    prefill_elapsed = time.perf_counter() - prefill_start

    # Compute per-layer breakdown
    m1_avg = _m1_total_ms / _m1_count if _m1_count else 0
    m1_dma_avg = _m1_dma_ms / _m1_count if _m1_count else 0
    m1_kernel_avg = _m1_kernel_ms / _m1_count if _m1_count else 0
    ffn_avg = _ffn_total_ms / _ffn_count if _ffn_count else 0
    layer_avg = _layer_total_ms / _layer_count if _layer_count else 0
    decode_per_tok = decode_wall_ms / token_count if token_count else 0

    print(f"\n{'='*60}")
    print(f"Generated {token_count} tokens in {prefill_elapsed:.1f}s total")
    print(f"Overall throughput: {token_count / prefill_elapsed:.2f} t/s")
    print(f"\n{'='*60}")
    print(f"Per-call breakdown (wall-clock, all per-layer aggregate):")
    print(f"  _forward_layer avg:     {layer_avg:.2f} ms  (×{_layer_count} calls)")
    print(f"  _forward_ffn avg:       {ffn_avg:.2f} ms  (×{_ffn_count} calls)")
    print(f"  _forward_ffn_m1 avg:    {m1_avg:.2f} ms  (×{_m1_count} calls)")
    print(f"    └─ DMA (load):        {m1_dma_avg:.2f} ms  ({m1_dma_avg/m1_avg*100:.0f}%)")
    print(f"    └─ kernel+sync:       {m1_kernel_avg:.2f} ms  ({m1_kernel_avg/m1_avg*100:.0f}%)")
    print(f"  Attn+MHC estimated:     {layer_avg - ffn_avg:.2f} ms  ({100 - ffn_avg/layer_avg*100:.0f}%)")
    print(f"\n{'='*60}")
    print(f"Decode-only (token {len(input_ids[0])}..{len(input_ids[0])+token_count-1}):")
    print(f"  Decode wall time:      {decode_wall_ms:.0f} ms")
    print(f"  Per-token decode:      {decode_per_tok:.1f} ms  ({1000/decode_per_tok:.2f} t/s)")
    print(f"\nDecomposition per token (from layer averages × 43 layers):")
    per_tok_layer = layer_avg * 43
    per_tok_ffn = ffn_avg * 43
    per_tok_m1 = m1_avg * 43
    per_tok_m1_dma = m1_dma_avg * 43
    per_tok_m1_kernel = m1_kernel_avg * 43
    per_tok_attn_mhc = per_tok_layer - per_tok_ffn
    overhead = decode_per_tok - per_tok_layer

    print(f"  Per-tok decode:         {decode_per_tok:.1f} ms  (100%)")
    print(f"  ├─ 43 layers total:     {per_tok_layer:.1f} ms  ({per_tok_layer/decode_per_tok*100:.0f}%)")
    print(f"  │   ├─ FFN:             {per_tok_ffn:.1f} ms  ({per_tok_ffn/decode_per_tok*100:.0f}%)")
    print(f"  │   │   ├─ M1 total:    {per_tok_m1:.1f} ms  ({per_tok_m1/decode_per_tok*100:.0f}%)")
    print(f"  │   │   │   ├─ DMA:     {per_tok_m1_dma:.1f} ms")
    print(f"  │   │   │   └─ kernel:  {per_tok_m1_kernel:.1f} ms")
    print(f"  │   │   └─ routing+ovhd:{per_tok_ffn - per_tok_m1:.1f} ms")
    print(f"  │   └─ Attn+MHC:        {per_tok_attn_mhc:.1f} ms  ({per_tok_attn_mhc/decode_per_tok*100:.0f}%)")
    print(f"  └─ Overhead:            {overhead:.1f} ms  ({overhead/decode_per_tok*100:.0f}%)")
    print(f"      (lm_head+embed+sample+loop overhead)")

    # ── Also print actual kernel-only time (measured inside forward_m1_fp4) ──
    print(f"\n--- Actual fused_moe kernel time (no sync overhead) ---")

    # Re-measure with a tighter instrument: measure only the kernel launches
    # inside forward_m1_fp4, without the load/overhead

    original_fwd_m1 = eng._fused_moe.forward_m1_fp4

    _kernel_inner_ms = []

    def traced_fwd_m1(hidden_states, fp4_data, routing_w):
        t0 = time.perf_counter()
        result = original_fwd_m1(hidden_states, fp4_data, routing_w)
        torch.cuda.synchronize()
        _kernel_inner_ms.append((time.perf_counter() - t0) * 1000)
        return result

    eng._fused_moe.forward_m1_fp4 = traced_fwd_m1

    print("Running second inference for kernel-only measurement...")
    input_ids2 = tok.encode("What is AI?")
    input_ids2 = torch.tensor([input_ids2], dtype=torch.long)

    _kernel_inner_ms.clear()
    list(eng.generate(input_ids2, max_new_tokens=15, temperature=0))

    if _kernel_inner_ms:
        avg_kernel = sum(_kernel_inner_ms) / len(_kernel_inner_ms)
        print(f"\nFused forward_m1_fp4 (kernel-only wall): {avg_kernel:.3f} ms/call  (×{len(_kernel_inner_ms)} calls)")
        print(f"  Per-token (×43 layers): {avg_kernel * 43:.1f} ms")
        print(f"  This = gate_up + down_accum Triton kernels + GPU sync")
    else:
        print("\nNo M1 kernel calls tracked (all hot-batched?)")


if __name__ == "__main__":
    main()
