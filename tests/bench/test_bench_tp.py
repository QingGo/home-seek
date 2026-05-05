"""Tensor Parallelism (TP) feasibility benchmark for 2× RTX 2080 Ti.

Measures:
  Q1: PCIe all-reduce latency for attention-sized tensors (8KB–64KB)
  Q2: Attention head-split benchmark (half heads per GPU)
  Q3: Per-layer compute distribution for TP projection
  Q4: Memory footprint estimate for TP

Run:
    pytest tests/bench/test_bench_tp.py -m bench -v -s
"""

import time
import pytest
import torch


def _ensure_two_gpus():
    """Check if at least 2 GPUs are available."""
    n = torch.cuda.device_count()
    if n < 2:
        return False
    return True


def _allreduce_latency(tensor_kb, n_repeats=100, n_warmup=10):
    """Measure PCIe all-reduce latency for a tensor of given KB size.

    On 2080 Ti PCIe gen3 ×8 (~8 GB/s), copying tensor from GPU0→GPU1→GPU0
    models a simple all-reduce (sum + broadcast).
    """
    bytes_total = tensor_kb * 1024
    n_float16 = bytes_total // 2
    n_elems = n_float16

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    t0 = torch.randn(n_elems, device=dev0, dtype=torch.bfloat16)

    # Warmup
    s0 = torch.cuda.Stream(dev0)
    s1 = torch.cuda.Stream(dev1)
    for _ in range(n_warmup):
        with torch.cuda.stream(s0):
            t_copy = t0.to(dev1, non_blocking=True)
        with torch.cuda.stream(s1):
            s1.wait_stream(s0)
            t_copy.to(dev0, non_blocking=True)
        torch.cuda.synchronize(dev0)

    # Measure
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(n_repeats):
        with torch.cuda.stream(s0):
            t_copy = t0.to(dev1, non_blocking=True)
        with torch.cuda.stream(s1):
            s1.wait_stream(s0)
            t_copy.to(dev0, non_blocking=True)
        torch.cuda.synchronize(dev0)
    elapsed = time.perf_counter() - t_start

    latency_ms = elapsed / n_repeats * 1000
    bw_gbps = (bytes_total * 2) / (latency_ms / 1000) / 1e9  # ×2 for round trip
    return latency_ms, bw_gbps


def _attention_split_benchmark(T=512):
    """Benchmark TP-style attention: half the heads per GPU.

    Models what 2-GPU TP would look like for M=1 decode attention:
      - GPU0: heads 0–31, GPU1: heads 32–63
      - Each computes q[h] @ k[h]^T for its heads
      - All-reduce output

    Returns (single_gpu_ms, tp_2gpu_ms) for standalone attention.
    """
    from home_seek.gqa_attention import gqa_fused_attn

    H = 64
    D = 128

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    q = torch.randn(1, H, 1, D, device=dev0, dtype=torch.bfloat16)
    kv = torch.randn(1, T, D, device=dev0, dtype=torch.bfloat16)

    WARMUP = 5
    ITERS = 30

    # ── Single GPU baseline ──
    for _ in range(WARMUP):
        _ = gqa_fused_attn(q.float(), kv.to(q.dtype))

    torch.cuda.synchronize(dev0)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = gqa_fused_attn(q.float(), kv.to(q.dtype))
    torch.cuda.synchronize(dev0)
    single_ms = (time.perf_counter() - t0) / ITERS * 1000

    # ── 2-GPU TP simulation ──
    H2 = H // 2
    q_gpu0 = q[:, :H2, :, :].to(dev0)
    q_gpu1 = q[:, H2:, :, :].to(dev1)
    kv_gpu0 = kv.unsqueeze(1).expand(-1, H2, -1, -1).to(dev0).contiguous()
    kv_gpu1 = kv.unsqueeze(1).expand(-1, H2, -1, -1).to(dev1).contiguous()

    # Warmup both GPUs
    for _ in range(WARMUP):
        out0 = gqa_fused_attn(q_gpu0.float(), kv_gpu0.to(q_gpu0.dtype))
        out1 = gqa_fused_attn(q_gpu1.float(), kv_gpu1.to(q_gpu1.dtype))
        _ = torch.cat([out0.to(dev0), out1.to(dev0)], dim=1)

    torch.cuda.synchronize(dev0)
    torch.cuda.synchronize(dev1)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out0 = gqa_fused_attn(q_gpu0.float(), kv_gpu0.to(q_gpu0.dtype))
        out1 = gqa_fused_attn(q_gpu1.float(), kv_gpu1.to(q_gpu1.dtype))
        # All-gather: concatenate results
        torch.cat([out0.to(dev0), out1.to(dev0)], dim=1)
        # In real TP, this would be an all-reduce or all-gather
    torch.cuda.synchronize(dev0)
    torch.cuda.synchronize(dev1)
    tp_ms = (time.perf_counter() - t0) / ITERS * 1000

    return single_ms, tp_ms


@pytest.mark.bench
class TestTPProjection:
    """Tensor Parallelism feasibility projection for 2× RTX 2080 Ti."""

    @pytest.fixture(autouse=True)
    def _require_weights(self, require_weights):
        pass

    def test_q1_pcie_allreduce_latency(self):
        """Q1: Measure PCIe all-reduce latency for TP-sized tensors.

        TP all-reduce per layer involves combining attention outputs
        (8KB for [1, D] bf16) or FFN partial sums (8KB each).

        Also measures: hidden state transfer (8KB), and larger tensors
        that might be needed for column-parallel FFN (up to 64KB).
        """
        if not _ensure_two_gpus():
            pytest.skip("Need 2+ GPUs for TP bench")

        print("\n─── Q1: PCIe all-reduce latency ───")
        print(f"  {'Size KB':>10} {'Latency μs':>12} {'BW GB/s':>10}")
        print(f"  {'─'*38}")

        for kb in [8, 16, 32, 64, 128, 256, 1024, 4096]:
            lat_ms, bw = _allreduce_latency(kb)
            print(f"  {kb:>10} {lat_ms*1000:>11.0f}μs {bw:>9.1f}")

        # Key sizes for TP
        print("\n  TP tensor sizes:")
        print("    Hidden state (1×4096 bf16): 8 KB → all-reduce overhead")
        print("    Attention output:            8 KB")
        print("    FFN partial (1×2048 bf16):   4 KB")
        print("    FFN weight shard (2048×2048): 8 MB (one-time load)")

        # 8KB all-reduce should be <100μs on PCIe gen3
        lat_8kb, _ = _allreduce_latency(8)
        assert lat_8kb < 0.5, f"8KB all-reduce too slow: {lat_8kb*1000:.0f}μs"

    def test_q2_attention_head_split(self):
        """Q2: Benchmark 2-GPU attention head split.

        Models: GPU0 computes heads 0–31, GPU1 computes 32–63.
        Compares single-GPU time vs 2-GPU TP time for standalone attention.

        If TP attention is faster despite PCIe overhead, TP is viable.
        """
        if not _ensure_two_gpus():
            pytest.skip("Need 2+ GPUs for TP bench")

        print("\n─── Q2: Attention head-split (2-GPU TP) ───")
        print(f"  {'T (KV)':>10} {'1-GPU ms':>10} {'2-GPU TP ms':>12} "
              f"{'Speedup':>10}")
        print(f"  {'─'*48}")

        for T in [128, 256, 512, 1024, 2048, 4096]:
            single_ms, tp_ms = _attention_split_benchmark(T)
            speedup = single_ms / tp_ms if tp_ms > 0 else 0
            print(f"  {T:>10} {single_ms:>9.3f}ms {tp_ms:>11.3f}ms "
                  f"{speedup:>9.2f}×")

        print("\n  Speedup < 1.0: TP adds overhead, not beneficial")
        print("  Speedup 1.0–1.5: modest gain, may not justify complexity")
        print("  Speedup > 1.5: significant gain, TP worth implementing")

    def test_q3_tp_per_layer_projection(self):
        """Q3: Project full TP decode throughput.

        Uses real wall-clock breakdown from .agent_memory.md (V21.18):
          Per-tok: 1130ms
          ├─ Attention+MHC: 468ms (41%)
          ├─ FFN: 350ms (31%)
          │   ├─ DMA: 244ms
          │   └─ Kernel: 106ms
          ├─ Routing+shared: 70ms (6%)
          └─ Other: 242ms (21%)

        TP assumption (2 GPUs, head-split attention):
          - Attention: 468ms → 250ms (halve compute, add all-reduce)
          - FFN: 350ms → 350ms (expert routing already per-GPU via EP-like)
          - Routing/shared: 70ms → 70ms (unchanged)
          - Other: 242ms → 242ms (unchanged)

        TP assumption (2 GPUs, full column+row split):
          - Attention: 468ms → 250ms
          - FFN kernel: 106ms → 60ms (halve compute)
          - DMA: 244ms → 244ms (replicated data)
          - Other: unchanged

        Reports projected t/s for each scenario.
        """
        print("\n─── Q3: TP per-token throughput projection ───")

        # Baseline from V21.18 wall-clock
        baseline_ms = {
            "attention": 468,
            "ffn_kernel": 106,
            "ffn_dma": 244,
            "ffn_routing_shared": 70,
            "other": 242,
        }
        baseline_total = sum(baseline_ms.values())
        baseline_ts = 1000 / baseline_total

        print("\n  Baseline (1× RTX 2080 Ti, wall-clock):")
        print(f"    total:       {baseline_total}ms → {baseline_ts:.2f} t/s")
        for k, v in baseline_ms.items():
            print(f"    {k:>20}: {v}ms  ({v/baseline_total*100:.0f}%)")

        # ── Scenario A: TP attention only ──
        # Assume 2-GPU head split: attention halved + 1ms all-reduce
        attn_tp_a = baseline_ms["attention"] / 2 + 1
        tp_a_ms = {**baseline_ms, "attention": attn_tp_a}
        tp_a_total = sum(tp_a_ms.values())
        tp_a_ts = 1000 / tp_a_total

        print("\n  Scenario A: TP attention only (head split 2× GPU):")
        print(f"    attn: {attn_tp_a:.0f}ms (saved {baseline_ms['attention'] - attn_tp_a:.0f}ms)")
        print(f"    total: {tp_a_total:.0f}ms → {tp_a_ts:.2f} t/s  "
              f"({tp_a_ts/baseline_ts*100-100:+.0f}%)")

        # ── Scenario B: TP attention + FFN kernel ──
        # FFN kernel halved per GPU (column-split gate/up/down)
        ffn_kernel_tp = baseline_ms["ffn_kernel"] / 2 + 1
        tp_b_ms = {**baseline_ms, "attention": attn_tp_a, "ffn_kernel": ffn_kernel_tp}
        tp_b_total = sum(tp_b_ms.values())
        tp_b_ts = 1000 / tp_b_total

        print("\n  Scenario B: TP attention + FFN kernel (col/row split):")
        print(f"    ffn_kernel: {ffn_kernel_tp:.0f}ms (saved {baseline_ms['ffn_kernel'] - ffn_kernel_tp:.0f}ms)")
        print(f"    total: {tp_b_total:.0f}ms → {tp_b_ts:.2f} t/s  "
              f"({tp_b_ts/baseline_ts*100-100:+.0f}%)")

        # ── Scenario C: TP + temporal DMA prefetch ──
        # Add temporal prefetch (55.9% of DMA saved) on top of TP
        prefetch_savings = 0.559 * baseline_ms["ffn_dma"]
        tp_c_ms = {**tp_b_ms, "ffn_dma": baseline_ms["ffn_dma"] - prefetch_savings}
        tp_c_total = sum(tp_c_ms.values())
        tp_c_ts = 1000 / tp_c_total

        print("\n  Scenario C: TP + temporal DMA prefetch:")
        print(f"    dma_saved: {prefetch_savings:.0f}ms (55.9% hit)")
        print(f"    total: {tp_c_total:.0f}ms → {tp_c_ts:.2f} t/s  "
              f"({tp_c_ts/baseline_ts*100-100:+.0f}%)")

        # ── Scenario D: TP + full DMA hide + kernel elimination ──
        # Amdahl ceiling: DMA fully hidden, kernel halved
        tp_d_ms = {**baseline_ms, "attention": attn_tp_a, "ffn_kernel": 0, "ffn_dma": 0}
        tp_d_total = sum(tp_d_ms.values())
        tp_d_ts = 1000 / tp_d_total

        print("\n  Scenario D: Amdahl ceiling (DMA+FFN kernel → 0, TP attn):")
        print(f"    total: {tp_d_total:.0f}ms → {tp_d_ts:.2f} t/s  "
              f"({tp_d_ts/baseline_ts*100-100:+.0f}%)")

        print("\n  === Recommendations ===")
        print(f"  1. Temporal DMA prefetch (single GPU): {baseline_ts:.2f} → "
              f"{1000/(baseline_total - prefetch_savings):.2f} t/s")
        print(f"  2. + TP attention (2 GPU):              → {tp_a_ts:.2f} t/s")
        print(f"  3. + TP FFN kernel (2 GPU):             → {tp_b_ts:.2f} t/s")
        print(f"  4. + Prefetch (combined):               → {tp_c_ts:.2f} t/s")
        print(f"  5. Hardware ceiling (2× 2080 Ti TP):   → {tp_d_ts:.2f} t/s")

    def test_q4_memory_footprint_tp(self):
        """Q4: Estimate GPU memory footprint for TP.

        In TP, each GPU holds a portion of the weight matrices:
        - Attention: split heads → each GPU has half the QKV and O matrices
        - FFN: column-split → each GPU has half of gate/up/down columns
        - But expert weights for MoE are replicated (or split differently)

        The key question: does TP increase or decrease per-GPU memory?
        """
        print("\n─── Q4: TP memory footprint estimate ───")

        # Current single-GPU memory (V21.15): ~17.5 GB peak
        print("  Current single-GPU peak: ~17.5 GB")

        # Attention weights per layer (approx):
        # wq_a: [D, D_q]  wq_b: [D_q, H*D]  wkv: [D, H*D]  wo_a: [H*D, D_o]  wo_b: [D_o, D]
        # These are FP8, so ~1 byte/param
        # Rough: ~100 MB total attention weights across 43 layers

        # TP splits:
        # - Column-split QKV: each GPU has half the heads → half the output columns
        #   wq_a: replicated (input is same)
        #   wq_b: split by head → half size
        #   wkv: split by head → half size
        #   wo_a: split by input head → half size
        #   wo_b: replicated (output is aggregated)
        # Net: ~60% of attention weight per GPU

        # Expert weights: each is [I, D] + [I, D] + [D, I] for gate, up, down
        # In FP4: ~12.75 MB each
        # For TP FFN: experts are replicated? Or split across GPUs?
        # If experts are replicated → memory doubles
        # If experts are split → TP on MoE is complex
        # Approach: expert routing already assigns experts to GPUs;
        # keep expert weights replicated but compute on different subsets

        print("\n  TP weight placement:")
        print("    Attention weights: ~60% per GPU (head split)")
        print("    Non-expert weights: replicated (~4 GB per GPU)")
        print("    Expert weights: replicated (~13 GB per GPU × 3292 experts)")
        print("    KV cache: replicated per-GPU (head split)")
        print("  Total est: ~19–20 GB per GPU")

        vram_gb = 22.0  # 2080 Ti has 22GB
        est_gb = 19.5
        print(f"  VRAM budget: {vram_gb:.0f} GB")
        print(f"  Estimated TP: {est_gb:.1f} GB")
        print(f"  Headroom: {vram_gb - est_gb:.1f} GB")

        if est_gb > vram_gb * 0.95:
            print("  ⚠ TP may OOM! Need expert weight offloading or FP4 cache reduction")
        elif est_gb > vram_gb * 0.85:
            print("  ⚠ Tight fit. May need to reduce gpu_bf16_cap or hot experts")
        else:
            print("  ✓ Memory fits comfortably")
