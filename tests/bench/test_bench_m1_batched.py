"""Isolated M1 kernel benchmarks: original per-expert vs batched.

Skipped by default (marker: bench).  Run with:
    pytest tests/bench/ -m bench -v
"""

import pytest
import torch

from home_seek.fused_moe import (
    _fused_moe_forward_m1_fp4,
    _fused_moe_forward_m1_fp4_batched,
)
from home_seek._fp4 import cast

# ── benchmark helpers ─────────────────────────────────────────────

I_REAL = 2048
D_REAL = 4096
NUM_EXPERTS = 8
WARMUP = 3
BENCH_ITERS = 20


def _make_expert_fp4(i_dim, d_dim, seed):
    torch.manual_seed(seed)
    w1_bf16 = torch.randn(i_dim, d_dim, dtype=torch.bfloat16)
    w3_bf16 = torch.randn(i_dim, d_dim, dtype=torch.bfloat16)
    w2_bf16 = torch.randn(d_dim, i_dim, dtype=torch.bfloat16)
    w1_p, w1_s = cast(w1_bf16, fmt="e2m1", block_size=(1, 32))
    w3_p, w3_s = cast(w3_bf16, fmt="e2m1", block_size=(1, 32))
    w2_p, w2_s = cast(w2_bf16, fmt="e2m1", block_size=(1, 32))
    return (w1_p.to("cuda"), w1_s.to("cuda"),
            w3_p.to("cuda"), w3_s.to("cuda"),
            w2_p.to("cuda"), w2_s.to("cuda"))


def _make_fp4_data(i_dim, d_dim, num_exp):
    fp4 = {}
    rw = {}
    for i in range(num_exp):
        fp4[i] = _make_expert_fp4(i_dim, d_dim, seed=42 + i)
        rw[i] = round(1.0 / num_exp, 6)
    return fp4, rw


def _gpu_timed(fn, *args, **kwargs):
    """Return (result, gpu_ms) via CUDA events."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    result = fn(*args, **kwargs)
    end_ev.record()
    torch.cuda.synchronize()
    return result, start_ev.elapsed_time(end_ev)


# ── benchmark tests ──────────────────────────────────────────────

@pytest.mark.bench
class TestM1BatchedBench:
    """Standalone M1 kernel GPU time benchmarks.
    Run only when evaluating kernel-level optimization changes:
        pytest tests/bench/ -m bench -v
    """

    _I = 64
    _D = 128
    _NUM_EXP = 8

    def test_batched_vs_original_gpu_time_tiny(self):
        """Tiny dims: batched kernel is significantly faster (occupancy wins)."""
        fp4, rw = _make_fp4_data(self._I, self._D, self._NUM_EXP)
        hidden = torch.randn(1, self._D, device="cuda", dtype=torch.bfloat16)

        for _ in range(WARMUP):
            _fused_moe_forward_m1_fp4(
                hidden, fp4, rw, swiglu_limit=10.0, triton_blocks=(32, 32))
            _fused_moe_forward_m1_fp4_batched(
                hidden, fp4, rw, swiglu_limit=10.0,
                triton_blocks=(32, 32), batch_bufs=None)

        times_orig = []
        times_bat = []
        for _ in range(BENCH_ITERS):
            _, g = _gpu_timed(
                _fused_moe_forward_m1_fp4,
                hidden, fp4, rw, swiglu_limit=10.0, triton_blocks=(32, 32))
            times_orig.append(g)
            _, g = _gpu_timed(
                _fused_moe_forward_m1_fp4_batched,
                hidden, fp4, rw, swiglu_limit=10.0,
                triton_blocks=(32, 32), batch_bufs=None)
            times_bat.append(g)

        avg_o = sum(times_orig) / len(times_orig)
        avg_b = sum(times_bat) / len(times_bat)
        ratio = avg_b / avg_o
        print(f"\n    tiny  orig={avg_o:.3f}ms  batched={avg_b:.3f}ms  "
              f"ratio={ratio:.2f}x  ({ratio*100-100:+.1f}%)")
        assert ratio < 0.75, f"tiny batched should be faster, got ratio={ratio:.3f}"

    def test_batched_vs_original_gpu_time_real(self):
        """Real dims (I=2048,D=4096): kernel-only GPU time comparison.

        Includes torch.cat overhead in batched measurement.
        The batched kernel IS faster (~21% pure GPU) but torch.cat (~0.83ms)
        eats the gains in this benchmark.  Use this test to track whether
        future kernel or copy-overhead improvements close the gap.
        """
        I, D = I_REAL, D_REAL
        fp4, rw = _make_fp4_data(I, D, NUM_EXPERTS)
        hidden = torch.randn(1, D, device="cuda", dtype=torch.bfloat16)

        for _ in range(WARMUP):
            _fused_moe_forward_m1_fp4(
                hidden, fp4, rw, swiglu_limit=10.0, triton_blocks=(32, 32))
            _fused_moe_forward_m1_fp4_batched(
                hidden, fp4, rw, swiglu_limit=10.0,
                triton_blocks=(32, 32), batch_bufs=None)

        times_orig = []
        times_bat = []
        for _ in range(BENCH_ITERS):
            _, g = _gpu_timed(
                _fused_moe_forward_m1_fp4,
                hidden, fp4, rw, swiglu_limit=10.0, triton_blocks=(32, 32))
            times_orig.append(g)
            _, g = _gpu_timed(
                _fused_moe_forward_m1_fp4_batched,
                hidden, fp4, rw, swiglu_limit=10.0,
                triton_blocks=(32, 32), batch_bufs=None)
            times_bat.append(g)

        avg_o = sum(times_orig) / len(times_orig)
        avg_b = sum(times_bat) / len(times_bat)
        print(f"\n    REAL  orig={avg_o:.3f}ms  batched={avg_b:.3f}ms  "
              f"ratio={avg_b/avg_o:.2f}x")

        # Batched kernel alone is ~21% faster; with torch.cat it's ~+12%.
        # Assert we don't catastrophically regress (e.g. >2x slower).
        assert avg_b < avg_o * 2.0, f"batched too slow: {avg_b:.1f}ms vs {avg_o:.1f}ms"

    def test_torchcat_overhead_real(self):
        """Isolate torch.cat cost for real-dim 8-expert weights."""
        I, D = I_REAL, D_REAL
        fp4, _ = _make_fp4_data(I, D, NUM_EXPERTS)
        eids = list(fp4.keys())

        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        torch.cat([fp4[e][0] for e in eids], dim=0)
        torch.cat([fp4[e][1] for e in eids], dim=0)
        torch.cat([fp4[e][2] for e in eids], dim=0)
        torch.cat([fp4[e][3] for e in eids], dim=0)
        torch.cat([fp4[e][4] for e in eids], dim=0)
        torch.cat([fp4[e][5] for e in eids], dim=0)
        end_ev.record()
        torch.cuda.synchronize()
        cat_ms = start_ev.elapsed_time(end_ev)
        total_mb = (
            NUM_EXPERTS * I * D // 2 * 4       # w1_p + w3_p (uint8) × 2
            + NUM_EXPERTS * I * D // 32 * 4 * 2  # w1_s + w3_s (float32) × 2
            + NUM_EXPERTS * D * I // 2 * 1       # w2_p (uint8)
            + NUM_EXPERTS * D * I // 32 * 4 * 1  # w2_s (float32)
        ) / (1024 * 1024)
        print(f"\n    torch.cat (6×{NUM_EXPERTS}): {cat_ms:.3f}ms  "
              f"(≈{total_mb:.0f}MB)")
        assert cat_ms < 5.0, f"unexpectedly slow cat: {cat_ms:.1f}ms"
