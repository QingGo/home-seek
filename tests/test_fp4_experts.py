import torch
import pytest
from unittest.mock import MagicMock


def _make_engine_stub():
    from home_seek.inference_engine import HomeSeekInferenceEngine, ExpertWeightCache
    eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
    eng.device = torch.device("cuda")
    eng.expert_cache = ExpertWeightCache(max_experts=16, device="cuda")
    eng._cpu_fallback_enabled = False
    eng._prefetch_worker = None
    eng.verbose = False
    eng.loader = MagicMock()
    eng.loader.get_weights.return_value = {}
    return eng


# FP4 model format: packed int8 [R, C/2], scale [R, C/32], deq [R, C]
_FP4_R, _FP4_C = 4, 64


@pytest.mark.fast
class TestRawEntryDevice:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_fp4_raw_stays_on_cpu(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.device = torch.device("cuda")
        data = torch.randint(0, 127, (_FP4_R, _FP4_C // 2), dtype=torch.int8, device="cpu")
        scale = torch.randn(_FP4_R, _FP4_C // 32, device="cpu")
        entry = eng._make_raw_entry(data, scale)
        assert entry is not None
        d, s, fmt = entry
        assert fmt == "fp4"
        assert str(d.device) == "cpu"

    def test_bf16_raw_moves_to_gpu(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.device = torch.device("cuda")
        data = torch.randn(4, 64, dtype=torch.bfloat16, device="cpu")
        entry = eng._make_raw_entry(data, None)
        assert entry is not None
        d, _, fmt = entry
        assert fmt == "bf16"
        assert str(d.device) == "cuda:0"

    def test_fp8_raw_moves_to_gpu(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.device = torch.device("cuda")
        data = torch.ones(4, 64, device="cpu", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale = torch.ones(4, 1, device="cpu")
        entry = eng._make_raw_entry(data, scale)
        assert entry is not None
        d, s, fmt = entry
        assert fmt == "fp8"
        assert str(d.device) == "cuda:0"


@pytest.mark.fast
class TestDequantizeEntry:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_deq_fp4_from_cpu(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=16, device="cuda")
        data = torch.randint(0, 127, (_FP4_R, _FP4_C // 2), dtype=torch.int8, device="cpu")
        scale = torch.ones(_FP4_R, _FP4_C // 32, device="cpu", dtype=torch.float32)
        entry = (data, scale, "fp4")
        result = cache._dequantize_entry(entry, cache.device)
        assert result is not None
        assert result.dtype == torch.bfloat16
        assert str(result.device) == "cuda:0"
        assert result.shape == (_FP4_R, _FP4_C)

    def test_deq_fp8_from_gpu(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=16, device="cuda")
        data = torch.ones(128, 128, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale = torch.tensor([[1.0]], device="cuda")
        entry = (data, scale, "fp8")
        result = cache._dequantize_entry(entry, cache.device)
        assert result is not None
        assert result.dtype == torch.bfloat16
        assert str(result.device) == "cuda:0"

    def test_deq_bf16_passthrough(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=16, device="cuda")
        data = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        entry = (data, None, "bf16")
        result = cache._dequantize_entry(entry, cache.device)
        assert result is data

    def test_deq_none(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=16, device="cuda")
        assert cache._dequantize_entry(None) is None


@pytest.mark.fast
class TestExpertCache:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_put_and_get_fp4(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=16, device="cuda")
        data_cpu = torch.randint(0, 127, (_FP4_R, _FP4_C // 2), dtype=torch.int8, device="cpu")
        scale_cpu = torch.randn(_FP4_R, _FP4_C // 32, device="cpu")
        entry = (data_cpu, scale_cpu, "fp4")
        cache.put("test", entry, entry, entry)
        raw = cache.get("test")
        assert raw is not None
        deq = cache.deq("test")
        assert deq is not None
        for w in deq:
            assert w.dtype == torch.bfloat16
            assert str(w.device) == "cuda:0"

    def test_cache_trim(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=8, device="cuda")
        data = (torch.randn(4, 64, dtype=torch.bfloat16, device="cuda"), None, "bf16")
        for i in range(10):
            cache.put(f"k{i}", data, data, data)
        assert len(cache) <= 8
        assert cache.get("k0") is None
        assert cache.get("k9") is not None


@pytest.mark.fast
class TestFusedMoE:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def _mock_load(self, I=32, D=256):
        return lambda layer, eid: (
            torch.randn(I, D, device="cuda", dtype=torch.bfloat16),
            torch.randn(I, D, device="cuda", dtype=torch.bfloat16),
            torch.randn(D, I, device="cuda", dtype=torch.bfloat16),
        )

    def test_small_batch_returns_correct_shape(self):
        from home_seek.fused_moe import FusedMoEFFN
        moe = FusedMoEFFN(num_experts=8, use_triton=False)
        B, D, topk = 2, 256, 6
        hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, 8, (B, topk), device="cuda")
        w = torch.randn(B, topk, device="cuda").softmax(dim=-1)
        result = moe.forward(hidden, idx, w, self._mock_load(D=D), 0)
        assert result.shape == (B, D)
        assert torch.isfinite(result).all()

    def test_large_batch_not_crash(self):
        from home_seek.fused_moe import FusedMoEFFN
        moe = FusedMoEFFN(num_experts=8, use_triton=False)
        B, D, topk = 16, 256, 6
        hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, 8, (B, topk), device="cuda")
        w = torch.randn(B, topk, device="cuda").softmax(dim=-1)
        result = moe.forward(hidden, idx, w, self._mock_load(D=D), 0)
        assert result.shape == (B, D)

    def test_single_token(self):
        from home_seek.fused_moe import FusedMoEFFN
        moe = FusedMoEFFN(num_experts=8, use_triton=False)
        B, D, topk = 1, 256, 6
        hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, 8, (B, topk), device="cuda")
        w = torch.randn(B, topk, device="cuda").softmax(dim=-1)
        result = moe.forward(hidden, idx, w, self._mock_load(D=D), 0)
        assert result.shape == (B, D)

    def test_swiglu_correctness(self):
        from home_seek.fused_moe import FusedMoEFFN
        moe = FusedMoEFFN(use_triton=False)
        B, I, D = 2, 16, 64
        h = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        activated = moe._swiglu(h, w1, w3)
        gate = h @ w1.t()
        up = h @ w3.t()
        expected = (gate.float().clamp(max=10.0) *
                    gate.float().sigmoid() *
                    up.float().clamp(min=-10.0, max=10.0)).to(torch.bfloat16)
        diff = (activated - expected).abs().max().item()
        assert diff < 0.01


@pytest.mark.fast
class TestExpertFFNPt:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_fused_expert_ffn_pt_shape(self):
        from home_seek.fused_moe import fused_expert_ffn_pt
        B, D, I = 4, 256, 32
        hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(D, I, device="cuda", dtype=torch.bfloat16)
        out = fused_expert_ffn_pt(hidden, w1, w3, w2, 10.0)
        assert out.shape == (B, D)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    def test_fused_expert_ffn_pt_vs_manual(self):
        from home_seek.fused_moe import fused_expert_ffn_pt
        B, D, I = 2, 64, 16
        hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(D, I, device="cuda", dtype=torch.bfloat16)
        out = fused_expert_ffn_pt(hidden, w1, w3, w2, 10.0)
        gate = hidden @ w1.t()
        up = hidden @ w3.t()
        g = gate.float().clamp(max=10.0)
        u = up.float().clamp(min=-10.0, max=10.0)
        expected = ((g * g.sigmoid() * u).to(torch.bfloat16) @ w2.t())
        assert (out - expected).abs().max().item() < 0.01


@pytest.mark.fast
class TestSharedExpert:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_shared_ffn_shape(self):
        from home_seek.fused_moe import SharedExpertFFN
        sffn = SharedExpertFFN(hidden_size=64, intermediate_size=32)
        B, T, D = 2, 4, 64
        h = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(32, D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(32, D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(D, 32, device="cuda", dtype=torch.bfloat16)
        out = sffn.forward(h, w1, w3, w2)
        assert out.shape == (B, T, D)
        assert out.dtype == torch.bfloat16

    def test_shared_ffn_finite(self):
        from home_seek.fused_moe import SharedExpertFFN
        sffn = SharedExpertFFN(hidden_size=64, intermediate_size=32, use_triton=True)
        B, T, D = 1, 2, 64
        h = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(32, D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(32, D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(D, 32, device="cuda", dtype=torch.bfloat16)
        out = sffn.forward(h, w1, w3, w2)
        assert torch.isfinite(out).all()


@pytest.mark.fast
class TestExpertDeq:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_load_expert_deq_none_for_missing(self):
        eng = _make_engine_stub()
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(eng, '_load_expert_weights', lambda li, eid: None)
            assert eng._load_expert_deq(999, 999) is None

    def test_expert_cache_deq_with_device(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=16, device="cuda")
        data_cpu = torch.randint(0, 127, (_FP4_R, _FP4_C // 2), dtype=torch.int8, device="cpu")
        scale_cpu = torch.randn(_FP4_R, _FP4_C // 32, device="cpu")
        cache.put("test", (data_cpu, scale_cpu, "fp4"),
                   (data_cpu, scale_cpu, "fp4"), (data_cpu, scale_cpu, "fp4"))
        result = cache.deq("test")
        assert result is not None
        for w in result:
            assert w.dtype == torch.bfloat16
            assert str(w.device) == "cuda:0"


@pytest.mark.fast
class TestMakeRawEntry:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_none(self):
        eng = _make_engine_stub()
        assert eng._make_raw_entry(None, None) is None

    def test_bf16_to_gpu(self):
        eng = _make_engine_stub()
        data = torch.randn(4, 64, dtype=torch.bfloat16, device="cpu")
        entry = eng._make_raw_entry(data, None)
        assert entry is not None
        assert entry[2] == "bf16"


@pytest.mark.fast
class TestFusedMoEFP4Triton:
    _I = 64
    _D = 128
    _B = 4

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def _make_fp4_weights(self, seed: int = 42):
        torch.manual_seed(seed)
        I, D = self._I, self._D
        w1_bf16 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w3_bf16 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w2_bf16 = torch.randn(D, I, device="cuda", dtype=torch.bfloat16)

        from home_seek._fp4 import cast
        w1_packed, w1_scale = cast(w1_bf16, fmt="e2m1", block_size=(1, 32))
        w3_packed, w3_scale = cast(w3_bf16, fmt="e2m1", block_size=(1, 32))
        w2_packed, w2_scale = cast(w2_bf16, fmt="e2m1", block_size=(1, 32))

        return (w1_bf16, w3_bf16, w2_bf16,
                w1_packed, w1_scale, w3_packed, w3_scale, w2_packed, w2_scale)

    def _deq_ref(self, packed, scale):
        from home_seek._fp4 import unpack_from_e2m1fn_x2
        deq = unpack_from_e2m1fn_x2(packed)
        if scale.dim() == 2:
            sf = scale.repeat_interleave(32, dim=1)
        else:
            sf = scale
        return deq.float() * sf.float()

    def test_fp4_vs_pt_reference_single_token(self):
        from home_seek.fused_moe import fused_expert_ffn_triton, fused_expert_ffn_pt
        (w1_bf16, w3_bf16, w2_bf16,
         w1p, w1s, w3p, w3s, w2p, w2s) = self._make_fp4_weights()

        torch.manual_seed(123)
        hidden = torch.randn(1, self._D, device="cuda", dtype=torch.bfloat16)

        out_triton = fused_expert_ffn_triton(
            hidden, w1p, w3p, w2p, swiglu_limit=10.0,
            w1_scale=w1s, w3_scale=w3s, w2_scale=w2s)
        out_ref = fused_expert_ffn_pt(
            hidden, w1_bf16, w3_bf16, w2_bf16, swiglu_limit=10.0)

        assert out_triton.shape == out_ref.shape == (1, self._D)
        assert torch.isfinite(out_triton).all()
        diff = (out_triton.float() - out_ref.float()).abs().max().item()
        ref_norm = out_ref.float().abs().max().item()
        assert diff < max(1.0, ref_norm * 0.5), \
            f"diff={diff:.4f} ref_max={ref_norm:.4f}"

    def test_fp4_vs_pt_reference_batch(self):
        from home_seek.fused_moe import fused_expert_ffn_triton, fused_expert_ffn_pt
        (w1_bf16, w3_bf16, w2_bf16,
         w1p, w1s, w3p, w3s, w2p, w2s) = self._make_fp4_weights()

        torch.manual_seed(456)
        hidden = torch.randn(self._B, self._D, device="cuda", dtype=torch.bfloat16)

        out_triton = fused_expert_ffn_triton(
            hidden, w1p, w3p, w2p, swiglu_limit=10.0,
            w1_scale=w1s, w3_scale=w3s, w2_scale=w2s)
        out_ref = fused_expert_ffn_pt(
            hidden, w1_bf16, w3_bf16, w2_bf16, swiglu_limit=10.0)

        diff = (out_triton.float() - out_ref.float()).abs().max().item()
        ref_norm = out_ref.float().abs().max().item()
        assert out_triton.shape == out_ref.shape
        assert torch.isfinite(out_triton).all()
        assert diff < max(2.0, ref_norm * 0.5), \
            f"diff={diff:.4f} ref_max={ref_norm:.4f}"

    def test_fp4_vs_deq_manual_reference(self):
        from home_seek.fused_moe import fused_expert_ffn_pt
        (w1_bf16, w3_bf16, w2_bf16,
         w1p, w1s, w3p, w3s, w2p, w2s) = self._make_fp4_weights()

        w1_deq = self._deq_ref(w1p, w1s).to(torch.bfloat16)
        w3_deq = self._deq_ref(w3p, w3s).to(torch.bfloat16)
        w2_deq = self._deq_ref(w2p, w2s).to(torch.bfloat16)

        hidden = torch.randn(2, self._D, device="cuda", dtype=torch.bfloat16)

        out_triton_fp4 = fused_expert_ffn_pt(
            hidden, w1_deq, w3_deq, w2_deq, swiglu_limit=10.0)
        out_ref_orig = fused_expert_ffn_pt(
            hidden, w1_bf16, w3_bf16, w2_bf16, swiglu_limit=10.0)

        cos_fp4 = torch.nn.functional.cosine_similarity(
            out_ref_orig.float().flatten(), out_triton_fp4.float().flatten(), dim=0)
        assert cos_fp4.item() > 0.95, f"FP4 dequantized vs original: {cos_fp4.item():.6f}"

    def test_fp4_swiglu_clamp(self):
        from home_seek.fused_moe import fused_expert_ffn_triton
        (w1_bf16, w3_bf16, w2_bf16,
         w1p, w1s, w3p, w3s, w2p, w2s) = self._make_fp4_weights()

        hidden = torch.randn(4, self._D, device="cuda", dtype=torch.bfloat16) * 3.0

        out = fused_expert_ffn_triton(
            hidden, w1p, w3p, w2p, swiglu_limit=2.0,
            w1_scale=w1s, w3_scale=w3s, w2_scale=w2s)
        assert out.shape == (4, self._D)
        assert torch.isfinite(out).all()

    def test_fp4_fuses_gate_up_computation(self):
        from home_seek.fused_moe import fused_expert_ffn_triton
        (w1_bf16, w3_bf16, w2_bf16,
         w1p, w1s, w3p, w3s, w2p, w2s) = self._make_fp4_weights()

        torch.manual_seed(789)
        hidden = torch.randn(1, self._D, device="cuda", dtype=torch.bfloat16)

        out_triton = fused_expert_ffn_triton(
            hidden, w1p, w3p, w2p, swiglu_limit=10.0,
            w1_scale=w1s, w3_scale=w3s, w2_scale=w2s)

        gate = hidden @ w1_bf16.t()
        up = hidden @ w3_bf16.t()
        g = gate.float().clamp(max=10.0)
        u = up.float().clamp(min=-10.0, max=10.0)
        activated = (g * g.sigmoid() * u).to(w1_bf16.dtype)
        out_ref = activated @ w2_bf16.t()

        diff = (out_triton.float() - out_ref.float()).abs().max().item()
        ref_norm = out_ref.float().abs().max().item()
        assert diff < max(2.0, ref_norm * 0.5), \
            f"FP4 Triton max diff vs manual: {diff:.4f} ref_max={ref_norm:.4f}"

    def test_fp4_wrong_dtype_graceful(self):
        from home_seek.fused_moe import fused_expert_ffn_triton
        hidden = torch.randn(2, self._D, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(self._I, self._D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(self._D, self._I, device="cuda", dtype=torch.bfloat16)

        out = fused_expert_ffn_triton(hidden, w1, w1, w2,
                                       swiglu_limit=10.0)
        assert out.shape == (2, self._D)


# ──────────────────────────────────────────────────────────────────────
# 回归测试：固化 V10-M4 FP4+GEMM 修复中遇到的 bug
# ──────────────────────────────────────────────────────────────────────


class TestFP4DequantizePitfalls:
    """Bugs fixed:
    - tl.join + tl.reshape DOES correctly interleave (was wrongly believed not to).
    - BLOCK_K_HALF must be <=16 so one tile covers one scale group (32 cols).
    - tl.dot needs [BK, BN] layout, not [BN, BK]; use tl.trans after interleave.
    """

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_interleave_via_join_reshape(self):
        """Bug: 曾认为 tl.join([N,K,1],[N,K,1]) reshape [N,2K] 不能交错。
        实际验证: 结果正确交替 lo/hi。"""
        import triton
        import triton.language as tl

        @triton.jit
        def _interleave_test(lo_ptr, hi_ptr, out_ptr, N, K,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            pid = tl.program_id(0)
            offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            mask_n = offs_n < N
            mask_k = offs_k < K
            lo = tl.load(lo_ptr + offs_n[:, None] * K + offs_k[None, :],
                         mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            hi = tl.load(hi_ptr + offs_n[:, None] * K + offs_k[None, :],
                         mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            interleaved = tl.reshape(
                tl.join(
                    tl.reshape(lo.to(tl.bfloat16), (BLOCK_N, BLOCK_K, 1)),
                    tl.reshape(hi.to(tl.bfloat16), (BLOCK_N, BLOCK_K, 1)),
                ),
                (BLOCK_N, 2 * BLOCK_K),
            )
            offs_out = tl.arange(0, 2 * BLOCK_K)
            mask_out = (offs_n[:, None] < N) & (offs_out[None, :] < 2 * K)
            tl.store(out_ptr + offs_n[:, None] * (2 * K) + offs_out[None, :],
                     interleaved, mask=mask_out)

        N, K = 4, 8  # powers of 2 for tl.arange
        N_real = 3
        lo = torch.arange(N_real * K, dtype=torch.float32).view(N_real, K).cuda()
        hi = torch.arange(100, 100 + N_real * K, dtype=torch.float32).view(N_real, K).cuda()
        out = torch.zeros(N, 2 * K, dtype=torch.bfloat16, device="cuda")

        _interleave_test[(1,)](lo, hi, out, N_real, K, BLOCK_N=N, BLOCK_K=K)
        torch.cuda.synchronize()

        for n in range(N_real):
            for k in range(K):
                assert out[n, 2 * k] == lo[n, k], \
                    f"interleave[{n},{2*k}]={out[n,2*k]} != lo[{n},{k}]={lo[n,k]}"
                assert out[n, 2 * k + 1] == hi[n, k], \
                    f"interleave[{n},{2*k+1}]={out[n,2*k+1]} != hi[{n},{k}]={hi[n,k]}"

    def test_deq_multi_scale_groups(self):
        """Bug: BLOCK_K_HALF=32 跨越 2 个 scale group, 只加载了一个 scale,
        导致一半值使用错误 scale。修复后 BLOCK_K_HALF=16 (单 group) 应无此问题。"""
        from home_seek.fused_moe import triton_dequantize_fp4_to_bf16
        from home_seek._fp4 import cast, unpack_from_e2m1fn_x2

        I, D = 8, 128  # 128 cols => 4 scale groups (128/32=4)
        torch.manual_seed(123)
        w = torch.randn(I, D, dtype=torch.bfloat16)
        w_packed, w_scale = cast(w, fmt="e2m1", block_size=(1, 32))
        w_packed_c = w_packed.cuda()
        w_scale_c = w_scale.to(torch.float32).cuda()

        deq_triton = triton_dequantize_fp4_to_bf16(w_packed_c, w_scale_c)

        deq_ref = unpack_from_e2m1fn_x2(w_packed_c)
        sf_ref = w_scale_c.repeat_interleave(32, dim=1)
        deq_ref = (deq_ref.float() * sf_ref.float()).to(torch.bfloat16)

        diff = (deq_triton.float() - deq_ref.float()).abs().max().item()
        ref_max = deq_ref.float().abs().max().item()
        assert diff < 0.1, \
            f"Multi-scale-group deq diff={diff:.6f} ref_max={ref_max:.4f}"

    def test_fused_gemm_tl_dot_layout(self):
        """Bug: fused kernel 输出 [BN, BK] 但 tl.dot 需要 [BK, BN] (column-major)。
        修复后经 tl.trans 转为 [BK, BN]。本测试验证 gate 投影与 PyTorch 一致。"""
        import triton
        import triton.language as tl
        from home_seek._fp4 import cast
        from home_seek.fused_moe import _FP4_LUT

        @triton.jit
        def _mini_gate_kernel(
            hidden_ptr, w_packed_ptr, w_scale_ptr, out_ptr,
            M, I, D, stride_hm, stride_hk,
            stride_wn, stride_wk, stride_sn, stride_sk,
            stride_om, stride_on, lut_ptr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BM + tl.arange(0, BM)
            offs_n = pid_n * BN + tl.arange(0, BN)
            offs_k = tl.arange(0, BK)
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            BK_HALF: tl.constexpr = BK // 2

            for k in range(0, D, BK):
                mask_k = (k + offs_k) < D
                mask_m = offs_m < M
                h = tl.load(
                    hidden_ptr + offs_m[:, None] * stride_hm + (k + offs_k)[None, :] * stride_hk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                k_half = k // 2
                offs_kh = k_half + tl.arange(0, BK_HALF)

                wp = tl.load(
                    w_packed_ptr + offs_n[:, None] * stride_wn + offs_kh[None, :] * stride_wk,
                    mask=(offs_n[:, None] < I) & (offs_kh[None, :] < (D // 2)), other=0).to(tl.uint8)
                lo = (wp & 0xF).to(tl.int32)
                hi = ((wp >> 4) & 0xF).to(tl.int32)
                lo_f32 = tl.load(lut_ptr + lo).to(tl.float32)
                hi_f32 = tl.load(lut_ptr + hi).to(tl.float32)
                sg = k // 32
                s = tl.load(
                    w_scale_ptr + offs_n[:, None] * stride_sn + sg * stride_sk,
                    mask=offs_n[:, None] < I, other=1.0).to(tl.float32)
                lo_f32 *= s
                hi_f32 *= s

                w_inter = tl.reshape(
                    tl.join(
                        tl.reshape(lo_f32.to(tl.bfloat16), (BN, BK_HALF, 1)),
                        tl.reshape(hi_f32.to(tl.bfloat16), (BN, BK_HALF, 1)),
                    ),
                    (BN, BK),
                )
                w_for_dot = tl.trans(w_inter, 1, 0)  # [BK, BN] ← 关键修复
                acc += tl.dot(h, w_for_dot)

            mask_m = offs_m[:, None] < M
            mask_n = offs_n[None, :] < I
            tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                     acc.to(out_ptr.dtype.element_ty),
                     mask=mask_m & mask_n)

        # Test params
        I_s, D_s = 32, 64
        torch.manual_seed(42)
        w = torch.randn(I_s, D_s, dtype=torch.bfloat16)
        w_p, w_s = cast(w, fmt="e2m1", block_size=(1, 32))
        w_p_c, w_s_c = w_p.cuda(), w_s.to(torch.float32).cuda()

        hidden = torch.randn(2, D_s, device="cuda", dtype=torch.bfloat16)
        lut = _FP4_LUT.cuda()

        out = torch.zeros(2, I_s, device="cuda", dtype=torch.bfloat16)
        BM, BN, BK = 16, 16, 32  # tl.dot needs N >= 16

        _mini_gate_kernel[(triton.cdiv(2, BM), triton.cdiv(I_s, BN))](
            hidden, w_p_c, w_s_c, out, 2, I_s, D_s,
            hidden.stride(0), hidden.stride(1),
            w_p_c.stride(0), w_p_c.stride(1),
            w_s_c.stride(0), w_s_c.stride(1),
            out.stride(0), out.stride(1), lut,
            BM=BM, BN=BN, BK=BK, num_stages=1,
        )
        torch.cuda.synchronize()

        # Reference: Python dequantize + matmul
        from home_seek._fp4 import unpack_from_e2m1fn_x2
        deq_w = (unpack_from_e2m1fn_x2(w_p_c).float() * w_s_c.repeat_interleave(32, dim=1).float()).to(torch.bfloat16)
        ref = hidden @ deq_w.t()

        diff = (out.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        assert diff < max(1.0, ref_max * 0.1), \
            f"tl.dot layout bug: gate diff={diff:.4f} ref_max={ref_max:.4f}"

    def test_scale_per_group_isolated(self):
        """Bug: 校验 per-group scale 在 dequantize 中正确应用。
        构造各 group 不同 scale 的权重 (1, 2, 3, 4...) 并验证反量化结果。"""
        from home_seek.fused_moe import triton_dequantize_fp4_to_bf16

        I, D = 4, 128  # 4 scale groups (128/32=4)
        torch.manual_seed(7)
        w_packed = torch.randint(0, 127, (I, D // 2), dtype=torch.int8, device="cuda")
        # Distinct scales per group: [1, 2, 3, 4] for each of 4 groups
        w_scale = torch.tensor([
            [1.0, 2.0, 3.0, 4.0] for _ in range(I)
        ], device="cuda", dtype=torch.float32)

        result = triton_dequantize_fp4_to_bf16(w_packed, w_scale)

        # Verify: group 0 columns (0..31) use scale 1, group 1 (32..63) use 2, etc.
        for g in range(4):
            col_start = g * 32
            col_slice = result[:, col_start:col_start + 32]
            # The dequantized values should have magnitude proportional to scale
            mean_abs = col_slice.float().abs().mean().item()
            # Different groups with different scales should differ
            assert mean_abs > 0, f"Group {g} has zero values"

        # Check that groups have proportional magnitudes
        g0 = result[:, 0:32].float().abs().mean().item()
        g2 = result[:, 64:96].float().abs().mean().item()
        # g2 should be ~3x g0 (scale 3x)
        assert g2 > g0 * 1.5, f"Scale group mismatch: g0={g0:.4f}, g2={g2:.4f}"


# ──────────────────────────────────────────────────────────────────────
# V10-M7: PCIe BAR 修复验证
# ──────────────────────────────────────────────────────────────────────


class TestPCIeBARFix:
    """Verify that _forward_legacy explicitly DMAs FP4 weights to GPU
    before Triton dequantize, avoiding the 0.2 GB/s PCIe BAR path."""

    _I = 32
    _D = 64
    _B = 2
    _topk = 4

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def _make_fp4_cpu_weights(self):
        """Create FP4-packed weights on CPU, simulating mmap-loaded weights."""
        from home_seek._fp4 import cast
        I, D = self._I, self._D
        w1_bf16 = torch.randn(I, D, dtype=torch.bfloat16)
        w3_bf16 = torch.randn(I, D, dtype=torch.bfloat16)
        w2_bf16 = torch.randn(D, I, dtype=torch.bfloat16)

        w1p, w1s = cast(w1_bf16, fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3_bf16, fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2_bf16, fmt="e2m1", block_size=(1, 32))

        # Move to CPU to simulate the real scenario
        return (
            w1p.cpu(), w1s.cpu(), w3p.cpu(), w3s.cpu(), w2p.cpu(), w2s.cpu()
        )

    def test_legacy_with_cpu_fp4_weights(self):
        """CPU FP4 weights in _forward_legacy → DMA to GPU → correct output."""
        from home_seek.fused_moe import FusedMoEFFN

        moe = FusedMoEFFN(num_experts=4, intermediate_size=self._I,
                          hidden_size=self._D, use_triton=True)

        w1p_c, w1s_c, w3p_c, w3s_c, w2p_c, w2s_c = self._make_fp4_cpu_weights()

        # All weights on CPU initially
        for t in [w1p_c, w1s_c, w3p_c, w3s_c, w2p_c, w2s_c]:
            assert str(t.device) == "cpu"

        def mock_load(layer, eid):
            return (w1p_c, w1s_c, w3p_c, w3s_c, w2p_c, w2s_c)

        hidden = torch.randn(self._B, self._D, device="cuda", dtype=torch.bfloat16)
        idx = torch.full((self._B, self._topk), 0, device="cuda", dtype=torch.long)
        weights = torch.ones(self._B, self._topk, device="cuda") / self._topk

        result = moe.forward(hidden, idx, weights, mock_load, 0)

        assert result.shape == (self._B, self._D)
        assert result.device.type == "cuda"
        assert torch.isfinite(result).all()

    def test_legacy_with_mixed_cpu_gpu_weights(self):
        """Some weights on CPU, some on GPU — all end up on GPU."""
        from home_seek.fused_moe import FusedMoEFFN

        moe = FusedMoEFFN(num_experts=4, intermediate_size=self._I,
                          hidden_size=self._D, use_triton=True)

        w1p_c, w1s_c, w3p_c, w3s_c, w2p_c, w2s_c = self._make_fp4_cpu_weights()

        # Put some on GPU, some on CPU to test mixed case
        w1p_g = w1p_c.cuda()
        w1s_g = w1s_c.cuda()

        # w1 on GPU, w3/w2 on CPU
        assert str(w1p_g.device) == "cuda:0"
        assert str(w3p_c.device) == "cpu"

        def mock_load(layer, eid):
            return (w1p_g, w1s_g, w3p_c, w3s_c, w2p_c, w2s_c)

        hidden = torch.randn(self._B, self._D, device="cuda", dtype=torch.bfloat16)
        idx = torch.full((self._B, self._topk), 0, device="cuda", dtype=torch.long)
        weights = torch.ones(self._B, self._topk, device="cuda") / self._topk

        result = moe.forward(hidden, idx, weights, mock_load, 0)

        assert result.shape == (self._B, self._D)
        assert result.device.type == "cuda"
        assert torch.isfinite(result).all()

    def test_legacy_with_cpu_bf16_weights(self):
        """Already-dequantized BF16 weights on CPU → moved to GPU in 3-tuple path."""
        from home_seek.fused_moe import FusedMoEFFN

        moe = FusedMoEFFN(num_experts=4, intermediate_size=self._I,
                          hidden_size=self._D, use_triton=True)

        w1 = torch.randn(self._I, self._D, dtype=torch.bfloat16, device="cpu")
        w3 = torch.randn(self._I, self._D, dtype=torch.bfloat16, device="cpu")
        w2 = torch.randn(self._D, self._I, dtype=torch.bfloat16, device="cpu")

        def mock_load(layer, eid):
            return (w1, w3, w2)

        hidden = torch.randn(self._B, self._D, device="cuda", dtype=torch.bfloat16)
        idx = torch.full((self._B, self._topk), 0, device="cuda", dtype=torch.long)
        weights = torch.ones(self._B, self._topk, device="cuda") / self._topk

        result = moe.forward(hidden, idx, weights, mock_load, 0)

        assert result.shape == (self._B, self._D)
        assert result.device.type == "cuda"
        assert torch.isfinite(result).all()

    def test_legacy_multi_expert_cpu_fp4(self):
        """Multiple experts with CPU FP4 weights → batched correctly."""
        from home_seek.fused_moe import FusedMoEFFN

        moe = FusedMoEFFN(num_experts=4, intermediate_size=self._I,
                          hidden_size=self._D, use_triton=True)

        def mock_load(layer, eid):
            from home_seek._fp4 import cast
            w1_bf16 = torch.randn(self._I, self._D, dtype=torch.bfloat16)
            w3_bf16 = torch.randn(self._I, self._D, dtype=torch.bfloat16)
            w2_bf16 = torch.randn(self._D, self._I, dtype=torch.bfloat16)
            w1p, w1s = cast(w1_bf16, fmt="e2m1", block_size=(1, 32))
            w3p, w3s = cast(w3_bf16, fmt="e2m1", block_size=(1, 32))
            w2p, w2s = cast(w2_bf16, fmt="e2m1", block_size=(1, 32))
            return (w1p.cpu(), w1s.cpu(), w3p.cpu(), w3s.cpu(), w2p.cpu(), w2s.cpu())

        hidden = torch.randn(self._B, self._D, device="cuda", dtype=torch.bfloat16)
        # Each token routes to different experts
        idx = torch.tensor([[0, 2, -1, -1], [1, 3, -1, -1]], device="cuda")
        weights = torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]], device="cuda")

        result = moe.forward(hidden, idx, weights, mock_load, 0)

        assert result.shape == (self._B, self._D)
        assert result.device.type == "cuda"
        assert torch.isfinite(result).all()

    def test_triton_deq_uses_gpu_path_when_cpu_moved(self):
        """After DMA, triton_dequantize_fp4_to_bf16 uses GPU Triton kernel."""
        from home_seek.fused_moe import triton_dequantize_fp4_to_bf16
        from home_seek._fp4 import cast

        I, D = 16, 64
        w_bf16 = torch.randn(I, D, dtype=torch.bfloat16)
        w_packed, w_scale = cast(w_bf16, fmt="e2m1", block_size=(1, 32))

        # Simulate what _forward_legacy now does: DMA to GPU
        w_packed_c = w_packed.cpu()
        w_scale_c = w_scale.cpu()

        device = torch.device("cuda")
        w_packed_g = w_packed_c.to(device, non_blocking=True)
        w_scale_g = w_scale_c.to(device, non_blocking=True)

        result = triton_dequantize_fp4_to_bf16(w_packed_g, w_scale_g)

        assert result.device.type == "cuda"
        assert result.shape == (I, D)
        assert result.dtype == torch.bfloat16
        assert torch.isfinite(result).all()

    def test_triton_deq_cpu_fallback_still_works(self):
        """CPU deq fallback (without DMA) still produces correct result."""
        from home_seek.fused_moe import triton_dequantize_fp4_to_bf16
        from home_seek._fp4 import cast

        I, D = 8, 64
        w_bf16 = torch.randn(I, D, dtype=torch.bfloat16)
        w_packed, w_scale = cast(w_bf16, fmt="e2m1", block_size=(1, 32))

        # CPU path: no DMA, the function falls back to tile_reference
        result = triton_dequantize_fp4_to_bf16(w_packed.cpu(), w_scale.cpu())

        assert result.device.type == "cpu"
        assert result.shape == (I, D)
        assert result.dtype == torch.bfloat16
