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

        from tile_reference import cast
        w1_packed, w1_scale = cast(w1_bf16, fmt="e2m1", block_size=(1, 32))
        w3_packed, w3_scale = cast(w3_bf16, fmt="e2m1", block_size=(1, 32))
        w2_packed, w2_scale = cast(w2_bf16, fmt="e2m1", block_size=(1, 32))

        return (w1_bf16, w3_bf16, w2_bf16,
                w1_packed, w1_scale, w3_packed, w3_scale, w2_packed, w2_scale)

    def _deq_ref(self, packed, scale):
        from tile_reference import unpack_from_e2m1fn_x2
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
        from home_seek.fused_moe import fused_expert_ffn_triton, fused_expert_ffn_pt
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
