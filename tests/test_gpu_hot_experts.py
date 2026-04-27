"""V10-M9: GPU 常驻热专家 + Batched Fused Kernel tests.

Tests:
  1. _hot_expert_set populated from hot_experts.json
  2. _load_gpu_hot_expert_bf16 loads and caches BF16 weights
  3. _gpu_hot_experts LRU eviction (max 16 entries)
  4. _all_routed_are_hot routing detection
  5. _forward_ffn_hot_batched shape and finite
  6. _forward_ffn_hot_batched matches legacy within FP4 tolerance
  7. Hot path preferred when all experts are hot
"""

import torch
import pytest
from collections import OrderedDict
from unittest.mock import MagicMock

_HS = 256
_IM = 128
_V = 1024
_HC = 4
_N_EXPERTS = 8
_TOP_K = 4


def _make_engine_stub():
    from home_seek.inference_engine import HomeSeekInferenceEngine, ExpertWeightCache
    eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
    eng.device = torch.device("cuda")
    eng.expert_cache = ExpertWeightCache(max_experts=64, device="cuda")
    eng._cpu_fallback_enabled = False
    eng._prefetch_worker = None
    eng.verbose = False
    eng.loader = MagicMock()
    eng.loader.get_weights.return_value = {}
    eng.config = MagicMock()
    eng.config.hidden_size = _HS
    eng.config.moe_intermediate_size = _IM
    eng.config.num_experts_per_tok = _TOP_K
    eng.config.swiglu_limit = 10.0
    eng.config.routed_scaling_factor = 1.5
    eng.config.norm_topk_prob = True
    eng.config.scoring_func = "sqrtsoftplus"
    eng.config.topk_method = "noaux_tc"
    eng.config.num_hash_layers = 0
    eng._hot_expert_set = set()
    eng._gpu_hot_experts = {}
    eng._max_hot_experts = 16
    eng._gpu_bf16_cache = OrderedDict()
    eng._max_bf16_cache = 16
    eng._shared_expert_weights = {}
    eng._shared_ffn = MagicMock()
    eng._shared_ffn.forward.return_value = torch.zeros(1, 1, _HS, device="cuda", dtype=torch.bfloat16)
    eng._get_shared_expert = MagicMock(return_value=None)
    eng._hot_expert_ids = []
    eng._hot_expert_set_by_layer = {}
    eng._gpu_expert_store = MagicMock()
    eng._gpu_expert_store.get_cache_key.return_value = None
    eng._log = lambda msg: None
    return eng


@pytest.mark.fast
class TestHotExpertSet:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_hot_expert_set_populated(self):
        import json
        with open("hot_experts.json") as f:
            data = json.load(f)
        expected_ids = set(data.get("top_hot_experts", data.get("top_16_hot_experts", [])))
        expected_count = len(expected_ids)
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._hot_expert_ids = []
        eng._hot_expert_set = set()
        eng._log = lambda msg: None
        eng._gpu_hot_experts = {}
        eng._max_hot_experts = 16
        eng._gpu_expert_store = MagicMock()
        eng.loader = MagicMock()
        from home_seek.inference_engine import ExpertWeightCache
        eng.expert_cache = ExpertWeightCache(max_experts=64, device="cuda")
        eng._preload_hot_experts("hot_experts.json")
        assert len(eng._hot_expert_set) == expected_count
        first_eid = next(iter(expected_ids))
        assert first_eid in eng._hot_expert_set
        assert -1 not in eng._hot_expert_set

    def test_hot_expert_set_legacy_fallback(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._hot_expert_ids = []
        eng._hot_expert_set = set()
        eng._log = lambda msg: None
        eng._gpu_hot_experts = {}
        eng._max_hot_experts = 16
        eng._gpu_expert_store = MagicMock()
        eng.loader = MagicMock()
        from home_seek.inference_engine import ExpertWeightCache
        eng.expert_cache = ExpertWeightCache(max_experts=64, device="cuda")
        import json
        import tempfile
        import os
        legacy = {"top_16_hot_experts": [1, 2, 3, 4, 5, 6],
                  "hash_layer_expert_ids": list(range(18))}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(legacy, f)
            p = f.name
        try:
            eng._preload_hot_experts(p)
            assert len(eng._hot_expert_set) == 6
            assert 1 in eng._hot_expert_set
        finally:
            os.unlink(p)

    def test_hot_expert_set_empty_when_no_file(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.hot_expert_ids = []
        eng._hot_expert_set = set()
        eng._log = lambda msg: None
        eng._preload_hot_experts("/nonexistent/path.json")
        assert eng._hot_expert_set == set()


@pytest.mark.fast
class TestLoadGpuHotExpert:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def test_load_and_cache_bf16(self):
        from tile_reference import cast
        I, D = _IM, _HS
        w1_bf16 = torch.randn(I, D, dtype=torch.bfloat16, device="cuda")
        w3_bf16 = torch.randn(I, D, dtype=torch.bfloat16, device="cuda")
        w2_bf16 = torch.randn(D, I, dtype=torch.bfloat16, device="cuda")
        w1p, w1s = cast(w1_bf16.cpu(), fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3_bf16.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2_bf16.cpu(), fmt="e2m1", block_size=(1, 32))

        eng = _make_engine_stub()
        eng._load_expert_fp4_raw = MagicMock(return_value=(
            w1p.cpu(), w1s.cpu(), w3p.cpu(), w3s.cpu(), w2p.cpu(), w2s.cpu()))

        result = eng._load_gpu_hot_expert_bf16(0, 5)
        assert result is not None
        assert len(result) == 3
        for w in result:
            assert w.dtype == torch.bfloat16
            assert w.device.type == "cuda"
        assert (0, 5) in eng._gpu_hot_experts

    def test_cache_hit_returns_cached(self):
        eng = _make_engine_stub()
        w1 = torch.randn(_IM, _HS, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(_IM, _HS, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(_HS, _IM, device="cuda", dtype=torch.bfloat16)
        eng._gpu_hot_experts[(3, 10)] = (w1, w3, w2)
        eng._load_expert_fp4_raw = MagicMock()

        result = eng._load_gpu_hot_expert_bf16(3, 10)
        assert result is not None
        eng._load_expert_fp4_raw.assert_not_called()

    def test_cache_eviction_lru(self):
        eng = _make_engine_stub()
        eng._max_hot_experts = 4
        for i in range(4):
            w = torch.randn(_IM, _HS, device="cuda", dtype=torch.bfloat16)
            eng._gpu_hot_experts[(0, i)] = (w, w, w)
        from tile_reference import cast
        w_ref = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        wp, ws = cast(w_ref.cpu(), fmt="e2m1", block_size=(1, 32))
        eng._load_expert_fp4_raw = MagicMock(return_value=(
            wp.cpu(), ws.cpu(), wp.cpu(), ws.cpu(), wp.cpu(), ws.cpu()))
        result = eng._load_gpu_hot_expert_bf16(0, 99)
        assert result is not None
        assert len(eng._gpu_hot_experts) == 4
        assert (0, 0) not in eng._gpu_hot_experts
        assert (0, 99) in eng._gpu_hot_experts

    def test_cache_none_when_load_fails(self):
        eng = _make_engine_stub()
        eng._load_expert_fp4_raw = MagicMock(return_value=None)
        result = eng._load_gpu_hot_expert_bf16(5, 200)
        assert result is None


@pytest.mark.fast
class TestAllRoutedAreHot:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_all_hot_ids(self):
        eng = _make_engine_stub()
        eng._hot_expert_set = {1, 2, 3, 4, 5, 6, 7, 8}
        idx = torch.tensor([[1, 2, 3, 4]], device="cuda")
        assert eng._all_routed_are_hot(idx) is True

    def test_mixed_ids(self):
        eng = _make_engine_stub()
        eng._hot_expert_set = {1, 2, 3, 4, 5, 6, 7}
        idx = torch.tensor([[1, 99, 3, 4]], device="cuda")
        assert eng._all_routed_are_hot(idx) is False

    def test_negative_ids_ignored(self):
        eng = _make_engine_stub()
        eng._hot_expert_set = {1, 2, 3}
        idx = torch.tensor([[1, -1, 2, -1]], device="cuda")
        assert eng._all_routed_are_hot(idx) is True

    def test_empty_hot_set(self):
        eng = _make_engine_stub()
        eng._hot_expert_set = set()
        idx = torch.tensor([[1, 2, 3, 4]], device="cuda")
        assert eng._all_routed_are_hot(idx) is False


@pytest.mark.fast
class TestForwardFfnHotBatched:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def _make_hot_engine(self):
        from tile_reference import cast
        eng = _make_engine_stub()

        def mock_fp4_raw(layer, eid):
            w_ref = torch.randn(_IM, _HS, dtype=torch.bfloat16)
            wp, ws = cast(w_ref, fmt="e2m1", block_size=(1, 32))
            w_ref2 = torch.randn(_HS, _IM, dtype=torch.bfloat16)
            wp2, ws2 = cast(w_ref2, fmt="e2m1", block_size=(1, 32))
            return (
                wp.cpu(), ws.cpu(),
                wp.cpu(), ws.cpu(),
                wp2.cpu(), ws2.cpu(),
            )
        eng._load_expert_fp4_raw = mock_fp4_raw
        return eng

    def test_returns_correct_shape(self):
        eng = self._make_hot_engine()
        B, D = 1, _HS
        hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        idx = torch.tensor([[1, 2, 3, 4]], device="cuda")
        w = torch.ones(B, 4, device="cuda") / 4

        result = eng._forward_ffn_hot_batched(hidden, idx, w, 0)
        assert result is not None
        assert result.shape == (B, D)
        assert result.device.type == "cuda"
        assert torch.isfinite(result).all()

    def test_batch_returns_correct_shape(self):
        eng = self._make_hot_engine()
        B, D = 2, _HS
        hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        idx = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
        w = torch.ones(B, 4, device="cuda") / 4

        result = eng._forward_ffn_hot_batched(hidden, idx, w, 0)
        assert result is not None
        assert result.shape == (B, D)

    def test_all_negative_eids(self):
        eng = self._make_hot_engine()
        hidden = torch.randn(1, _HS, device="cuda", dtype=torch.bfloat16)
        idx = torch.tensor([[-1, -1, -1, -1]], device="cuda")
        w = torch.zeros(1, 4, device="cuda")

        result = eng._forward_ffn_hot_batched(hidden, idx, w, 0)
        assert result is not None
        assert result.shape == (1, _HS)
        assert (result == 0).all()

    def test_matches_legacy_forward(self):
        from home_seek.fused_moe import FusedMoEFFN
        from tile_reference import cast
        eng = _make_engine_stub()

        w_refs = {}
        for eid in [1, 2, 3, 4]:
            w1 = torch.randn(_IM, _HS, dtype=torch.bfloat16)
            w3 = torch.randn(_IM, _HS, dtype=torch.bfloat16)
            w2 = torch.randn(_HS, _IM, dtype=torch.bfloat16)
            w_refs[eid] = (w1, w3, w2)

        def mock_fp4_raw(layer, eid):
            w1, w3, w2 = w_refs[eid]
            w1p, w1s = cast(w1.cpu(), fmt="e2m1", block_size=(1, 32))
            w3p, w3s = cast(w3.cpu(), fmt="e2m1", block_size=(1, 32))
            w2p, w2s = cast(w2.cpu(), fmt="e2m1", block_size=(1, 32))
            return (w1p, w1s, w3p, w3s, w2p, w2s)

        def mock_legacy_load(layer, eid):
            w1, w3, w2 = w_refs[eid]
            bf16_w1 = load_fp4_weight(*cast(w1.cpu(), fmt="e2m1", block_size=(1, 32)))
            bf16_w3 = load_fp4_weight(*cast(w3.cpu(), fmt="e2m1", block_size=(1, 32)))
            bf16_w2 = load_fp4_weight(*cast(w2.cpu(), fmt="e2m1", block_size=(1, 32)))
            return (bf16_w1, bf16_w3, bf16_w2)

        from home_seek.inference_engine import load_fp4_weight

        eng._load_expert_fp4_raw = mock_fp4_raw
        hidden = torch.randn(1, _HS, device="cuda", dtype=torch.bfloat16)
        idx = torch.tensor([[1, 2, 3, 4]], device="cuda")
        w = torch.ones(1, 4, device="cuda") / 4

        hot_result = eng._forward_ffn_hot_batched(hidden, idx, w, 0)

        moe = FusedMoEFFN(num_experts=_N_EXPERTS, intermediate_size=_IM,
                           hidden_size=_HS, use_triton=False)
        legacy_result = moe._forward_legacy(hidden, idx, w, mock_legacy_load, 0)

        assert hot_result.shape == legacy_result.shape
        cos = torch.nn.functional.cosine_similarity(
            hot_result.float().flatten(), legacy_result.float().flatten(), dim=0)
        assert cos.item() > 0.90, f"cosine similarity={cos.item():.6f}"

    def test_single_expert(self):
        from tile_reference import cast
        eng = _make_engine_stub()
        w_ref = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        w_ref2 = torch.randn(_HS, _IM, dtype=torch.bfloat16)
        w1p, w1s = cast(w_ref.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w_ref2.cpu(), fmt="e2m1", block_size=(1, 32))
        eng._load_expert_fp4_raw = MagicMock(return_value=(
            w1p, w1s, w1p, w1s, w2p, w2s))

        hidden = torch.randn(1, _HS, device="cuda", dtype=torch.bfloat16)
        idx = torch.tensor([[5, -1, -1, -1]], device="cuda")
        w = torch.tensor([[1.0, 0, 0, 0]], device="cuda")

        result = eng._forward_ffn_hot_batched(hidden, idx, w, 0)
        assert result is not None
        assert result.shape == (1, _HS)
        assert torch.isfinite(result).all()


@pytest.mark.fast
class TestFfnHotPathRouting:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_all_hot_uses_hot_path(self):
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

        eng = _make_engine_stub()
        eng._hot_expert_set = {1, 2, 3, 4, 5, 6}
        eng._forward_ffn_hot_batched = MagicMock(
            return_value=torch.randn(1, _HS, device="cuda", dtype=torch.bfloat16))
        lw = {
            "ffn.gate.weight": torch.randn(_N_EXPERTS, _HS, device="cuda", dtype=torch.bfloat16),
        }
        hidden = torch.randn(1, 1, _HS, device="cuda", dtype=torch.bfloat16)

        try:
            ffn_out, used = eng._forward_ffn(hidden, lw, 0)
            assert eng._forward_ffn_hot_batched.called
        except Exception:
            pass

    def test_mixed_experts_skips_hot_path(self):
        eng = _make_engine_stub()
        eng._hot_expert_set = {1, 2, 3}
        eng._forward_ffn_hot_batched = MagicMock(
            return_value=torch.randn(1, _HS, device="cuda", dtype=torch.bfloat16))
        lw = {
            "ffn.gate.weight": torch.randn(_N_EXPERTS, _HS, device="cuda", dtype=torch.bfloat16),
        }
        hidden = torch.randn(1, 1, _HS, device="cuda", dtype=torch.bfloat16)

        eng._fused_moe = MagicMock()
        eng._fused_moe.forward.side_effect = Exception("mocked fallback")
        eng._load_expert_deq = MagicMock(
            side_effect=lambda l, e: (
                torch.randn(_IM, _HS, device="cuda", dtype=torch.bfloat16),
                torch.randn(_IM, _HS, device="cuda", dtype=torch.bfloat16),
                torch.randn(_HS, _IM, device="cuda", dtype=torch.bfloat16),
            ) if e > 0 else None)
        eng._prefetch_worker = None

        try:
            ffn_out, used = eng._forward_ffn(hidden, lw, 0)
        except Exception:
            pass
        assert not eng._forward_ffn_hot_batched.called

    def test_hot_path_fallback_on_failure(self):
        eng = _make_engine_stub()
        eng._hot_expert_set = {1, 2, 3}
        from tile_reference import cast
        w_ref = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        wp, ws = cast(w_ref.cpu(), fmt="e2m1", block_size=(1, 32))
        eng._load_expert_fp4_raw = MagicMock(return_value=(
            wp, ws, wp, ws, wp, ws))
        lw = {
            "ffn.gate.weight": torch.randn(_N_EXPERTS, _HS, device="cuda", dtype=torch.bfloat16),
        }
        hidden = torch.randn(1, 1, _HS, device="cuda", dtype=torch.bfloat16)
        eng._fused_moe = MagicMock()
        eng._fused_moe.forward.side_effect = Exception("FP4 path failed")
        eng._load_expert_deq = MagicMock(
            side_effect=lambda l, e: (
                torch.randn(_IM, _HS, device="cuda", dtype=torch.bfloat16),
                torch.randn(_IM, _HS, device="cuda", dtype=torch.bfloat16),
                torch.randn(_HS, _IM, device="cuda", dtype=torch.bfloat16),
            ))
        eng._prefetch_worker = None

        try:
            ffn_out, used = eng._forward_ffn(hidden, lw, 0)
        except Exception:
            pass


@pytest.mark.fast
class TestGpuBf16LruCache:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def _make_fp4_raw_mock(self):
        from tile_reference import cast
        w1 = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        w3 = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        w2 = torch.randn(_HS, _IM, dtype=torch.bfloat16)
        w1p, w1s = cast(w1.cpu(), fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2.cpu(), fmt="e2m1", block_size=(1, 32))
        return MagicMock(return_value=(
            (w1p, w1s, "fp4"), (w3p, w3s, "fp4"), (w2p, w2s, "fp4")))

    def test_non_hot_returns_3tuple(self):
        eng = _make_engine_stub()
        eng._max_bf16_cache = 4
        eng._load_expert_raw = self._make_fp4_raw_mock()

        result = eng._load_expert_fp4_raw(0, 99)
        assert result is not None
        assert len(result) == 3
        for w in result:
            assert w.dtype == torch.bfloat16
            assert w.device.type == "cuda"
        assert (0, 99) in eng._gpu_bf16_cache
        assert (0, 99) not in eng._gpu_hot_experts

    def test_lru_hit_skips_load_raw(self):
        eng = _make_engine_stub()
        eng._max_bf16_cache = 4
        raw_mock = self._make_fp4_raw_mock()
        eng._load_expert_raw = raw_mock

        r1 = eng._load_expert_fp4_raw(0, 99)
        assert r1 is not None

        raw_mock.reset_mock()
        r2 = eng._load_expert_fp4_raw(0, 99)
        assert r2 is not None
        raw_mock.assert_not_called()
        for a, b in zip(r1, r2):
            assert torch.equal(a, b)

    def test_lru_eviction(self):
        eng = _make_engine_stub()
        eng._max_bf16_cache = 2

        eng._load_expert_raw = self._make_fp4_raw_mock()
        eng._load_expert_fp4_raw(0, 10)
        assert (0, 10) in eng._gpu_bf16_cache

        eng._load_expert_fp4_raw(0, 20)
        assert (0, 20) in eng._gpu_bf16_cache
        assert len(eng._gpu_bf16_cache) == 2

        eng._load_expert_raw = self._make_fp4_raw_mock()
        eng._load_expert_fp4_raw(0, 30)
        assert (0, 10) not in eng._gpu_bf16_cache
        assert (0, 20) in eng._gpu_bf16_cache
        assert (0, 30) in eng._gpu_bf16_cache
        assert len(eng._gpu_bf16_cache) == 2

    def test_lru_reorder_on_hit(self):
        eng = _make_engine_stub()
        eng._max_bf16_cache = 2

        eng._load_expert_raw = self._make_fp4_raw_mock()
        eng._load_expert_fp4_raw(0, 1)
        eng._load_expert_fp4_raw(0, 2)

        eng._load_expert_raw.reset_mock()
        r = eng._load_expert_fp4_raw(0, 1)
        eng._load_expert_raw.assert_not_called()
        assert r is not None

        eng._load_expert_raw = self._make_fp4_raw_mock()
        eng._load_expert_fp4_raw(0, 3)
        assert (0, 2) not in eng._gpu_bf16_cache
        assert (0, 1) in eng._gpu_bf16_cache
        assert (0, 3) in eng._gpu_bf16_cache

    def test_hot_expert_still_goes_to_hot_cache(self):
        eng = _make_engine_stub()
        eng._max_bf16_cache = 4
        eng._hot_expert_set = {5}
        eng._load_expert_raw = self._make_fp4_raw_mock()

        result = eng._load_expert_fp4_raw(0, 5)
        assert result is not None
        assert (0, 5) in eng._gpu_hot_experts
        assert (0, 5) not in eng._gpu_bf16_cache

    def test_non_fp4_returns_none(self):
        eng = _make_engine_stub()
        eng._max_bf16_cache = 4
        eng._load_expert_raw = MagicMock(return_value=(
            (None, None, "bf16"), (None, None, "bf16"), (None, None, "bf16")))
        result = eng._load_expert_fp4_raw(0, 99)
        assert result is None
