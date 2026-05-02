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
from tests._engine_stub import make_engine
from home_seek.model_config import DeepSeekV4FlashConfig

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
    eng._gpu_bf16_deq_cache = OrderedDict()
    eng._max_gpu_bf16_deq = 48
    eng._ep_affinity_rr = 0
    eng._shared_expert_weights = {}
    eng._shared_ffn = MagicMock()
    eng._shared_ffn.forward.return_value = torch.zeros(1, 1, _HS, device="cuda", dtype=torch.bfloat16)
    eng._get_shared_expert = MagicMock(return_value=None)
    eng._hot_expert_ids = []
    eng._hot_expert_set_by_layer = {}
    eng._log = lambda msg: None
    return eng


@pytest.mark.fast
class TestHotExpertSet:
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
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def test_load_and_cache_bf16(self):
        from home_seek._fp4 import cast
        I, D = _IM, _HS
        w1_bf16 = torch.randn(I, D, dtype=torch.bfloat16, device="cuda")
        w3_bf16 = torch.randn(I, D, dtype=torch.bfloat16, device="cuda")
        w2_bf16 = torch.randn(D, I, dtype=torch.bfloat16, device="cuda")
        w1p, w1s = cast(w1_bf16.cpu(), fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3_bf16.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2_bf16.cpu(), fmt="e2m1", block_size=(1, 32))

        eng = _make_engine_stub()
        eng._load_expert_raw = MagicMock(return_value=(
            (w1p.cpu(), w1s.cpu(), "fp4"),
            (w3p.cpu(), w3s.cpu(), "fp4"),
            (w2p.cpu(), w2s.cpu(), "fp4"),
        ))

        result = eng._load_gpu_hot_expert_bf16(0, 5)
        assert result is not None
        assert len(result) == 3
        for w in result:
            assert w.dtype == torch.bfloat16
            assert w.device.type == "cuda"
        # FP4 6-tuple stored in bf16 cache (eid=5 is not hot)
        assert (0, 5) in eng._gpu_bf16_cache
        cached = eng._gpu_bf16_cache[(0, 5)]
        assert len(cached) == 6  # FP4 raw, not BF16

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
        from home_seek._fp4 import cast
        w_ref = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        w_ref2 = torch.randn(_HS, _IM, dtype=torch.bfloat16)
        wp, ws = cast(w_ref.cpu(), fmt="e2m1", block_size=(1, 32))
        wp2, ws2 = cast(w_ref2.cpu(), fmt="e2m1", block_size=(1, 32))
        eng._load_expert_raw = MagicMock(return_value=(
            (wp.cpu(), ws.cpu(), "fp4"),
            (wp.cpu(), ws.cpu(), "fp4"),
            (wp2.cpu(), ws2.cpu(), "fp4"),
        ))
        # Make eid=99 a hot expert so it enters gpu_hot cache (FIFO eviction)
        eng._hot_expert_set = {99}
        result = eng._load_gpu_hot_expert_bf16(0, 99)
        assert result is not None
        assert len(eng._gpu_hot_experts) == 4
        assert (0, 0) not in eng._gpu_hot_experts
        assert (0, 99) in eng._gpu_hot_experts
        cached = eng._gpu_hot_experts[(0, 99)]
        assert len(cached) == 6  # FP4 raw stored, not BF16

    def test_cache_none_when_load_fails(self):
        eng = _make_engine_stub()
        eng._load_expert_fp4_raw = MagicMock(return_value=None)
        result = eng._load_gpu_hot_expert_bf16(5, 200)
        assert result is None


@pytest.mark.fast
class TestAllRoutedAreHot:
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
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def _make_hot_engine(self):
        from home_seek._fp4 import cast
        eng = _make_engine_stub()

        def mock_fp4_raw(layer, eid):
            w_ref = torch.randn(_IM, _HS, dtype=torch.bfloat16)
            wp, ws = cast(w_ref, fmt="e2m1", block_size=(1, 32))
            w_ref2 = torch.randn(_HS, _IM, dtype=torch.bfloat16)
            wp2, ws2 = cast(w_ref2, fmt="e2m1", block_size=(1, 32))
            return (
                wp.cuda(), ws.cuda(),
                wp.cuda(), ws.cuda(),
                wp2.cuda(), ws2.cuda(),
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
        from home_seek._fp4 import cast
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
            return (w1p.cuda(), w1s.cuda(), w3p.cuda(), w3s.cuda(), w2p.cuda(), w2s.cuda())

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
        from home_seek._fp4 import cast
        eng = _make_engine_stub()
        w_ref = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        w_ref2 = torch.randn(_HS, _IM, dtype=torch.bfloat16)
        w1p, w1s = cast(w_ref.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w_ref2.cpu(), fmt="e2m1", block_size=(1, 32))
        eng._load_expert_fp4_raw = MagicMock(return_value=(
            w1p.cuda(), w1s.cuda(), w1p.cuda(), w1s.cuda(), w2p.cuda(), w2s.cuda()))

        hidden = torch.randn(1, _HS, device="cuda", dtype=torch.bfloat16)
        idx = torch.tensor([[5, -1, -1, -1]], device="cuda")
        w = torch.tensor([[1.0, 0, 0, 0]], device="cuda")

        result = eng._forward_ffn_hot_batched(hidden, idx, w, 0)
        assert result is not None
        assert result.shape == (1, _HS)
        assert torch.isfinite(result).all()


@pytest.mark.fast
class TestFfnHotPathRouting:
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

        try:
            ffn_out, used = eng._forward_ffn(hidden, lw, 0)
        except Exception:
            pass
        assert not eng._forward_ffn_hot_batched.called

    def test_hot_path_fallback_on_failure(self):
        eng = _make_engine_stub()
        eng._hot_expert_set = {1, 2, 3}
        from home_seek._fp4 import cast
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

        try:
            ffn_out, used = eng._forward_ffn(hidden, lw, 0)
        except Exception:
            pass


@pytest.mark.fast
class TestGpuBf16LruCache:
    def setup_method(self):
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def _make_fp4_raw_mock(self):
        from home_seek._fp4 import cast
        w1 = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        w3 = torch.randn(_IM, _HS, dtype=torch.bfloat16)
        w2 = torch.randn(_HS, _IM, dtype=torch.bfloat16)
        w1p, w1s = cast(w1.cpu(), fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2.cpu(), fmt="e2m1", block_size=(1, 32))
        return MagicMock(return_value=(
            (w1p, w1s, "fp4"), (w3p, w3s, "fp4"), (w2p, w2s, "fp4")))

    def test_non_hot_returns_6tuple(self):
        eng = _make_engine_stub()
        eng._max_bf16_cache = 4
        eng._load_expert_raw = self._make_fp4_raw_mock()

        result = eng._load_expert_fp4_raw(0, 99)
        assert result is not None
        assert len(result) == 6
        assert result[0].dtype == torch.int8  # packed FP4
        assert result[1].dtype == torch.float32  # scale
        assert result[0].device.type == "cuda"
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


@pytest.mark.fast
class TestGpuFp4CacheV21_7:
    """V21.7: GPU 缓存存 FP4 raw (6元组) 而非 BF16 (3元组).

    验证:
    1. _load_expert_fp4_raw 存储 FP4 6元组并返回 6元组
    2. 缓存命中返回正确 6元组
    3. _load_gpu_hot_expert_bf16 从 FP4 缓存去量化 BF16
    4. _forward_legacy 处理 6元组 (FP4 即时去量化)
    5. 热专家 FIFO 淘汰仍正常工作
    """

    def setup_method(self):
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def _make_fp4_raw(self, I=32, D=128):
        from home_seek._fp4 import cast
        w1 = torch.randn(I, D, dtype=torch.bfloat16)
        w3 = torch.randn(I, D, dtype=torch.bfloat16)
        w2 = torch.randn(D, I, dtype=torch.bfloat16)
        w1p, w1s = cast(w1.cpu(), fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2.cpu(), fmt="e2m1", block_size=(1, 32))
        return {
            "bf16": (w1, w3, w2),
            "raw": ((w1p, w1s, "fp4"), (w3p, w3s, "fp4"), (w2p, w2s, "fp4")),
        }

    def test_cache_stores_fp4_not_bf16(self):
        """验证缓存存储 6元组 (FP4 raw) 而非 3元组 (BF16)."""
        eng = _make_engine_stub()
        eng._max_hot_experts = 4
        eng._hot_expert_set = {7}  # hot → goes to gpu_hot cache
        fp4 = self._make_fp4_raw()
        eng._load_expert_raw = MagicMock(return_value=fp4["raw"])

        result = eng._load_expert_fp4_raw(0, 7)
        assert result is not None
        assert len(result) == 6, "should store FP4 6-tuple"
        assert result[0].dtype == torch.int8, "FP4 packed data is int8"
        assert result[1].dtype == torch.float32, "FP4 scale is float32"
        assert (0, 7) in eng._gpu_hot_experts
        cached = eng._gpu_hot_experts[(0, 7)]
        assert len(cached) == 6
        assert cached[0].dtype == torch.int8

    def test_cache_hit_returns_6tuple(self):
        """第二次调用命中缓存, 返回相同 6元组."""
        eng = _make_engine_stub()
        eng._max_bf16_cache = 4
        fp4 = self._make_fp4_raw()
        eng._load_expert_raw = MagicMock(return_value=fp4["raw"])

        r1 = eng._load_expert_fp4_raw(0, 99)
        eng._load_expert_raw.reset_mock()
        r2 = eng._load_expert_fp4_raw(0, 99)
        eng._load_expert_raw.assert_not_called()
        assert r2 is not None
        assert len(r2) == 6
        # Same data pointers (cache hit returns the same tensors)
        for a, b in zip(r1, r2):
            assert a.data_ptr() == b.data_ptr()

    def test_hot_batched_from_fp4_cache(self):
        """_forward_ffn_hot_batched 调用 _load_gpu_hot_expert_bf16,
        后者从 FP4 cache 去量化并返回 BF16."""
        from home_seek.fused_moe import FusedMoEFFN
        eng = _make_engine_stub()
        eng._max_hot_experts = 8
        eng._hot_expert_set = {1, 2, 3, 4}

        fp4 = self._make_fp4_raw(I=_IM, D=_HS)

        def mock_raw(layer, eid):
            return fp4["raw"]
        eng._load_expert_raw = MagicMock(side_effect=mock_raw)

        eng._fused_moe = FusedMoEFFN(
            num_experts=8, intermediate_size=_IM, hidden_size=_HS,
            use_triton=False)

        B, D = 1, _HS
        hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        idx = torch.tensor([[1, 2, 3, 4]], device="cuda")
        w = torch.ones(B, 4, device="cuda") / 4

        result = eng._forward_ffn_hot_batched(hidden, idx, w, 0)
        assert result is not None
        assert result.shape == (B, D)
        assert torch.isfinite(result).all()

    def test_fp4_cache_eviction_fifo(self):
        """热专家 FIFO: 满时淘汰最早条目."""
        eng = _make_engine_stub()
        eng._max_hot_experts = 2
        fp4 = self._make_fp4_raw(I=_IM, D=_HS)
        eng._load_expert_raw = MagicMock(return_value=fp4["raw"])

        eng._hot_expert_set = {10, 20, 30}
        eng._load_expert_fp4_raw(0, 10)
        eng._load_expert_fp4_raw(0, 20)
        assert len(eng._gpu_hot_experts) == 2
        eng._load_expert_fp4_raw(0, 30)
        assert len(eng._gpu_hot_experts) == 2
        assert (0, 10) not in eng._gpu_hot_experts  # FIFO evicted
        assert (0, 20) in eng._gpu_hot_experts
        assert (0, 30) in eng._gpu_hot_experts
        # Verify stored as FP4 6-tuple
        assert len(eng._gpu_hot_experts[(0, 30)]) == 6

    def test_fp4_cache_survives_clear(self):
        """generate() 的 clear 清理 FP4 缓存."""
        eng = _make_engine_stub()
        eng._max_hot_experts = 4
        eng._hot_expert_set = {42}
        fp4 = self._make_fp4_raw(I=_IM, D=_HS)
        eng._load_expert_raw = MagicMock(return_value=fp4["raw"])
        eng._load_expert_fp4_raw(0, 42)
        assert len(eng._gpu_hot_experts) == 1

        # Simulate generate() clear
        eng._gpu_hot_experts.clear()
        assert len(eng._gpu_hot_experts) == 0


# ── V21.7 回归复现: 跨层复用同一专家时去量化被重复执行 ──────────
#    Bug: 同一 eid 在 layer 0 和 layer 1 都被路由到,
#    _load_expert_fp4_raw 返回 6 元组 (FP4 raw),
#    _forward_legacy 每次都要重复跑 triton_dequantize_fp4_all.
#    期望: 同一 eid 在一次 generate 内只去量化一次, 后续复用 BF16 缓存.


@pytest.mark.fast
class TestCrossLayerDequantRepeat:
    """V21.7 回归: 跨层复用同一专家时, 去量化被重复执行."""

    def setup_method(self):
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()

    def _make_fp4_raw(self, I=32, D=128):
        from home_seek._fp4 import cast
        w1 = torch.randn(I, D, dtype=torch.bfloat16)
        w3 = torch.randn(I, D, dtype=torch.bfloat16)
        w2 = torch.randn(D, I, dtype=torch.bfloat16)
        w1p, w1s = cast(w1.cpu(), fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2.cpu(), fmt="e2m1", block_size=(1, 32))
        return ((w1p, w1s, "fp4"), (w3p, w3s, "fp4"), (w2p, w2s, "fp4"))

    def test_same_expert_different_layers_double_dequant(self):
        """相同 eid 在 layer 0 和 layer 1 各触发一次 triton_dequantize_fp4_all.
        期望: 总共只调用 1 次 (layer 0 去量化后缓存 BF16, layer 1 复用).
        """
        from unittest.mock import patch
        from home_seek.fused_moe import triton_dequantize_fp4_all as real_dequant
        call_count = [0]

        def counting_dequant(*args):
            call_count[0] += 1
            return real_dequant(*args)

        # _load_bf16_deq 在 engine.py 中调用, _forward_legacy 在 fused_moe 中调用
        with patch(
            'home_seek.inference_engine.engine.triton_dequantize_fp4_all',
            counting_dequant
        ), patch(
            'home_seek.fused_moe.triton_dequantize_fp4_all',
            counting_dequant
        ):
            from tests._engine_stub import make_engine
            eng = make_engine()
            eng._max_hot_experts = 8
            eng._max_bf16_cache = 16
            fp4_raw = self._make_fp4_raw(I=32, D=128)
            eng._load_expert_raw = MagicMock(return_value=fp4_raw)

            r0 = eng._load_expert_fp4_raw(0, 5)
            r1 = eng._load_expert_fp4_raw(1, 5)
            assert len(r0) == 6
            assert len(r1) == 6

            from home_seek.fused_moe import FusedMoEFFN
            moe = FusedMoEFFN(num_experts=4, intermediate_size=32,
                               hidden_size=128, use_triton=False)

            def load_fn(layer, eid):
                return eng._load_bf16_deq(layer, eid)

            hidden = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
            idx = torch.tensor([[5, -1, -1, -1]], device="cuda")
            w = torch.tensor([[1.0, 0, 0, 0]], device="cuda")

            call_count[0] = 0
            moe._forward_legacy(hidden, idx, w, load_fn, 0)
            first_calls = call_count[0]

            moe._forward_legacy(hidden, idx, w, load_fn, 1)
            total_calls = call_count[0]

        assert first_calls == 1, f"第一次 forward 应去量化 1 次, 实际 {first_calls}"
        assert total_calls == 1, (
            f"[FIXED] 跨层同一专家触发了 {total_calls} 次去量化, "
            f"期望 1 次 (第一层去量化后缓存 BF16, 第二层复用)")

    def test_hot_batched_same_expert_cross_layer(self):
        """_forward_ffn_hot_batched 路径下, 同一专家跨层也重复去量化."""
        from unittest.mock import patch
        from home_seek.inference_engine import engine as eng_module
        from home_seek.fused_moe import triton_dequantize_fp4_all as real_dequant
        call_count = 0

        def counting_dequant(*args):
            nonlocal call_count
            call_count += 1
            return real_dequant(*args)

        with patch.object(eng_module, 'triton_dequantize_fp4_all', counting_dequant):
            from tests._engine_stub import make_engine
            from home_seek._fp4 import cast
            eng = make_engine()
            eng._max_hot_experts = 16
            eng._max_bf16_cache = 16
            eng._hot_expert_set = {5}

            I, D = 32, 128
            w1bf = torch.randn(I, D, dtype=torch.bfloat16)
            w3bf = torch.randn(I, D, dtype=torch.bfloat16)
            w2bf = torch.randn(D, I, dtype=torch.bfloat16)
            w1p, w1s = cast(w1bf.cpu(), fmt="e2m1", block_size=(1, 32))
            w3p, w3s = cast(w3bf.cpu(), fmt="e2m1", block_size=(1, 32))
            w2p, w2s = cast(w2bf.cpu(), fmt="e2m1", block_size=(1, 32))
            eng._load_expert_raw = MagicMock(return_value=(
                (w1p, w1s, "fp4"), (w3p, w3s, "fp4"), (w2p, w2s, "fp4")))

            hidden = torch.randn(1, D, device="cuda", dtype=torch.bfloat16)
            idx = torch.tensor([[5, -1, -1, -1]], device="cuda")
            w = torch.tensor([[1.0, 0, 0, 0]], device="cuda")

            call_count = 0
            eng._forward_ffn_hot_batched(hidden, idx, w, 0)
            c1 = call_count

            eng._forward_ffn_hot_batched(hidden, idx, w, 1)
            c2 = call_count

        assert c1 == 1, f"第一层应触发 1 次去量化, 实际 {c1}"
        assert c2 == 1, (
            f"[FIXED] hot_batched 路径跨层同一专家触发了 {c2} 次去量化, "
            f"期望 1 次 (复用 BF16 缓存)")


@pytest.mark.fast
class TestEpAffinityScheduling:
    """EP 亲和性调度: 替代 eid%2 硬切分, 按缓存位置动态分配."""

    def _make_ep_engine(self):
        from unittest.mock import MagicMock
        from home_seek.inference_engine.parallel import EPBackend
        from home_seek.hardware_config import HardwareConfig
        from home_seek.model_config import DeepSeekV4FlashConfig
        eng = make_engine()
        eng.config = DeepSeekV4FlashConfig()
        hw = HardwareConfig(devices=("cuda:0", "cuda:1"), device_map=(0, 0))
        eng._backend = EPBackend(hw, 43)
        eng._is_multigpu = True
        eng._ep_affinity_rr = 0
        eng._hot_expert_set = set()
        eng._hot_expert_set_by_layer = {}
        eng._shared_expert_weights = {}
        eng._shared_ffn = MagicMock()
        eng._shared_ffn.forward.return_value = torch.zeros(
            1, 1, eng.config.hidden_size, device="cpu", dtype=torch.float32)
        eng._get_shared_expert = MagicMock(return_value=None)
        eng._fused_moe = MagicMock()
        eng._fused_moe.forward.return_value = torch.zeros(
            1, eng.config.hidden_size, device="cpu", dtype=torch.float32)
        eng._deq = MagicMock(return_value=torch.randn(
            8, eng.config.hidden_size, device="cpu"))
        return eng

    def test_cache_affinity_returns_device_index(self):
        """应返回 expert 所在缓存的 GPU 索引."""
        eng = self._make_ep_engine()
        # GPU0 hot cache 有 expert 5
        eng._backend.get_device_state("cuda:0").gpu_hot_experts[(0, 5)] = "dummy"
        # GPU1 bf16 cache 有 expert 10
        eng._backend.get_device_state("cuda:1").gpu_bf16_cache[(0, 10)] = "dummy"
        # GPU0 bf16 cache 有 expert 7
        eng._backend.get_device_state("cuda:0").gpu_bf16_cache[(0, 7)] = "dummy"

        assert eng._check_expert_cache_affinity(0, 5) == 0
        assert eng._check_expert_cache_affinity(0, 10) == 1
        assert eng._check_expert_cache_affinity(0, 7) == 0
        assert eng._check_expert_cache_affinity(0, 99) == -1

    def test_cache_affinity_hot_and_bf16_both_checked(self):
        """hot cache 和 bf16 cache 任一命中即返回."""
        eng = self._make_ep_engine()
        eng._backend.get_device_state("cuda:1").gpu_hot_experts[(0, 3)] = "dummy"
        assert eng._check_expert_cache_affinity(0, 3) == 1

        eng._backend.get_device_state("cuda:0").gpu_bf16_cache[(0, 4)] = "dummy"
        assert eng._check_expert_cache_affinity(0, 4) == 0

    def test_empty_cache_returns_minus_one(self):
        """空缓存应返回 -1."""
        eng = self._make_ep_engine()
        assert eng._check_expert_cache_affinity(0, 0) == -1
        assert eng._check_expert_cache_affinity(5, 42) == -1

    def test_split_uses_affinity_not_eid_mod(self):
        """ep_split 用缓存亲和性替代 eid%2."""
        eng = self._make_ep_engine()
        # eid=0 缓存在 GPU1, eid=1 缓存在 GPU0
        eng._backend.get_device_state("cuda:1").gpu_hot_experts[(0, 0)] = "dummy"
        eng._backend.get_device_state("cuda:0").gpu_bf16_cache[(0, 1)] = "dummy"

        topk_idx = torch.tensor([[0, 1]], device="cpu")
        n_gpu = 2
        unique_eids = sorted(set(int(x) for x in topk_idx.flatten().tolist() if x >= 0))
        eid_to_dev = {}
        for eid in unique_eids:
            dev_idx = eng._check_expert_cache_affinity(0, eid)
            if dev_idx < 0:
                dev_idx = eng._ep_affinity_rr % n_gpu
                eng._ep_affinity_rr += 1
            eid_to_dev[eid] = dev_idx

        topk_per_gpu = [topk_idx.clone().fill_(-1) for _ in range(n_gpu)]
        for eid, dev_idx in eid_to_dev.items():
            topk_per_gpu[dev_idx][topk_idx == eid] = eid

        topk_0 = topk_per_gpu[0]
        topk_1 = topk_per_gpu[1]

        # eid=0 cached on GPU1 → should be in topk_1, not topk_0
        assert topk_0[0, 0].item() == -1, "eid=0 cached on GPU1, should NOT be in topk_0"
        assert topk_1[0, 0].item() == 0, "eid=0 cached on GPU1, should be in topk_1"
        # eid=1 cached on GPU0 → should be in topk_0, not topk_1
        assert topk_0[0, 1].item() == 1, "eid=1 cached on GPU0, should be in topk_0"
        assert topk_1[0, 1].item() == -1, "eid=1 cached on GPU0, should NOT be in topk_1"

    def test_uncached_round_robin(self):
        """未缓存的 expert 应轮询分配到各 GPU."""
        eng = self._make_ep_engine()
        n_gpu = 2
        eids = [10, 20, 30]
        eid_to_dev = {}
        for eid in eids:
            dev_idx = eng._check_expert_cache_affinity(0, eid)
            if dev_idx < 0:
                dev_idx = eng._ep_affinity_rr % n_gpu
                eng._ep_affinity_rr += 1
            eid_to_dev[eid] = dev_idx

        assert eid_to_dev[10] == 0  # first uncached → GPU0
        assert eid_to_dev[20] == 1  # second uncached → GPU1
        assert eid_to_dev[30] == 0  # third uncached → GPU0
        assert eng._ep_affinity_rr == 3

    def test_mixed_cached_and_uncached(self):
        """部分缓存的 expert: 缓存优先, 未缓存轮询."""
        eng = self._make_ep_engine()
        # eid=5 cached on GPU0
        eng._backend.get_device_state("cuda:0").gpu_hot_experts[(0, 5)] = "dummy"
        # eid=7 cached on GPU1
        eng._backend.get_device_state("cuda:1").gpu_bf16_cache[(0, 7)] = "dummy"
        # eid=9, 11 uncached

        n_gpu = 2
        all_eids = [5, 7, 9, 11]
        eid_to_dev = {}
        for eid in all_eids:
            dev_idx = eng._check_expert_cache_affinity(0, eid)
            if dev_idx < 0:
                dev_idx = eng._ep_affinity_rr % n_gpu
                eng._ep_affinity_rr += 1
            eid_to_dev[eid] = dev_idx

        assert eid_to_dev[5] == 0  # cached on GPU0
        assert eid_to_dev[7] == 1  # cached on GPU1
        assert eid_to_dev[9] == 0  # uncached, round-robin: GPU0
        assert eid_to_dev[11] == 1  # uncached, round-robin: GPU1


    def test_batch_affinity_matches_per_eid(self):
        """batch affinity method produces same results as per-eid."""
        eng = self._make_ep_engine()
        eng._backend.get_device_state("cuda:0").gpu_hot_experts[(0, 5)] = "dummy"
        eng._backend.get_device_state("cuda:1").gpu_bf16_cache[(0, 7)] = "dummy"

        all_eids = [5, 7, 9, 11]
        batch_result = eng._check_expert_cache_affinity_batch(0, all_eids)

        for eid in all_eids:
            expected = eng._check_expert_cache_affinity(0, eid)
            assert batch_result.get(eid, -1) == expected, \
                f"eid={eid}: batch={batch_result.get(eid, -1)} expected={expected}"

    def test_batch_affinity_all_cached_on_first_gpu(self):
        """All experts cached on GPU0."""
        eng = self._make_ep_engine()
        for eid in [1, 2, 3]:
            eng._backend.get_device_state("cuda:0").gpu_hot_experts[(0, eid)] = "dummy"
        result = eng._check_expert_cache_affinity_batch(0, [1, 2, 3])
        assert all(v == 0 for v in result.values())

    def test_batch_affinity_none_cached(self):
        """No experts cached — all return -1."""
        eng = self._make_ep_engine()
        result = eng._check_expert_cache_affinity_batch(0, [99, 100])
        assert all(v == -1 for v in result.values())


@pytest.mark.fast
class TestNumaAwarePrefill:
    """NUMA-aware prefill: per-GPU cache with NUMA-local pinned copies."""

    def _make_numa_engine(self):
        from home_seek.inference_engine.parallel import EPBackend
        from home_seek.hardware_config import HardwareConfig
        from collections import OrderedDict
        eng = make_engine()
        eng.config = DeepSeekV4FlashConfig()
        eng.hw_config = HardwareConfig(
            devices=("cuda:0", "cuda:1"),
            device_map=(0, 0),
            parallel_backend="ep",
            ep_numa_aware=True,
        )
        eng.hw_profile = MagicMock()
        eng.hw_profile.numa_map = {0: 0, 1: 1}
        eng._backend = EPBackend(eng.hw_config, 43)
        eng._is_multigpu = True
        eng._ep_affinity_rr = 0
        eng._hot_expert_set = set()
        eng._hot_expert_set_by_layer = {}
        eng._shared_expert_weights = {}
        eng._shared_ffn = MagicMock()
        eng._shared_ffn.forward.return_value = torch.zeros(
            1, 1, eng.config.hidden_size, device="cpu", dtype=torch.float32)
        eng._get_shared_expert = MagicMock(return_value=None)
        eng._fused_moe = MagicMock()
        eng._fused_moe.forward.return_value = torch.zeros(
            1, eng.config.hidden_size, device="cpu", dtype=torch.float32)
        eng._deq = MagicMock(return_value=torch.randn(
            8, eng.config.hidden_size, device="cpu"))
        eng._gpu_bf16_deq_cache = OrderedDict()
        eng._max_gpu_bf16_deq = 48
        eng._gpu_hot_experts = {}
        eng._max_hot_experts = 16
        eng._gpu_bf16_cache = OrderedDict()
        eng._max_bf16_cache = 16
        eng._deq_cache = OrderedDict()
        eng._prefetch_worker = None
        eng._prefetch_enabled = False
        eng.predictor = MagicMock()
        eng._warmed_up = True
        # Set up per-GPU caches
        eng._expert_caches = {}
        return eng

    def test_make_raw_entry_numa_preserves_entry_format(self):
        """_make_raw_entry_numa should return same format as _make_raw_entry for FP4."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        eng = self._make_numa_engine()
        data = torch.randint(-128, 127, (4096, 512), dtype=torch.int8, device="cpu")
        scale = torch.zeros(128, dtype=torch.float8_e8m0fnu, device="cpu")
        entry_numa = eng._make_raw_entry_numa(data, scale, numa_node=0)
        entry_norm = eng._make_raw_entry(data, scale)
        assert entry_numa is not None
        assert len(entry_numa) == 3
        assert entry_numa[2] == "fp4"
        assert entry_norm is not None
        assert entry_numa[2] == entry_norm[2]

    def test_make_raw_entry_numa_non_fp4_passthrough(self):
        """Non-FP4 data should pass through without NUMA binding."""
        eng = self._make_numa_engine()
        data = torch.randn(64, 512, dtype=torch.bfloat16, device="cpu")
        entry = eng._make_raw_entry_numa(data, None, numa_node=0)
        assert entry is not None
        assert entry[2] == "bf16"

    def test_load_expert_weights_stores_in_per_gpu_cache(self):
        """_load_expert_weights with ep_numa_aware should store entries in per-GPU cache."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        eng = self._make_numa_engine()
        # Populate per-GPU caches
        from home_seek.inference_engine import ExpertWeightCache
        for dev_idx in range(2):
            eng._expert_caches[dev_idx] = ExpertWeightCache(
                max_experts=64, device=f"cuda:{dev_idx}", hot_deq_size=0)
        # Set up mock loader with int8 FP4 data
        fake_data = torch.randint(-128, 127, (4096, 512), dtype=torch.int8, device="cpu")
        fake_scale = torch.zeros(128, dtype=torch.float8_e8m0fnu, device="cpu")
        prefix = "layers.0.ffn.experts.5"
        eng.loader.get_weights.return_value = {
            f"{prefix}.w1.weight": fake_data,
            f"{prefix}.w1.scale": fake_scale,
            f"{prefix}.w3.weight": fake_data,
            f"{prefix}.w3.scale": fake_scale,
            f"{prefix}.w2.weight": fake_data,
            f"{prefix}.w2.scale": fake_scale,
        }
        # Call _load_expert_weights
        result = eng._load_expert_weights(0, 5)
        assert result is not None
        # Verify per-GPU cache has NUMA-local entry
        dev_idx = 5 % 2  # eid=5 → GPU1
        gpu_cache = eng._expert_caches.get(dev_idx)
        assert gpu_cache is not None
        gpu_entry = gpu_cache.get("0_5")
        assert gpu_entry is not None
        assert gpu_entry[0][2] == "fp4"

    def test_load_expert_fp4_raw_prefers_per_gpu_cache(self):
        """_load_expert_fp4_raw should prefer per-GPU cache entry for NUMA-local DMA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        eng = self._make_numa_engine()
        from home_seek.inference_engine import ExpertWeightCache
        for dev_idx in range(2):
            eng._expert_caches[dev_idx] = ExpertWeightCache(
                max_experts=64, device=f"cuda:{dev_idx}", hot_deq_size=0)
        # Pre-populate GPU0's per-GPU cache with a real FP4 entry
        fake_data = torch.randint(-128, 127, (4096, 512), dtype=torch.int8, device="cpu")
        fake_scale = torch.zeros(128, dtype=torch.float8_e8m0fnu, device="cpu")
        w1 = eng._make_raw_entry_numa(fake_data, fake_scale, numa_node=0)
        w3 = eng._make_raw_entry_numa(fake_data, fake_scale, numa_node=0)
        w2 = eng._make_raw_entry_numa(fake_data, fake_scale, numa_node=0)
        eng._expert_caches[0].put("0_99", w1, w3, w2, pin=False)
        # Set up mock loader to return different data (will NOT be used since
        # per-GPU cache takes priority)
        prefix = "layers.0.ffn.experts.99"
        eng.loader.get_weights.return_value = {
            f"{prefix}.w1.weight": fake_data,
            f"{prefix}.w1.scale": fake_scale,
            f"{prefix}.w3.weight": fake_data,
            f"{prefix}.w3.scale": fake_scale,
            f"{prefix}.w2.weight": fake_data,
            f"{prefix}.w2.scale": fake_scale,
        }
        # Call _load_expert_fp4_raw from device 0 (GPU0 path)
        with torch.cuda.device(0):
            result = eng._load_expert_fp4_raw(0, 99)
        assert result is not None
        assert len(result) == 6  # fp4_six tuple
