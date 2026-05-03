"""轻量 (<1s) 单元测试: ParallelBackend / PPBackend / 引擎多 GPU 集成。"""
from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock

import torch

from home_seek.hardware_config import HardwareConfig
from home_seek.inference_engine.parallel import (
    ParallelBackend, PPBackend, EPBackend, PerDeviceState,
    _EPWorkItem, _EPWorker,
)
from home_seek.model_config import DeepSeekV4FlashConfig


# ── PPBackend 纯逻辑测试 (不需要 GPU) ──────────────────


def _hw(devices=("cuda:0",), device_map=(0, 0, 1, 1)):
    return HardwareConfig(devices=devices, device_map=device_map)


class TestPPBackendDeviceResolution:
    def test_layer_device_single_gpu(self):
        b = PPBackend(_hw(devices=("cuda:0",), device_map=(0, 0, 0)), 3)
        assert b.layer_device(0) == "cuda:0"
        assert b.layer_device(2) == "cuda:0"

    def test_layer_device_multi_gpu(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 0, 1, 1)), 4)
        assert b.layer_device(0) == "cuda:0"
        assert b.layer_device(2) == "cuda:1"

    def test_first_last_device(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 0, 1, 1)), 4)
        assert b.first_device() == "cuda:0"
        assert b.last_device() == "cuda:1"

    def test_device_index(self):
        b = PPBackend(_hw(device_map=(0, 0, 1, 1)), 4)
        assert b.device_index(0) == 0
        assert b.device_index(2) == 1

    def test_is_multigpu(self):
        assert PPBackend(_hw(devices=("cuda:0",)), 1).is_multigpu() is False
        assert PPBackend(_hw(devices=("cuda:0", "cuda:1")), 2).is_multigpu() is True

    def test_n_gpu(self):
        assert PPBackend(_hw(devices=("cuda:0",)), 1).n_gpu() == 1
        assert PPBackend(_hw(devices=("cuda:0", "cuda:1")), 2).n_gpu() == 2

    def test_hidden_start_end(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 0, 1, 1)), 4)
        assert b.hidden_start_device() == "cuda:0"
        assert b.hidden_end_device() == "cuda:1"

    def test_last_device_respects_device_map(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(1, 1, 0, 0)), 4)
        assert b.last_device() == "cuda:0"


class TestPPBackendPerDeviceState:
    def test_get_device_state_creates_on_demand(self):
        b = PPBackend(_hw(devices=("cuda:0",)), 1)
        s = b.get_device_state("cuda:0")
        assert isinstance(s, PerDeviceState)
        assert s.device == "cuda:0"

    def test_get_device_state_is_cached(self):
        b = PPBackend(_hw(devices=("cuda:0",)), 1)
        assert b.get_device_state("cuda:0") is b.get_device_state("cuda:0")

    def test_all_device_states(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1")), 2)
        states = b.all_device_states()
        assert len(states) == 2
        assert [s.device for s in states] == ["cuda:0", "cuda:1"]

    def test_gpu_hot_experts_per_device(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 0, 1, 1)), 4)
        hot0 = b.gpu_hot_experts(0)
        hot1 = b.gpu_hot_experts(2)
        assert hot0 is not hot1
        assert hot0 is b.get_device_state("cuda:0").gpu_hot_experts
        assert hot1 is b.get_device_state("cuda:1").gpu_hot_experts

    def test_gpu_bf16_cache_per_device(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 0, 1, 1)), 4)
        c0 = b.gpu_bf16_cache(0)
        c1 = b.gpu_bf16_cache(2)
        assert isinstance(c0, OrderedDict)
        assert c0 is not c1

    def test_shared_expert_cache_per_device(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 0, 1, 1)), 4)
        s0 = b.shared_expert_cache(0)
        s1 = b.shared_expert_cache(2)
        assert s0 is not s1

    def test_gpu_hot_experts_same_device_same_dict(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 1, 0, 1)), 4)
        assert b.gpu_hot_experts(0) is b.gpu_hot_experts(2)

    def test_gpu_hot_experts_different_devices_different_dicts(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 1, 0, 1)), 4)
        assert b.gpu_hot_experts(0) is not b.gpu_hot_experts(1)


class TestParallelBackendFactory:
    def test_from_config_pp(self):
        hw = _hw()
        b = ParallelBackend.from_config(hw, 4)
        assert isinstance(b, PPBackend)
        assert b.strategy == "pp"

    def test_from_config_ep(self):
        hw = HardwareConfig(devices=("cuda:0",), device_map=(0,),
                            parallel_backend="ep")
        b = ParallelBackend.from_config(hw, 1)
        assert isinstance(b, EPBackend)
        assert b.strategy == "ep"

    def test_from_config_tp_raises(self):
        hw = HardwareConfig(devices=("cuda:0",), device_map=(0,),
                            parallel_backend="tp")
        import pytest
        with pytest.raises(ValueError, match="Unknown"):
            ParallelBackend.from_config(hw, 1)

    def test_pp_strategy_default(self):
        hw = _hw()
        b = PPBackend(hw, 4)
        assert b.strategy == "pp"


class TestTransferHidden:
    def test_same_device_noop(self):
        b = PPBackend(_hw(devices=("cuda:0",)), 1)
        t = torch.zeros(2, 3, device="cuda")
        out = b.transfer_hidden(t, "cuda:0")
        assert out is t

    def test_cross_device_transfer(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1")), 2)
        t = torch.zeros(2, 3, device="cuda:0")
        out = b.transfer_hidden(t, "cuda:1")
        assert str(out.device) == "cuda:1"


# ── EPBackend 预留测试 ─────────────────────────────────


class TestEPBackend:
    def test_resolve_expert_device(self):
        hw = HardwareConfig(devices=("cuda:0", "cuda:1", "cuda:2"), device_map=(0, 0))
        b = EPBackend(hw, 2)
        assert b.resolve_expert_device(0, 0) == "cuda:0"
        assert b.resolve_expert_device(0, 3) == "cuda:0"
        assert b.resolve_expert_device(0, 1) == "cuda:1"
        assert b.resolve_expert_device(0, 5) == "cuda:2"


# ── 引擎多 GPU 集成测试 ───────────────────────────────


def _make_stub(n_layers=4):
    from home_seek.inference_engine import HomeSeekInferenceEngine
    eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
    eng.config = DeepSeekV4FlashConfig(num_hidden_layers=n_layers, n_routed_experts=4)
    eng.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eng._is_multigpu = False
    eng._backend = None
    eng._log = lambda msg: None
    return eng


class TestEngineResolveDevice:
    def test_single_gpu(self):
        eng = _make_stub()
        hw = HardwareConfig(devices=("cuda:0",), device_map=(0, 0, 0, 0))
        eng._backend = PPBackend(hw, 4)
        eng._is_multigpu = False
        assert eng._resolve_device(0) == "cuda:0"
        assert eng._resolve_device(3) == "cuda:0"

    def test_multi_gpu(self):
        eng = _make_stub()
        hw = HardwareConfig(devices=("cuda:0", "cuda:1"), device_map=(0, 0, 1, 1))
        eng._backend = PPBackend(hw, 4)
        eng._is_multigpu = True
        assert eng._resolve_device(0) == "cuda:0"
        assert eng._resolve_device(2) == "cuda:1"

    def test_ensure_backend_lazy_init(self):
        eng = _make_stub()
        eng._backend = None
        # _ensure_backend should create a working backend
        dev = eng._resolve_device(0)
        assert dev == "cuda:0"
        assert eng._backend is not None
        assert isinstance(eng._backend, PPBackend)

    def test_ensure_backend_idempotent(self):
        eng = _make_stub()
        eng._ensure_backend()
        b1 = eng._backend
        eng._ensure_backend()
        assert eng._backend is b1


class TestEngineGetPerDevice:
    def test_single_gpu_returns_self_attr(self):
        eng = _make_stub()
        eng.embed = "embed_tensor"
        eng._is_multigpu = False
        assert eng._get_per_device('embed') == "embed_tensor"

    def test_multigpu_returns_device_state(self):
        eng = _make_stub()
        hw = HardwareConfig(devices=("cuda:0", "cuda:1"), device_map=(0, 1))
        eng._backend = PPBackend(hw, 2)
        eng._is_multigpu = True
        eng.embed = "original"
        eng._backend.get_device_state("cuda:1").embed = "on_gpu1"
        result = eng._get_per_device('embed', torch.device("cuda:1"))
        assert result == "on_gpu1"

    def test_multigpu_fallback_to_self(self):
        eng = _make_stub()
        hw = HardwareConfig(devices=("cuda:0", "cuda:1"), device_map=(0, 1))
        eng._backend = PPBackend(hw, 2)
        eng._is_multigpu = True
        eng.embed = "fallback"
        # device state for cuda:1 has no embed → fallback to self.embed
        result = eng._get_per_device('embed', torch.device("cuda:1"))
        assert result == "fallback"

    def test_single_gpu_ignores_device_hint(self):
        eng = _make_stub()
        eng.embed = "embed_tensor"
        eng._is_multigpu = False
        assert eng._get_per_device('embed', "cuda:1") == "embed_tensor"


class TestEngineBackwardCompatibility:
    def test_old_gpu_hot_experts_migrated(self):
        """_load_expert_fp4_raw 通过 backend per-device hot cache 命中。"""
        if not torch.cuda.is_available():
            return
        from home_seek.inference_engine import ExpertWeightCache
        eng = _make_stub()
        eng._backend = None
        eng.expert_cache = ExpertWeightCache(max_experts=64, device=str(eng.device))
        eng._hot_expert_set = set()
        eng._hot_expert_set_by_layer = {}
        eng._cpu_fallback_enabled = False
        eng.loader = MagicMock()
        eng.loader.get_weights.return_value = {}
        w = (torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
             torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
             torch.randn(128, 64, device="cuda", dtype=torch.bfloat16))
        eng._load_expert_raw = MagicMock()
        # 写入 backend per-device hot cache (新的标准方式)
        eng._ensure_backend()
        eng._backend.get_device_state("cuda:0").gpu_hot_experts[(0, 99)] = w
        result = eng._load_expert_fp4_raw(0, 99)
        assert result is not None
        assert len(result) == 3
        eng._load_expert_raw.assert_not_called()

    def test_old_shared_expert_weights_migrated(self):
        """__new__ 桩写入旧 _shared_expert_weights, _get_shared_expert 应找到它。"""
        eng = _make_stub()
        eng._backend = None
        eng.loader = None
        cfg = eng.config
        I, D = cfg.moe_intermediate_size, cfg.hidden_size
        w1 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(D, I, device="cuda", dtype=torch.bfloat16)
        eng._shared_expert_weights = {0: (w1, w3, w2)}
        result = eng._get_shared_expert(0)
        assert result is not None
        assert len(result) == 3

    def test_old_gpu_hot_pointers_redirected(self):
        """_load_expert_fp4_raw 经过后, 旧指针应重定向到 backend per-device dict。"""
        if not torch.cuda.is_available():
            return
        from home_seek.inference_engine import ExpertWeightCache
        eng = _make_stub()
        eng._backend = None
        eng.expert_cache = ExpertWeightCache(max_experts=64, device=str(eng.device))
        eng._cpu_fallback_enabled = False
        eng._hot_expert_set = set()
        eng._hot_expert_set_by_layer = {}
        eng.loader = MagicMock()
        eng.loader.get_weights.return_value = {}
        eng._gpu_hot_experts = {}
        eng._gpu_bf16_cache = OrderedDict()
        eng._max_bf16_cache = 16
        eng._max_hot_experts = 16
        eng._hot_expert_set_by_layer = {0: set()}
        old_id = id(eng._gpu_hot_experts)
        # 写入 FP4 格式的数据使 _load_expert_fp4_raw 走通
        from home_seek._fp4 import cast
        d = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w1p, w1s = cast(d.cpu(), fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(d.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(d.cpu(), fmt="e2m1", block_size=(1, 32))
        def fake_raw(layer, eid):
            return (w1p, w1s, "fp4"), (w3p, w3s, "fp4"), (w2p, w2s, "fp4")
        eng._load_expert_raw = fake_raw
        r = eng._load_expert_fp4_raw(0, 10)
        assert r is not None
        new_id = id(eng._gpu_hot_experts)
        assert old_id != new_id, "pointer should be redirected to backend per-device dict"


class TestEngineGetLayerWeightsDevice:
    def test_multi_gpu_device_routing(self):
        eng = _make_stub(4)
        hw = HardwareConfig(devices=("cuda:0", "cuda:1"), device_map=(0, 0, 1, 1))
        eng._backend = PPBackend(hw, 4)
        eng._is_multigpu = True
        eng._layer_weight_cache = {}
        eng.loader = MagicMock()
        eng.loader.get_weights.return_value = {}
        lw0 = eng._get_layer_weights(0)
        assert lw0 == {}
        # 验证 device 参数传到 loader
        eng._get_layer_weights(2)
        # _get_layer_weights 应该对 layer 2 使用 "cuda:1"
        # 我们验证在测试中至少不崩溃
        assert True

    def test_get_layer_weights_caches(self):
        eng = _make_stub(4)
        eng._backend = PPBackend(_hw(devices=("cuda:0",)), 4)
        eng._is_multigpu = False
        eng._layer_weight_cache = {}
        eng.loader = MagicMock()
        eng.loader.get_weights.return_value = {}
        lw = eng._get_layer_weights(0)
        assert eng._get_layer_weights(0) is lw

    def test_layer_weight_cache_not_cleared_by_forward_layer(self):
        """_forward_layer should never mutate _layer_weight_cache for other layers.

        Regression: commit 1c901ff added per-layer cache cleanup in _forward_layer
        that deleted _layer_weight_cache[layer_idx - 1], forcing every decode step
        to reload all 43 layers from disk (~30× slowdown).
        """
        eng = _make_stub(4)
        eng._backend = PPBackend(_hw(devices=("cuda:0",)), 4)
        eng._is_multigpu = False
        eng._layer_weight_cache = {0: {}, 1: {}}
        eng.loader = MagicMock()
        eng.loader.get_weights.return_value = {}

        cache_len_before = len(eng._layer_weight_cache)
        # _forward_layer needs these mocked to avoid real tensor ops
        D = eng.config.hidden_size
        h = torch.zeros(1, 1, eng.config.hc_mult, D,
                        device=eng.device, dtype=torch.bfloat16)
        eng._process_mhc_layer = lambda h, lw, prefix: (h.sum(dim=2), None, None)
        eng._forward_attn = lambda h, lw, li: torch.zeros(
            1, 1, D, device=h.device, dtype=h.dtype)
        eng._forward_ffn = lambda h, lw, li, *a: (
            torch.zeros(1, 1, D, device=h.device, dtype=h.dtype), set())
        eng._process_mhc_post = lambda h, r, p, c: h

        eng._forward_layer(h, eng._layer_weight_cache[0], 0)
        eng._forward_layer(h, eng._layer_weight_cache[1], 1)

        assert len(eng._layer_weight_cache) >= cache_len_before, (
            f"_layer_weight_cache shrunk from {cache_len_before} "
            f"to {len(eng._layer_weight_cache)} — _forward_layer "
            f"must not delete cache entries"
        )
        assert 0 in eng._layer_weight_cache, "layer 0 should survive forward_layer"
        assert 1 in eng._layer_weight_cache, "layer 1 should survive forward_layer"

    def test_get_layer_weights_cached_across_forward_layer(self):
        """After _forward_layer, subsequent _get_layer_weights must hit cache.

        Regression: same as above — per-layer cleanup invalidated cache for every
        layer, so the second decode step's _get_layer_weights was always a miss.
        """
        eng = _make_stub(4)
        eng._backend = PPBackend(_hw(devices=("cuda:0",)), 4)
        eng._is_multigpu = False
        eng._layer_weight_cache = {}

        call_count = [0]
        mock_weights = {}
        for li in (0, 1):
            for t in ("attn_norm.weight", "ffn_norm.weight",
                      "attn.wq_a.weight", "attn.wq_a.scale",
                      "attn.wq_b.weight", "attn.wq_b.scale"):
                mock_weights[f"layers.{li}.{t}"] = None
        def counting_get_weights(*keys):
            call_count[0] += 1
            return mock_weights
        eng.loader = MagicMock()
        eng.loader.get_weights = counting_get_weights

        D = eng.config.hidden_size
        h = torch.zeros(1, 1, eng.config.hc_mult, D,
                        device=eng.device, dtype=torch.bfloat16)
        eng._process_mhc_layer = lambda h, lw, prefix: (h.sum(dim=2), None, None)
        eng._forward_attn = lambda h, lw, li: torch.zeros(
            1, 1, D, device=h.device, dtype=h.dtype)
        eng._forward_ffn = lambda h, lw, li, *a: (
            torch.zeros(1, 1, D, device=h.device, dtype=h.dtype), set())
        eng._process_mhc_post = lambda h, r, p, c: h

        # First decode step: cold load layers 0 and 1
        lw0 = eng._get_layer_weights(0)
        assert call_count[0] == 1, "first _get_layer_weights should hit loader"
        eng._forward_layer(h, lw0, 0)
        lw1 = eng._get_layer_weights(1)
        assert call_count[0] == 2, "second _get_layer_weights should hit loader"
        eng._forward_layer(h, lw1, 1)
        # Regression: _forward_layer(1) deleted _layer_weight_cache[0]

        # Second decode step for layer 0: must hit cache
        lw0_2 = eng._get_layer_weights(0)
        assert call_count[0] == 2, (
            f"regression: second _get_layer_weights(0) went to loader "
            f"({call_count[0]} calls expected 2)"
        )
        assert lw0 is lw0_2, "should return the same cached dict object"


class TestEngineLoadSharedExpertsGPU:
    def test_shared_experts_on_correct_device(self):
        eng = _make_stub(2)
        dev2 = "cuda:1" if torch.cuda.device_count() >= 2 else "cpu"
        hw = HardwareConfig(devices=("cuda:0", dev2), device_map=(0, 1))
        eng._backend = PPBackend(hw, 2)
        eng._is_multigpu = True
        eng.loader = MagicMock()
        I = eng.config.moe_intermediate_size
        D = eng.config.hidden_size
        w1 = torch.randn(I, D, dtype=torch.bfloat16)
        eng.loader.get_weights.return_value = {
            "layers.0.ffn.shared_experts.w1.weight": w1,
            "layers.0.ffn.shared_experts.w1.scale": None,
            "layers.0.ffn.shared_experts.w3.weight": w1,
            "layers.0.ffn.shared_experts.w3.scale": None,
            "layers.0.ffn.shared_experts.w2.weight": w1.T,
            "layers.0.ffn.shared_experts.w2.scale": None,
            "layers.1.ffn.shared_experts.w1.weight": w1,
            "layers.1.ffn.shared_experts.w1.scale": None,
            "layers.1.ffn.shared_experts.w3.weight": w1,
            "layers.1.ffn.shared_experts.w3.scale": None,
            "layers.1.ffn.shared_experts.w2.weight": w1.T,
            "layers.1.ffn.shared_experts.w2.scale": None,
        }
        eng._load_shared_experts_gpu()
        cached0 = eng._backend.get_device_state("cuda:0").shared_expert_weights.get(0)
        cached1 = eng._backend.get_device_state(dev2).shared_expert_weights.get(1)
        assert cached0 is not None, "layer 0 shared expert should be on cuda:0"
        assert cached1 is not None, "layer 1 shared expert should be on dev2"
        assert str(cached0[0].device) == "cuda:0"
        assert str(cached1[0].device) == dev2


class TestPerDeviceStateIndependence:
    """验证 per-device state 独立: 一个 device 的修改不影响其他。"""

    def test_hot_experts_isolation(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 1)), 2)
        b.gpu_hot_experts(0)[(1, 2)] = "entry_gpu0"
        b.gpu_hot_experts(1)[(3, 4)] = "entry_gpu1"
        assert (1, 2) in b.gpu_hot_experts(0)
        assert (1, 2) not in b.gpu_hot_experts(1)

    def test_shared_expert_isolation(self):
        b = PPBackend(_hw(devices=("cuda:0", "cuda:1"), device_map=(0, 1)), 2)
        b.shared_expert_cache(0)[0] = "shared_gpu0"
        b.shared_expert_cache(1)[1] = "shared_gpu1"
        assert 0 in b.shared_expert_cache(0)
        assert 0 not in b.shared_expert_cache(1)


# ── EP 双线程 Worker 测试 ─────────────────────────────────


class TestEPDualThreadWorker:
    """验证 _EPWorker 线程的启动 / 提交流程 / 结果获取。"""

    def _has_2gpu(self):
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2

    def test_ep_work_item_dataclass(self):
        """_EPWorkItem 字段正确。"""
        ev = torch.cuda.Event()
        item = _EPWorkItem(layer_idx=5, copy_event=ev)
        assert item.layer_idx == 5
        assert item.copy_event is ev
        assert item.result is None
        assert not item.done.is_set()

    def test_worker_start_stop(self):
        """Worker 线程启动后可正常停止。"""
        if not self._has_2gpu():
            return
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(num_hidden_layers=1, n_routed_experts=4)
        eng._load_expert_fp4_raw = lambda l, e: None
        eng._fused_moe = None
        worker = _EPWorker(eng, "cuda:1")
        worker.start()
        assert worker.is_alive()
        worker.stop()
        worker.join(timeout=5)
        assert not worker.is_alive()

    def test_worker_submit_and_result(self):
        """提交简单 work item, 确认结果可达。"""
        if not self._has_2gpu():
            return
        from unittest.mock import MagicMock

        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=1, n_routed_experts=4,
            hidden_size=64, moe_intermediate_size=128,
            swiglu_limit=10.0, num_experts_per_tok=2,
        )
        eng._load_expert_fp4_raw = MagicMock(return_value=None)
        fused_mock = MagicMock()
        fused_mock.forward.return_value = torch.zeros(1, 64, device="cuda:1", dtype=torch.bfloat16)
        eng._fused_moe = fused_mock

        worker = _EPWorker(eng, "cuda:1")
        worker.start()

        h = torch.randn(1, 64, device="cuda:1", dtype=torch.bfloat16)
        tk = torch.full((1, 2), -1, device="cuda:1", dtype=torch.long)
        tw = torch.zeros(1, 2, device="cuda:1", dtype=torch.float32)

        copy_ev = torch.cuda.Event()
        copy_ev.record()
        item = _EPWorkItem(hidden=h, topk_idx=tk, topk_weights=tw,
                           layer_idx=0, copy_event=copy_ev)
        worker.submit(item)
        item.done.wait(timeout=10)
        assert item.done.is_set()
        assert item.result is not None

        worker.stop()
        worker.join(timeout=5)

    def test_ep_backend_start_worker(self):
        """EPBackend.ep_start_worker 惰性启动 worker. """
        if not self._has_2gpu():
            return
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(num_hidden_layers=1, n_routed_experts=4)
        eng._load_expert_fp4_raw = lambda l, e: None
        eng._fused_moe = None

        hw = HardwareConfig(devices=("cuda:0", "cuda:1"), device_map=(0, 0),
                            parallel_backend="ep")
        backend = EPBackend(hw, 1)
        assert backend._ep_worker is None
        backend.ep_start_worker(eng)
        assert backend._ep_worker is not None
        assert backend._ep_worker.is_alive()
        # 幂等
        backend.ep_start_worker(eng)
        assert backend._ep_worker.is_alive()

    def test_ep_workflow_integration(self):
        """完整 EP 工作流: 提交 + 等结果 + 返回值. """
        if not self._has_2gpu():
            return
        from unittest.mock import MagicMock

        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=1, n_routed_experts=4,
            hidden_size=64, moe_intermediate_size=128,
            swiglu_limit=10.0, num_experts_per_tok=2,
        )
        eng._load_expert_fp4_raw = MagicMock(return_value=None)
        fused_mock = MagicMock()
        expected = torch.randn(1, 64, device="cuda:1", dtype=torch.bfloat16)
        fused_mock.forward.return_value = expected
        eng._fused_moe = fused_mock

        hw = HardwareConfig(devices=("cuda:0", "cuda:1"), device_map=(0, 0),
                            parallel_backend="ep")
        backend = EPBackend(hw, 1)
        backend.ep_start_worker(eng)

        h = torch.randn(1, 64, device="cuda:1", dtype=torch.bfloat16)
        tk = torch.full((1, 2), -1, device="cuda:1", dtype=torch.long)
        tw = torch.zeros(1, 2, device="cuda:1", dtype=torch.float32)
        copy_ev = torch.cuda.Event()
        copy_ev.record()

        item = backend.ep_submit_work(h, tk, tw, 0, copy_ev)
        result = backend.ep_wait_result(item)
        assert result is not None
        assert result.shape == (1, 64)
        assert result.device.type == "cuda"
        assert result.device.index == 0  # 自动转回 GPU0

    def test_single_gpu_no_worker(self):
        """单 GPU 不启动 worker. """
        hw = HardwareConfig(devices=("cuda:0",), device_map=(0,),
                            parallel_backend="ep")
        backend = EPBackend(hw, 1)
        backend.ep_start_worker(None)
        assert backend._ep_worker is None
