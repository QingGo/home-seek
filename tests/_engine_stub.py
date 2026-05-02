"""Shared engine stub builder — eliminates per-test __new__ + manual attribute setup."""

from __future__ import annotations
from unittest.mock import MagicMock
from collections import OrderedDict
import torch

from home_seek.inference_engine import (
    HomeSeekInferenceEngine, ExpertWeightCache, WeightLoader,
)
from home_seek.model_config import DeepSeekV4FlashConfig


class EngineStub:
    """Lightweight engine for unit tests. Defaults match real engine behavior."""

    def __init__(self, config: DeepSeekV4FlashConfig | None = None):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = config or DeepSeekV4FlashConfig()
        eng.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eng.expert_cache = ExpertWeightCache(max_experts=64, device=str(eng.device))
        eng._cpu_fallback_enabled = False
        eng.verbose = False
        eng.loader = MagicMock()
        eng.loader.get_weights.return_value = {}
        eng._hot_expert_ids = []
        eng._hash_expert_ids = []
        eng._hot_expert_set = set()
        eng._hot_expert_set_by_layer = {}
        eng._gpu_hot_experts = {}
        eng._max_hot_experts = 16
        eng._gpu_bf16_cache = OrderedDict()
        eng._max_bf16_cache = 16
        eng._gpu_bf16_deq_cache = OrderedDict()
        eng._max_gpu_bf16_deq = 48
        eng._ep_affinity_rr = 0
        eng._shared_expert_weights = {}
        eng._shared_ffn = MagicMock()
        eng._shared_ffn.forward.return_value = torch.zeros(1, 1, eng.config.hidden_size,
                                                            device=eng.device, dtype=torch.bfloat16)
        eng._get_shared_expert = MagicMock(return_value=None)
        eng._fused_moe = MagicMock()
        eng._fused_moe.use_triton = False
        eng._log = lambda msg: None
        eng._deq_cache = OrderedDict()
        eng._prefetch_worker = None
        eng._prefetch_enabled = False
        eng._is_multigpu = False
        eng._devices = ("cuda:0",)
        eng._device_map = tuple([0] * eng.config.num_hidden_layers)
        eng.predictor = MagicMock()
        eng._warmed_up = True
        eng._mtp_num_draft = 2
        self._eng = eng

    def get(self) -> HomeSeekInferenceEngine:
        return self._eng

    def with_config(self, **kwargs) -> EngineStub:
        self._eng.config = DeepSeekV4FlashConfig(**kwargs)
        return self

    def with_loader(self, loader: WeightLoader) -> EngineStub:
        self._eng.loader = loader
        return self

    def with_hot_experts(self, eids: set[int]) -> EngineStub:
        self._eng._hot_expert_ids = list(eids)
        self._eng._hot_expert_set = eids
        self._eng._hot_expert_set_by_layer = {}
        return self

    def with_expert_cache(self, cache: ExpertWeightCache) -> EngineStub:
        self._eng.expert_cache = cache
        return self

    def mock_loader_weight(self, key: str, tensor: torch.Tensor) -> EngineStub:
        self._eng.loader.get_weights.return_value = {key: tensor}
        return self


def make_engine(**config_kwargs) -> HomeSeekInferenceEngine:
    """One-liner: create a test engine with optional config overrides."""
    return EngineStub(DeepSeekV4FlashConfig(**config_kwargs) if config_kwargs else None).get()
