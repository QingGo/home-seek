from __future__ import annotations

import os
import json
import time
import logging
import torch

_logger = logging.getLogger(__name__)
import torch.nn.functional as F
from collections import OrderedDict
from torch.profiler import record_function
from home_seek.inference_engine.weight_loader import WeightLoader, load_fp8_weight, load_fp4_weight
from home_seek.inference_engine.layer_state import LayerState
from home_seek.inference_engine.expert_cache import ExpertWeightCache

import math
from home_seek.utils import rms_norm
from home_seek.model_config import DeepSeekV4FlashConfig
from home_seek.hardware_config import HardwareConfig
from home_seek.inference_engine.parallel import ParallelBackend
from home_seek.mhc import mhc_split_sinkhorn
from home_seek.fused_moe import FusedMoEFFN, SharedExpertFFN, clear_deq_cache, triton_dequantize_fp4_all
from home_seek.compressor import Compressor as NewCompressor
from home_seek.lightning_indexer import LightningIndexer
from home_seek.hybrid_kv_cache import HybridKVCache


try:
    from home_seek.encoding_dsv4 import encode_messages
except ImportError:
    encode_messages = None


def precompute_freqs_cis(dim: int, seqlen: int, theta: float = 10000.0,
                         original_seq_len: int = 0, factor: float = 1.0,
                         beta_fast: int = 32, beta_slow: int = 1,
                         start_pos: int = 0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0 and factor > 1.0:
        low = math.floor(dim * math.log(original_seq_len / (beta_fast * 2 * math.pi)) / (2 * math.log(theta)))
        high = math.ceil(dim * math.log(original_seq_len / (beta_slow * 2 * math.pi)) / (2 * math.log(theta)))
        low = max(low, 0)
        high = min(high, dim // 2 - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low + 1e-3)).clamp(0, 1)
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1.0 - smooth) + freqs * smooth
    t = torch.arange(start_pos, start_pos + seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, rd: int = 64, inverse: bool = False) -> torch.Tensor:
    x_rope = x[..., -rd:]
    x_pass = x[..., :-rd]
    T = x_rope.shape[-2]
    if x_rope.ndim == 4:
        h = x_rope.shape[1]
        freqs = freqs_cis[:T].view(1, 1, T, rd // 2).expand(-1, h, -1, -1)
    else:
        freqs = freqs_cis[:T].view(1, T, rd // 2)
    if inverse:
        freqs = freqs.conj()
    x_rope_complex = torch.view_as_real(
        torch.view_as_complex(x_rope.float().reshape(*x_rope.shape[:-1], -1, 2)) * freqs
    ).flatten(-2)
    return torch.cat([x_pass, x_rope_complex.to(x.dtype)], dim=-1)


def _fp8_simulate(x, block_size=64):
    """In-place FP8 quantize+dequantize simulation (QAT). Matches demo act_quant."""
    orig_dtype = x.dtype
    x_f32 = x.float()
    *front, N = x_f32.shape
    x_flat = x_f32.reshape(-1, N)
    M = x_flat.shape[0]
    num_blocks = (N + block_size - 1) // block_size
    if N % block_size != 0:
        pad = block_size - N % block_size
        x_flat = F.pad(x_flat, (0, pad))
    x_blocks = x_flat.reshape(M, num_blocks, block_size)
    amax = x_blocks.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-4)
    scale = 2 ** torch.ceil(torch.log2(amax / 448.0))
    x_q = (x_blocks / scale).clamp(-448.0, 448.0)
    x_dq = x_q * scale
    x_flat.copy_(x_dq.view_as(x_flat)[:, :N])
    x.copy_(x_f32.to(orig_dtype))
    return x






class HomeSeekInferenceEngine:
    # 测试桩兼容: __new__ 创建的实例可能没有这些属性
    _backend = None
    _is_multigpu = False
    _devices = ("cuda:0",)
    _device_map = ()

    def _ensure_backend(self):
        if self._backend is not None:
            return
        from home_seek.hardware_config import HardwareConfig as _HC
        from home_seek.inference_engine.parallel import ParallelBackend as _PB
        n_layers = 43
        try:
            raw = self.config.num_hidden_layers
            if isinstance(raw, int):
                n_layers = raw
        except Exception:
            pass
        hw = _HC(devices=("cuda:0",), device_map=tuple([0] * n_layers))
        self._backend = _PB.from_config(hw, n_layers)

    def __init__(self, weight_dir: str = "weights", device: str = "cuda", verbose: bool = False,
                 hot_experts_path: str = "hot_experts.json", preload_all: bool = False,
                 reprobe: bool = False, use_triton: bool = True,
                 use_gqa_fusion: bool = False,
                 hw_overrides: dict | None = None):
        config_path = os.path.join(weight_dir, "config.json")
        self.config = DeepSeekV4FlashConfig.from_json(config_path)
        self.weight_dir = weight_dir
        raw_dev = torch.device(device)
        self.device = torch.device(raw_dev.type, raw_dev.index if raw_dev.index is not None else 0)
        self.verbose = verbose
        self.loader = WeightLoader(weight_dir, str(self.device))
        from home_seek.hw_profile import probe_hardware, load_profile
        self.hw_profile = probe_hardware(force=reprobe, weight_dir=weight_dir) if reprobe else load_profile()
        if self.hw_profile is None:
            self.hw_profile = probe_hardware(force=True, weight_dir=weight_dir)
        # ── HardwareConfig: 从硬件探测自动推导所有硬件参数 ────────
        self.hw_config = HardwareConfig.auto(
            self.hw_profile, self.config,
            **(hw_overrides or {}),
        )

        # CPU expert cache size: 兼顾 cgroup 限制
        try:
            _cgroup_max = 96 * 1024**3
            for _p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
                if os.path.exists(_p):
                    with open(_p) as _f:
                        _v = _f.read().strip()
                    if _v.isdigit() and int(_v) < 10**15:
                        _cgroup_max = int(_v)
                        break
            _avail_bytes = _cgroup_max
            _I = self.config.moe_intermediate_size
            _D = self.config.hidden_size
            _per_expert_bytes = int(_I * _D * 51 / 32)
            _reserve_bytes = 8 * 1024**3
            _max_by_ram = max(0, (_avail_bytes - _reserve_bytes) // _per_expert_bytes)
            _total_experts = self.config.num_hidden_layers * self.config.n_routed_experts
            # P2: CPU cache 从 50% RAM 扩大到 75%, 减少文件 I/O
            _cpu_ratio = _max_by_ram * 3 // 4
            cache_size = min(self.hw_config.cpu_cache_max, max(2048, min(_cpu_ratio, _total_experts)))
            self._log(f"Host: {_avail_bytes/1e9:.0f} GB cgroup limit, "
                      f"_per_expert={_per_expert_bytes/1e6:.1f} MB, "
                      f"cache_size={cache_size} (~{cache_size*_per_expert_bytes/1e9:.0f} GB)")
        except Exception:
            cache_size = self.hw_config.cpu_cache_max
            self._log(f"Host: unknown RAM, cache_size={cache_size}")
        self.expert_cache = ExpertWeightCache(max_experts=cache_size, device=str(self.device), hot_deq_size=0)
        self._expert_caches: dict[int, ExpertWeightCache] = {}
        """P2: EP+numa_aware 时每 GPU 独立的 ExpertWeightCache, 确保 NUMA-local 分配。"""
        self.layer_states: dict[int, LayerState] = {}
        self._deq_cache: OrderedDict = OrderedDict()
        self._compressors: dict[int, NewCompressor] = {}
        self._indexers: dict[int, LightningIndexer] = {}
        self._hybrid_kv: dict[int, HybridKVCache] = {}
        self._global_pos = 0
        self._load_global_weights()
        self._use_triton = use_triton
        self._use_gqa_fusion = use_gqa_fusion
        from home_seek.expert_predictor import RecordingPredictor, HeuristicPredictor
        self.predictor = RecordingPredictor(
            HeuristicPredictor(self.config.num_hidden_layers, self.config.num_experts_per_tok))
        self._cpu_fallback_enabled = False
        self._hot_expert_ids: list[int] = []
        self._hash_expert_ids: list[int] = []
        self._hot_expert_set: set[int] = set()
        self._hot_expert_set_by_layer: dict[int, set[int]] = {}
        self._gpu_hot_experts: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        self._max_hot_experts = self.hw_config.gpu_hot_max
        self._gpu_bf16_cache: OrderedDict = OrderedDict()
        self._max_bf16_cache = self.hw_config.gpu_bf16_max
        # V21.7: 跨层 BF16 去量化缓存 (keyed by eid, LRU 48, ~2.3GB)
        self._gpu_bf16_deq_cache: OrderedDict = OrderedDict()
        self._max_gpu_bf16_deq = 48
        # Per-layer online hot usage tracking for dynamic GPU cache update
        self._hot_usage_counter: dict[int, dict[int, int]] = {}
        self._hot_usage_window: list[tuple[int, int]] = []
        self._hot_usage_window_size = 1024
        self._ep_affinity_rr = 0  # round-robin counter for EP affinity scheduling
        self._preload_hot_experts(hot_experts_path)
        self._kv_offload_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._compress_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._prefetch_enabled = self.hw_config.prefetch_enabled
        self._prefetch_prev_layer_eids: dict[int, list[int]] = {}
        self._prefetch_current_eids: dict[int, list[int]] = {}

        self._shared_expert_weights: dict[int, tuple | None] = {}
        self._shared_experts_loaded = False
        # Clear module-level deq cache to prevent cross-GPU memory accumulation
        from home_seek.fused_moe import clear_deq_cache
        clear_deq_cache()
        self._deq_cache_hwm = 0
        self._mtp_weights: dict[str, torch.Tensor] = {}
        self._mtp_loaded = False
        self._mtp_eager = False
        self._mtp_num_draft = self.hw_config.mtp_num_draft
        self._last_routed_eids: list[int] = []
        self._warmed_up = False
        self._layer_weight_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._phase = "idle"

        self._ep_ev_gpu0_start = torch.cuda.Event(enable_timing=True)
        self._ep_ev_gpu0_end = torch.cuda.Event(enable_timing=True)
        self._ep_ev_copy_start = torch.cuda.Event(enable_timing=True)
        self._ep_ev_copy_end = torch.cuda.Event(enable_timing=True)

        self._fused_moe = FusedMoEFFN(
            num_experts=self.config.n_routed_experts,
            intermediate_size=self.config.moe_intermediate_size,
            hidden_size=self.config.hidden_size,
            swiglu_limit=self.config.swiglu_limit,
            use_triton=True,
            triton_blocks=self.hw_config.triton_blocks,
            cublas_max_tokens=self.hw_config.cublas_max_tokens,
        )
        self._shared_ffn = SharedExpertFFN(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.moe_intermediate_size,
            swiglu_limit=self.config.swiglu_limit,
            use_triton=self._use_triton,
            triton_blocks=self.hw_config.triton_blocks,
            cublas_max_tokens=self.hw_config.cublas_max_tokens,
        )

        # ── 并行后端 (必须早于 _load_shared_experts_gpu) ──
        self._backend = ParallelBackend.from_config(
            self.hw_config, self.config.num_hidden_layers)
        self._is_multigpu = self._backend.is_multigpu()
        if self._is_multigpu:
            self._log(f"Parallel backend: {self._backend.strategy.upper()}, "
                      f"{self._backend.n_gpu()} devices, "
                      f"device_map={self._backend.device_map}")
        # 多 GPU: 选择性复制 — embed 只需 hidden_start_device, lm_head+norm 只需 hidden_end_device
        if self._is_multigpu:
            first_dev = self._backend.hidden_start_device()
            last_dev = self._backend.hidden_end_device()
            self._backend.get_device_state(first_dev).embed = \
                self.embed.to(first_dev) if self.embed is not None else None
            self._backend.get_device_state(last_dev).lm_head = \
                self.lm_head.to(last_dev) if self.lm_head is not None else None
            self._backend.get_device_state(last_dev).norm_weight = \
                self.norm_weight.to(last_dev) if self.norm_weight is not None else None

        self._load_shared_experts_gpu()
        self._load_mtp_weights()

        # P2: EP+numa_aware → 每 GPU 独立 ExpertWeightCache (NUMA-local 分配)
        # 容量: 默认 cache 存 hot/pinned, per-GPU 存该 GPU 的 non-hot expert
        if self._is_multigpu and getattr(self._backend, 'strategy', 'pp') == 'ep' \
                and self.hw_config.ep_numa_aware:
            pinned_count = len(getattr(self.expert_cache, 'pinned', set()))
            dev_count = self._backend.n_gpu()
            per_dev = max(256, (self.expert_cache.max_experts - pinned_count) // dev_count)
            self._log(f"Per-GPU cache: {dev_count} × ~{per_dev} experts (NUMA local)")
            for dev_idx in range(dev_count):
                cache = ExpertWeightCache(
                    max_experts=per_dev,
                    device=str(self._backend.devices[dev_idx]),
                    hot_deq_size=0,
                )
                self._expert_caches[dev_idx] = cache

        if self._hot_expert_ids:
            self._preload_hot_experts_cpu_cache()
            self._preload_gpu_hot_experts()
        self._stop_token_ids = self._load_stop_token_ids()
        if not preload_all:
            self._conditional_preload()
        else:
            self._preload_all_experts()
        warmup_ok = self._warmup()
        if not warmup_ok:
            self._log("WARNING: Warmup had failures — first inference will be slow "
                      "due to JIT compilation on the critical path.")

    def _get_per_device(self, attr: str, device_hint: torch.device | str | None = None):
        """返回 attr (embed/lm_head/norm_weight) 在指定 device 上的副本。

        单卡: 直接返回 self.attr.
        多卡: 从 _backend 的 per-device 状态取.
        """
        if not getattr(self, '_is_multigpu', False) or device_hint is None:
            return getattr(self, attr, None)
        self._ensure_backend()
        dev = str(device_hint)
        state = self._backend.get_device_state(dev)
        val = getattr(state, attr, None)
        if val is not None:
            return val
        return getattr(self, attr, None)

    def _log(self, msg):
        _logger.info(msg)

    def _preload_hot_experts(self, hot_experts_path: str):
        if not os.path.exists(hot_experts_path):
            self._log(f"No hot experts file at {hot_experts_path}, skipping")
            return
        with open(hot_experts_path) as f:
            data = json.load(f)
        self._hot_expert_ids = data.get("top_hot_experts",
                                        data.get("top_16_hot_experts", []))
        self._hot_expert_set = set(self._hot_expert_ids)

        per_layer = data.get("top_hot_experts_by_layer", {})
        self._hot_expert_set_by_layer = {}
        for k, v in per_layer.items():
            self._hot_expert_set_by_layer[int(k)] = set(v)

        if not self._hot_expert_set_by_layer and hasattr(self, 'config') and hasattr(self.config, 'num_hidden_layers'):
            for lidx in range(self.config.num_hidden_layers):
                self._hot_expert_set_by_layer[lidx] = self._hot_expert_set

        hash_ids = data.get("hash_layer_expert_ids", [])
        self._hash_expert_ids = hash_ids[:18] if len(hash_ids) > 18 else hash_ids
        self._log(f"Hot experts: {len(self._hot_expert_ids)} IDs ({len(self._hot_expert_set_by_layer)} layers), "
                  f"Hash experts: {len(self._hash_expert_ids)} IDs")

    def _load_stop_token_ids(self) -> set[int]:
        tok_path = os.path.join(self.weight_dir, 'tokenizer.json')
        if not os.path.exists(tok_path):
            raise FileNotFoundError(
                f"Tokenizer file not found at {tok_path}. "
                "Please download the model weights to the correct location."
            )
        with open(tok_path) as f:
            data = json.load(f)
        for t in data.get('added_tokens', []):
            if t.get('content') == '<｜end▁of▁sentence｜>':
                return {t['id']}
        raise ValueError(
            "Stop token not found in tokenizer.json. "
            "Expected an added_token with content '<｜end▁of▁sentence｜>'."
        )

    def _preload_gpu_hot_experts(self):
        if not self._hot_expert_ids:
            return
        self._ensure_backend()
        count = 0
        preload_layers = min(4, self.config.num_hidden_layers)
        for layer in range(preload_layers):
            layer_set = self._hot_expert_set_by_layer.get(layer, self._hot_expert_set)
            gpu_hot = self._backend.gpu_hot_experts(layer)
            for eid in list(layer_set)[:16]:
                if (layer, eid) not in gpu_hot:
                    self._load_expert_fp4_raw(layer, eid)
                    count += 1
                if len(gpu_hot) >= self._max_hot_experts:
                    break
            if len(gpu_hot) >= self._max_hot_experts:
                break
        self._log(f"Pre-loaded {count} hot expert FP4 weights on GPU")

    def _preload_hot_experts_cpu_cache(self):
        hot_ids = set(self._hot_expert_ids)
        hot_ids.update(self._hash_expert_ids)
        count = 0
        for layer_idx in range(self.config.num_hidden_layers):
            layer_hot = self._hot_expert_set_by_layer.get(layer_idx, set(self._hot_expert_ids))
            layer_ids = hot_ids if layer_idx < self.config.num_hash_layers else layer_hot
            for eid in layer_ids:
                cache_key = f"{layer_idx}_{eid}"
                if self.expert_cache.get(cache_key) is not None:
                    continue
                prefix = f"layers.{layer_idx}.ffn.experts.{eid}"
                keys = [f"{prefix}.w1.weight", f"{prefix}.w1.scale",
                        f"{prefix}.w3.weight", f"{prefix}.w3.scale",
                        f"{prefix}.w2.weight", f"{prefix}.w2.scale"]
                tensors = self.loader.get_weights(*keys)
                w1 = tensors.get(keys[0]); s1 = tensors.get(keys[1])
                w3 = tensors.get(keys[2]); s3 = tensors.get(keys[3])
                w2 = tensors.get(keys[4]); s2 = tensors.get(keys[5])
                if w1 is None:
                    continue
                w1_entry = self._make_raw_entry(w1, s1)
                w3_entry = self._make_raw_entry(w3, s3)
                w2_entry = self._make_raw_entry(w2, s2)
                self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=True)
                # P2: EP+numa_aware → also store NUMA-local copy in per-GPU cache.
                # Use pin=False so the cache stays within max_experts (LRU eviction).
                if self._expert_caches and self.hw_config.ep_numa_aware:
                    self._ensure_backend()
                    if self._backend.strategy == "ep":
                        n = self._backend.n_gpu()
                        gpu_cache = self._expert_caches.get(eid % n)
                        if gpu_cache is not None and gpu_cache.get(cache_key) is None:
                            numa_map = getattr(self.hw_profile, 'numa_map', {})
                            gpu_pci_idx = int(str(self._backend.devices[eid % n]).split(":")[1])
                            numa_node = numa_map.get(str(gpu_pci_idx), numa_map.get(gpu_pci_idx, -1))
                            w1_numa = self._make_raw_entry_numa(w1, s1, numa_node)
                            w3_numa = self._make_raw_entry_numa(w3, s3, numa_node)
                            w2_numa = self._make_raw_entry_numa(w2, s2, numa_node)
                            if w1_numa is not None:
                                gpu_cache.put(cache_key, w1_numa, w3_numa, w2_numa, pin=False)
                count += 1
        if count > 0:
            self._log(f"Pre-loaded {count} hot expert raw entries into CPU cache "
                      f"({count * 48 // 1024} MB packed)")

    def _load_global_weights(self):
        def safe(key):
            return self.loader.get_weight(key, device=self.device)

        self.embed = safe("embed.weight")
        if self.embed is not None and not isinstance(self.embed, torch.Tensor):
            self.embed = None
        self.embed = self.embed.to(torch.bfloat16) if self.embed is not None else None
        self._log(f"embed.weight: {self.embed.shape if self.embed is not None else 'missing'}")

        if self.config.tie_word_embeddings and self.embed is not None:
            self.lm_head = self.embed
            self._log("lm_head: tied to embed.weight (shared)")
        else:
            head = safe("head.weight")
            output = safe("output.weight")
            if head is not None:
                self.lm_head = head.to(torch.bfloat16)
            elif output is not None:
                self.lm_head = output.to(torch.bfloat16)
            else:
                self.lm_head = self.embed
        self._log(f"lm_head: {self.lm_head.shape if self.lm_head is not None else 'missing'}")

        norm = safe("norm.weight")
        self.norm_weight = norm.to(torch.bfloat16) if norm is not None else None
        self._log(f"norm.weight: {self.norm_weight.shape if self.norm_weight is not None else 'missing'}")

        self.hc_head_fn = safe("hc_head_fn")
        self.hc_head_base = safe("hc_head_base")
        self.hc_head_scale = safe("hc_head_scale")
        if self.hc_head_fn is not None:
            self.hc_head_fn = self.hc_head_fn.to(torch.float32)
        if self.hc_head_base is not None:
            self.hc_head_base = self.hc_head_base.to(torch.float32)
        if self.hc_head_scale is not None:
            self.hc_head_scale = self.hc_head_scale.to(torch.float32)
        self._log(f"hc_head_fn: {self.hc_head_fn.shape if self.hc_head_fn is not None else 'missing'}")

    def _resolve_device(self, layer_idx: int) -> str:
        """返回第 layer_idx 层所在的 device 字符串。"""
        self._ensure_backend()
        return self._backend.layer_device(layer_idx)

    def _get_layer_weights(self, layer_idx: int, device: str | None = None):
        if layer_idx in self._layer_weight_cache:
            return self._layer_weight_cache[layer_idx]
        lw = {}
        if device is None:
            device = self._resolve_device(layer_idx) if self._is_multigpu \
                else str(self.device)
        keys_needed = [
            f"layers.{layer_idx}.{t}" for t in [
                "attn_norm.weight", "ffn_norm.weight",
                "attn.wq_a.weight", "attn.wq_a.scale",
                "attn.wq_b.weight", "attn.wq_b.scale",
                "attn.wkv.weight", "attn.wkv.scale",
                "attn.wo_a.weight", "attn.wo_a.scale",
                "attn.wo_b.weight", "attn.wo_b.scale",
                "attn.q_norm.weight", "attn.kv_norm.weight",
                "attn.attn_sink",
                "ffn.gate.weight", "ffn.gate.scale",
                "ffn.gate.bias",
                "ffn.gate.tid2eid",
                "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
                "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
                "attn.compressor.wkv.weight",
                "attn.compressor.wgate.weight",
                "attn.compressor.norm.weight",
                "attn.compressor.ape",
                "attn.indexer.wq_b.weight", "attn.indexer.wq_b.scale",
                "attn.indexer.weights_proj.weight",
                "attn.indexer.compressor.wkv.weight",
                "attn.indexer.compressor.wgate.weight",
                "attn.indexer.compressor.norm.weight",
                "attn.indexer.compressor.ape",
            ]
        ]

        present = self.loader.get_weights(*keys_needed)
        target = torch.device(device)
        for full_key in keys_needed:
            short_key = full_key.replace(f"layers.{layer_idx}.", "")
            if full_key in present and present[full_key] is not None:
                t = present[full_key].to(target, non_blocking=True)
                if short_key in ("attn_norm.weight", "ffn_norm.weight") and t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                lw[short_key] = t
        self._layer_weight_cache[layer_idx] = lw
        return lw

    def _get_compressor(self, layer_idx: int, lw: dict) -> NewCompressor | None:
        if not hasattr(self, '_compressors'):
            self._compressors = {}
        if layer_idx in self._compressors:
            return self._compressors[layer_idx]

        compress_ratio = self.config.get_compress_ratio(layer_idx)
        if compress_ratio == 0:
            return None

        c_wkv = lw.get("attn.compressor.wkv.weight")
        c_wgate = lw.get("attn.compressor.wgate.weight")
        c_ape = lw.get("attn.compressor.ape")
        c_norm = lw.get("attn.compressor.norm.weight")

        if c_wkv is None:
            return None

        overlap = compress_ratio == 4
        coff = 2 if overlap else 1
        head_dim = self.config.head_dim

        if c_wkv.dtype not in (torch.bfloat16, torch.float32):
            c_wkv = c_wkv.to(torch.float32)
        if c_wgate.dtype not in (torch.bfloat16, torch.float32):
            c_wgate = c_wgate.to(torch.float32)
        if c_ape is not None and c_ape.dtype != torch.float32:
            c_ape = c_ape.to(torch.float32)

        layer_dev = self._resolve_device(layer_idx)
        dev = torch.device(layer_dev)
        compressor = NewCompressor(
            ratio=compress_ratio,
            head_dim=coff * head_dim // coff,
            coff=coff,
            ape=c_ape,
            wkv=c_wkv.to(dev) if c_wkv.device.type != dev.type else c_wkv,
            wgate=c_wgate.to(dev) if c_wgate.device.type != dev.type else c_wgate,
            norm_w=c_norm,
            device=layer_dev,
        )
        self._compressors[layer_idx] = compressor
        return compressor

    def _get_indexer(self, layer_idx: int, lw: dict) -> LightningIndexer | None:
        if not hasattr(self, '_indexers'):
            self._indexers = {}
        if layer_idx in self._indexers:
            return self._indexers[layer_idx]

        compress_ratio = self.config.get_compress_ratio(layer_idx)
        if compress_ratio != 4:
            return None

        idx_wq_b_data = lw.get("attn.indexer.wq_b.weight")
        if idx_wq_b_data is None:
            return None

        idx_wq_b = self._deq("attn.indexer.wq_b.weight",
                             idx_wq_b_data,
                             lw.get("attn.indexer.wq_b.scale"), layer_idx)
        weights_proj = lw.get("attn.indexer.weights_proj.weight")

        idx_c_wkv = lw.get("attn.indexer.compressor.wkv.weight")
        idx_c_wgate = lw.get("attn.indexer.compressor.wgate.weight")
        idx_c_norm = lw.get("attn.indexer.compressor.norm.weight")
        idx_c_ape = lw.get("attn.indexer.compressor.ape")

        if idx_c_wkv is not None and idx_c_wkv.dtype not in (torch.bfloat16, torch.float32):
            idx_c_wkv = idx_c_wkv.to(torch.float32)
        if idx_c_wgate is not None and idx_c_wgate.dtype not in (torch.bfloat16, torch.float32):
            idx_c_wgate = idx_c_wgate.to(torch.float32)
        if idx_c_ape is not None and idx_c_ape.dtype != torch.float32:
            idx_c_ape = idx_c_ape.to(torch.float32)

        layer_dev = self._resolve_device(layer_idx)
        dev = torch.device(layer_dev)
        indexer = LightningIndexer(
            index_n_heads=self.config.index_n_heads,
            index_head_dim=self.config.index_head_dim,
            index_topk=self.config.index_topk,
            compress_ratio=compress_ratio,
            q_lora_rank=self.config.q_lora_rank,
            device=layer_dev,
        )
        indexer.set_weights(
            wq_b=idx_wq_b.to(dev) if idx_wq_b.device.type != dev.type else idx_wq_b,
            weights_proj=(
                weights_proj.to(dev)
                if weights_proj is not None and weights_proj.device.type != dev.type
                else weights_proj
            ),
            compressor_wkv=(
                idx_c_wkv.to(dev)
                if idx_c_wkv is not None and idx_c_wkv.device.type != dev.type
                else idx_c_wkv
            ),
            compressor_wgate=(
                idx_c_wgate.to(dev)
                if idx_c_wgate is not None and idx_c_wgate.device.type != dev.type
                else idx_c_wgate
            ),
            compressor_norm=idx_c_norm,
            compressor_ape=(
                idx_c_ape.to(dev)
                if idx_c_ape is not None and idx_c_ape.device.type != dev.type
                else idx_c_ape
            ),
        )
        self._indexers[layer_idx] = indexer
        return indexer

    def _deq(self, name, data, scale, layer_idx=None):
        if data is None:
            return None
        if layer_idx is not None:
            cache_key = (layer_idx, name)
            cached = self._deq_cache.get(cache_key)
            if cached is not None:
                self._deq_cache.move_to_end(cache_key)
                return cached
        if data.dtype == torch.bfloat16:
            result = data
        elif data.dtype == torch.float8_e4m3fn:
            result = load_fp8_weight(data, scale)
        elif data.dtype == torch.int8:
            result = load_fp4_weight(data, scale)
        else:
            result = data.to(torch.bfloat16)
        if layer_idx is not None:
            if len(self._deq_cache) >= 16:
                self._deq_cache.popitem(last=False)
            self._deq_cache[cache_key] = result
        return result

    def _compress_kv(self, hidden: torch.Tensor, lw: dict, layer_idx: int, state: LayerState):
        """Compress KV using the paper-aligned Compressor.

        Stores compressed KV in both LayerState (flat tensors) and
        HybridKVCache (Phase 2 paged cache).
        """
        compress_ratio = self.config.get_compress_ratio(layer_idx)
        if compress_ratio == 0:
            return state

        compressor = self._get_compressor(layer_idx, lw)
        if compressor is None:
            return state

        start_pos = getattr(self, '_global_pos', 0)
        compressed = compressor.compress(hidden, start_pos)
        if compressed is None:
            return state

        # Flat tensor storage on LayerState
        if state.compressed_kv_data is None:
            state.compressed_kv_data = compressed.squeeze(0)  # [num_blocks, head_dim]
        else:
            state.compressed_kv_data = torch.cat(
                [state.compressed_kv_data, compressed.squeeze(0)], dim=0)

        # Phase 2: HybridKVCache (paged, block-aligned)
        hybrid = self._get_hybrid_kv(layer_idx)
        for t in range(compressed.shape[1]):
            entry = compressed[:, t:t + 1, :]
            hybrid.append_compressed(entry.squeeze(0))

        return state

    def _get_hybrid_kv(self, layer_idx: int) -> HybridKVCache:
        if not hasattr(self, '_hybrid_kv'):
            self._hybrid_kv = {}
        if layer_idx not in self._hybrid_kv:
            compress_ratio = self.config.get_compress_ratio(layer_idx)
            indexer_dim = getattr(self.config, 'index_head_dim', 128) if compress_ratio == 4 else 0
            self._hybrid_kv[layer_idx] = HybridKVCache(
                compress_ratio=compress_ratio,
                head_dim=self.config.head_dim,
                indexer_dim=indexer_dim,
                device=self._resolve_device(layer_idx),
            )
        return self._hybrid_kv[layer_idx]

    def _compute_indexer(self, q_latent: torch.Tensor, hidden_states: torch.Tensor,
                         lw: dict, state: LayerState, layer_idx: int) -> torch.Tensor:
        """Compute top-k compressed KV indices using the LightningIndexer.

        Falls back to old direct computation when indexer weights are incomplete.
        Returns selected KV tensor [B, 1, k, head_dim] or None.
        """
        indexer = self._get_indexer(layer_idx, lw)
        if indexer is not None:
            return self._compute_indexer_lightning(q_latent, hidden_states, lw, state, layer_idx)

        # Fallback: use old direct computation (backward compat for tests / partial weights)
        compressed_kv = state.compressed_kv_data
        if compressed_kv is None or compressed_kv.shape[0] == 0:
            return None

        idx_wq_b = self._deq("attn.indexer.wq_b.weight",
                              lw.get("attn.indexer.wq_b.weight"),
                              lw.get("attn.indexer.wq_b.scale"), layer_idx)
        if idx_wq_b is None:
            return None

        n_compressed, c_dim = compressed_kv.shape

        idx_q = torch.matmul(q_latent.to(idx_wq_b.dtype), idx_wq_b.t())
        idx_q = idx_q.view(q_latent.shape[0], -1, self.config.index_n_heads, self.config.index_head_dim)
        idx_q = idx_q.transpose(1, 2)

        idx_k = state.compressed_kv_idx
        if idx_k is None or idx_k.shape[0] == 0:
            return self._get_compressed_attention_kv(state)

        idx_k_4d = idx_k.unsqueeze(0).unsqueeze(0)
        i_dim = self.config.index_head_dim
        scores = torch.matmul(
            idx_q.float() * (i_dim ** -0.5),
            idx_k_4d.float().transpose(-2, -1),
        )
        scores_pooled = scores[:, :, -1:, :].mean(dim=1).squeeze(1)

        k = min(self.config.index_topk, n_compressed)
        _, topk_indices = torch.topk(scores_pooled, k, dim=-1)
        selected_indices = topk_indices if topk_indices.dim() == 2 else topk_indices.squeeze(0)

        selected_kv = compressed_kv[selected_indices[0]]
        h_dim = self.config.head_dim
        if c_dim != h_dim:
            if c_dim > h_dim:
                selected_kv = selected_kv.view(-1, c_dim // h_dim, h_dim).mean(dim=1)
            else:
                pad = torch.zeros(k, h_dim - c_dim, device=selected_kv.device, dtype=selected_kv.dtype)
                selected_kv = torch.cat([selected_kv, pad], dim=-1)
        return selected_kv.unsqueeze(0).unsqueeze(1).contiguous()

    def _compute_indexer_lightning(self, q_latent: torch.Tensor, hidden_states: torch.Tensor,
                                    lw: dict, state: LayerState, layer_idx: int) -> torch.Tensor:
        """Compute top-k compressed KV indices using the LightningIndexer.

        Returns selected KV tensor [B, 1, k, head_dim] or None.
        """
        indexer = self._get_indexer(layer_idx, lw)
        if indexer is None:
            return self._get_compressed_attention_kv(state)

        compressed_kv = state.compressed_kv_data
        if compressed_kv is None or compressed_kv.shape[0] == 0:
            return None

        global_pos = getattr(self, '_global_pos', 0)
        win = self.config.sliding_window
        offset = hidden_states.shape[1] if global_pos == 0 else win

        topk_idxs = indexer.compute_indexer(
            hidden_states, q_latent, start_pos=global_pos, offset=offset)
        if topk_idxs is None:
            return self._get_compressed_attention_kv(state)

        # topk_idxs: [B, T, k] with values into the full KV cache, or -1 for invalid
        # For attention, we need to gather the compressed KV entries
        B, T, k = topk_idxs.shape
        h_dim = self.config.head_dim
        c_dim = compressed_kv.shape[-1]

        # Build selected KV: use the compressed KV cache directly
        # topk_idxs values are positions in the full KV (SWA + compressed)
        # For simplicity, gather from compressed KV by subtracting offset
        selected = compressed_kv.unsqueeze(0).unsqueeze(0)  # [1, 1, N, c_dim]
        n_comp = selected.shape[2]

        adjusted_idxs = topk_idxs - offset  # Convert to compressed KV indices
        adjusted_idxs = adjusted_idxs.clamp(0, n_comp - 1)

        # For the last token position only (decode), gather selected KV
        gather_idx = adjusted_idxs[0, -1:, :]  # [1, k]

        result = compressed_kv[gather_idx[0].clamp(0, n_comp - 1)]  # [k, c_dim]
        valid = (gather_idx[0] >= 0) & (gather_idx[0] < n_comp)
        result[~valid] = 0

        if c_dim != h_dim:
            if c_dim > h_dim:
                result = result.view(-1, c_dim // h_dim, h_dim).mean(dim=1)
            else:
                pad = torch.zeros(k, h_dim - c_dim, device=result.device, dtype=result.dtype)
                result = torch.cat([result, pad], dim=-1)

        return result.unsqueeze(0).unsqueeze(1).contiguous()

    def _get_compressed_attention_kv(self, state: LayerState, lw=None) -> torch.Tensor | None:
        """Get all compressed KV entries for attention (HCA fallback)."""
        compressed_kv = state.compressed_kv_data
        if compressed_kv is None or compressed_kv.shape[0] == 0:
            return None
        c_dim = compressed_kv.shape[-1]
        h_dim = self.config.head_dim
        if c_dim != h_dim:
            if c_dim > h_dim:
                compressed_kv = compressed_kv.view(compressed_kv.shape[0], -1, h_dim).mean(dim=1)
            else:
                pad = torch.zeros(compressed_kv.shape[0], h_dim - c_dim,
                                  device=compressed_kv.device, dtype=compressed_kv.dtype)
                compressed_kv = torch.cat([compressed_kv, pad], dim=-1)
        compressed_kv = compressed_kv.unsqueeze(0).unsqueeze(1).contiguous()
        return compressed_kv

    def _expand_kv(self, kv_latent):
        B, T, _ = kv_latent.shape
        k = kv_latent.unsqueeze(2).transpose(1, 2)
        return k, k

    def _forward_mhc(self, hidden_4d: torch.Tensor, hc_base: torch.Tensor,
                     hc_fn: torch.Tensor, hc_scale: torch.Tensor, apply_pre: bool = True):
        B, T, hc_mult, D = hidden_4d.shape
        fn_in_features = hc_fn.shape[-1]
        expected_in = D * hc_mult
        if fn_in_features != expected_in:
            self._log(f"mHC skip: hc_fn needs {fn_in_features}-d input, hidden is {D}-d "
                      f"(expected expanded to {expected_in})")
            return hidden_4d.sum(dim=2), None, None
        dev = hidden_4d.device
        hc_fn = hc_fn.to(dev)
        hc_base = hc_base.to(dev)
        hc_scale = hc_scale.to(dev)

        # PyTorch fallback (mHC tilelang kernel removed)
        hidden_flat = hidden_4d.reshape(B, T, expected_in).float()
        rsqrt = torch.rsqrt(hidden_flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = torch.matmul(hidden_flat * rsqrt, hc_fn.float().t())
        pre, post, comb = mhc_split_sinkhorn(
            mixes, hc_scale.to(torch.float32), hc_base.to(torch.float32),
            hc_mult=hc_mult, sinkhorn_iters=self.config.hc_sinkhorn_iters, eps=self.config.hc_eps,
        )
        if apply_pre:
            scaled = hidden_4d * pre.unsqueeze(-1)
            hidden = scaled.sum(dim=2)
        else:
            hidden = hidden_4d.sum(dim=2)
        return hidden.to(hidden_4d.dtype), post, comb

    def _forward_attn(self, hidden_states, lw, layer_idx):
        B, T, D = hidden_states.shape

        wq_a = self._deq("attn.wq_a.weight", lw.get("attn.wq_a.weight"), lw.get("attn.wq_a.scale"), layer_idx)
        wq_b = self._deq("attn.wq_b.weight", lw.get("attn.wq_b.weight"), lw.get("attn.wq_b.scale"), layer_idx)
        wkv = self._deq("attn.wkv.weight", lw.get("attn.wkv.weight"), lw.get("attn.wkv.scale"), layer_idx)
        wo_a = self._deq("attn.wo_a.weight", lw.get("attn.wo_a.weight"), lw.get("attn.wo_a.scale"), layer_idx)
        wo_b = self._deq("attn.wo_b.weight", lw.get("attn.wo_b.weight"), lw.get("attn.wo_b.scale"), layer_idx)
        q_norm = lw.get("attn.q_norm.weight")
        kv_norm = lw.get("attn.kv_norm.weight")
        attn_sink = lw.get("attn.attn_sink")
        if q_norm is not None:
            q_norm = q_norm.to(torch.bfloat16)
        if kv_norm is not None:
            kv_norm = kv_norm.to(torch.bfloat16)
        if attn_sink is not None:
            attn_sink = attn_sink.to(torch.bfloat16)

        if wq_a is None or wq_b is None or wkv is None:
            return hidden_states

        q_latent = torch.matmul(hidden_states.to(wq_a.dtype), wq_a.t())
        if q_norm is not None:
            q_latent = rms_norm(q_latent, q_norm)
        q = torch.matmul(q_latent, wq_b.t())
        q = q.view(B, T, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)

        kv_latent = torch.matmul(hidden_states.to(wkv.dtype), wkv.t())
        if kv_norm is not None:
            kv_latent = rms_norm(kv_latent, kv_norm)

        compress_ratio = self.config.get_compress_ratio(layer_idx)
        if compress_ratio > 0:
            rope_theta = self.config.compress_rope_theta
            rope_original_seq_len = self.config.rope_scaling_original_max_position_embeddings
            rope_factor = self.config.rope_scaling_factor
            rope_beta_fast = self.config.rope_scaling_beta_fast
            rope_beta_slow = self.config.rope_scaling_beta_slow
        else:
            rope_theta = self.config.rope_theta
            rope_original_seq_len = 0
            rope_factor = 1.0
            rope_beta_fast = 32
            rope_beta_slow = 1
        rope_dim = self.config.qk_rope_head_dim
        start_pos = getattr(self, '_global_pos', 0)
        freqs_cis = precompute_freqs_cis(
            rope_dim, T, theta=rope_theta,
            original_seq_len=rope_original_seq_len, factor=rope_factor,
            beta_fast=rope_beta_fast, beta_slow=rope_beta_slow,
            start_pos=start_pos,
        ).to(q.device)
        q = apply_rotary_emb(q, freqs_cis, rd=self.config.qk_rope_head_dim)
        kv_latent = apply_rotary_emb(kv_latent, freqs_cis, rd=self.config.qk_rope_head_dim)

        # FP8 simulation on non-RoPE KV dims (QAT, matches demo act_quant)
        if kv_latent.shape[-1] > rope_dim:
            _fp8_simulate(kv_latent[..., :-rope_dim], block_size=64)

        layer_dev = self._resolve_device(layer_idx) if self._is_multigpu else str(self.device)
        state = self.layer_states.setdefault(layer_idx, LayerState(device=layer_dev))
        state.append_kv(kv_latent)
        all_kv_latent = state.all_kv()

        attn_type = "csa" if compress_ratio == 4 else ("hca" if compress_ratio > 0 else "swa")

        if attn_type in ("csa", "hca"):
            state = self._compress_kv(hidden_states, lw, layer_idx, state)

        T_kv = all_kv_latent.shape[1]
        sw = min(self.config.sliding_window, T_kv)
        kv_sw = all_kv_latent[:, -sw:, :]
        k_sw, v_sw = self._expand_kv(kv_sw)

        if attn_type == "csa":
            indexed_kv = self._compute_indexer(q_latent, hidden_states, lw, state, layer_idx)
            if indexed_kv is None:
                compressed_kv = self._get_compressed_attention_kv(state)
            else:
                compressed_kv = indexed_kv
        elif attn_type == "hca":
            compressed_kv = self._get_compressed_attention_kv(state)
        else:
            compressed_kv = None
            state.compressed_kv_data = None

        if compressed_kv is not None:
            if compressed_kv.dim() < k_sw.dim():
                compressed_kv = compressed_kv.unsqueeze(1).expand(-1, k_sw.shape[1], -1, -1)
            if compressed_kv.shape[1] != k_sw.shape[1]:
                compressed_kv = compressed_kv.expand(-1, k_sw.shape[1], -1, -1)
            if compressed_kv.shape[3] != k_sw.shape[3]:
                compressed_kv = compressed_kv[:, :, :, :k_sw.shape[3]]
            k_all = torch.cat([k_sw, compressed_kv.to(k_sw.dtype)], dim=-2)
        else:
            k_all = k_sw

        if self._use_gqa_fusion:
            from home_seek.gqa_attention import gqa_fused_attn

            _has_sink = attn_sink is not None and attn_sink.numel() == self.config.num_attention_heads
            _causal_mask = None
            if T > 1 and not _has_sink:
                _k_sw_len = k_sw.shape[-2]
                _new_start = _k_sw_len - T
                if _new_start >= 0:
                    _triu_full = torch.full((T, T), float('-inf'), device=q.device, dtype=torch.float32)
                    _triu = torch.triu(_triu_full, diagonal=1)
                    _causal_mask = torch.zeros(T, k_all.shape[-2], device=q.device, dtype=torch.float32)
                    _causal_mask[:, _new_start:_k_sw_len] = _triu

            out = gqa_fused_attn(
                q.float(),
                k_all.to(q.dtype),
                causal_mask=_causal_mask,
                attn_sink=attn_sink.float() if _has_sink else None,
            ).to(torch.bfloat16)
        else:
            n_kv = self.config.num_key_value_heads
            n_groups = self.config.num_attention_heads // n_kv
            scale_f = self.config.head_dim ** -0.5
            v_all = k_all

            k_exp = k_all.unsqueeze(1).expand(-1, n_groups, -1, -1, -1)
            k_expanded = k_exp.reshape(B, -1, k_all.shape[-2], k_all.shape[-1])
            v_exp = v_all.unsqueeze(1).expand(-1, n_groups, -1, -1, -1)
            v_expanded = v_exp.reshape(B, -1, v_all.shape[-2], v_all.shape[-1])

            attn = torch.matmul(q.float() * scale_f, k_expanded.float().transpose(-2, -1))

            if T > 1 and not (attn_sink is not None and attn_sink.numel() == self.config.num_attention_heads):
                k_sw_len = k_sw.shape[-2]
                new_start = k_sw_len - T
                if new_start >= 0:
                    triu = torch.triu(torch.full((T, T), float('-inf'), device=q.device, dtype=attn.dtype), diagonal=1)
                    mask = torch.zeros(T, k_all.shape[-2], device=q.device, dtype=attn.dtype)
                    mask[:, new_start:k_sw_len] = triu
                    attn = attn + mask

            if attn_sink is not None and attn_sink.numel() == self.config.num_attention_heads:
                sink_val = attn_sink.view(1, -1, 1, 1).to(attn.dtype)
                attn_with_sink = torch.cat([attn, sink_val.expand(-1, -1, T, -1)], dim=-1)
                P_all = F.softmax(attn_with_sink, dim=-1)
                attn_p = P_all[:, :, :, :-1].to(v_expanded.dtype)
            else:
                attn_p = F.softmax(attn, dim=-1).to(v_expanded.dtype)
            out = torch.matmul(attn_p, v_expanded)

        out = apply_rotary_emb(out, freqs_cis, rd=self.config.qk_rope_head_dim, inverse=True)

        out = out.transpose(1, 2).contiguous()

        if wo_a is not None and wo_b is not None:
            out_g = out.view(B, T, self.config.o_groups, -1)
            wo_a_g = wo_a.view(self.config.o_groups, self.config.o_lora_rank, self.config.hidden_size)
            out_combined = torch.einsum('btgd,grd->btgr', out_g.to(wo_a_g.dtype), wo_a_g)
            out_combined = out_combined.reshape(B, T, -1)
            out = torch.matmul(out_combined.to(wo_b.dtype), wo_b.t())
        else:
            out = out.view(B, T, self.config.num_attention_heads * self.config.head_dim)
        return out.to(hidden_states.dtype)

    def _compute_hash_experts(self, input_ids: torch.Tensor, layer_idx: int, tid2eid: torch.Tensor):
        B, T = input_ids.shape
        eids = tid2eid[input_ids]
        weights = torch.full((B, T, self.config.num_experts_per_tok), 1.0 / self.config.num_experts_per_tok,
                             device=input_ids.device)
        return eids, weights

    def _compute_routing_experts(self, hidden_states, gate_w, gate_bias):
        from home_seek.router import compute_expert_affinity_with_bias
        return compute_expert_affinity_with_bias(
            hidden_states, gate_w, gate_bias,
            top_k=self.config.num_experts_per_tok,
            routed_scaling_factor=self.config.routed_scaling_factor,
        )

    def _make_raw_entry_numa(self, data, scale, numa_node: int = -1):
        """Like _make_raw_entry but binds thread to numa_node before pin_memory().

        Ensures allocated pinned pages are local to the given NUMA node,
        reducing cross-NUMA DMA latency when the target GPU reads them.
        Non-FP4 paths (GPU-bound) are unaffected by the binding.
        """
        if numa_node < 0 or data is None or data.dtype != torch.int8:
            return self._make_raw_entry(data, scale)
        import os
        from home_seek.topology_prober import bind_thread_to_numa
        orig = os.sched_getaffinity(0)
        try:
            bind_thread_to_numa(numa_node)
            return self._make_raw_entry(data, scale)
        finally:
            try:
                os.sched_setaffinity(0, orig)
            except Exception:
                pass

    def _make_raw_entry(self, data, scale):
        if data is None:
            return None
        if data.dtype == torch.bfloat16 or data.dtype == torch.float32:
            data = data.to(self.device, non_blocking=True)
            return (data, None, "bf16")
        if data.dtype == torch.float8_e4m3fn:
            data = data.to(self.device, non_blocking=True)
            sf = scale.to(torch.float32).to(self.device, non_blocking=True) if scale is not None else None
            return (data, sf, "fp8")
        if data.dtype == torch.int8:
            same_device = (data.device.type == self.device.type and
                           (data.device.index == self.device.index or
                            self.device.index is None and data.device.index == 0))
            if same_device:
                scale_gpu = scale.to(torch.float32) if scale is not None and scale.dtype != torch.float32 else scale
                return (data, scale_gpu, "fp4_gpu")
            dev = "cpu"
            data = data.to(dev).pin_memory()
            if scale is not None:
                if scale.dtype == torch.float8_e8m0fnu:
                    scale = scale.to(dev).pin_memory()
                elif scale.dtype != torch.float32:
                    scale = scale.to(torch.float32).to(dev).pin_memory()
                else:
                    scale = scale.to(dev).pin_memory()
            return (data, scale, "fp4")
        data = data.to(self.device, non_blocking=True)
        return (data, scale, "bf16")

    def _load_expert_weights(self, layer_idx, eid):
        cache_key = f"{layer_idx}_{eid}"
        cached = self.expert_cache.get(cache_key)
        if cached is not None:
            return cached

        prefix = f"layers.{layer_idx}.ffn.experts.{eid}"
        keys = [f"{prefix}.w1.weight", f"{prefix}.w1.scale",
                f"{prefix}.w3.weight", f"{prefix}.w3.scale",
                f"{prefix}.w2.weight", f"{prefix}.w2.scale"]
        tensors = self.loader.get_weights(*keys)
        w1 = tensors.get(keys[0]); s1 = tensors.get(keys[1])
        w3 = tensors.get(keys[2]); s3 = tensors.get(keys[3])
        w2 = tensors.get(keys[4]); s2 = tensors.get(keys[5])

        if w1 is None:
            return None

        w1_entry = self._make_raw_entry(w1, s1)
        w3_entry = self._make_raw_entry(w3, s3)
        w2_entry = self._make_raw_entry(w2, s2)

        layer_hot = self._hot_expert_set_by_layer.get(layer_idx, self._hot_expert_set)
        pin = layer_idx < self.config.num_hash_layers or eid in layer_hot
        self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=pin)
        # P2: EP+numa_aware → per-GPU cache with NUMA-local pinned copies.
        # Allocates pinned memory on the correct NUMA node during initial load
        # from disk, avoiding cross-NUMA DMA when GPU1 reads it.
        if self._expert_caches:
            self._ensure_backend()
            if self._backend.strategy == "ep":
                n = self._backend.n_gpu()
                gpu_cache = self._expert_caches.get(eid % n)
                if gpu_cache is not None and gpu_cache.get(cache_key) is None:
                    if self.hw_config.ep_numa_aware:
                        numa_map = getattr(self.hw_profile, 'numa_map', {})
                        gpu_pci_idx = int(str(self._backend.devices[eid % n]).split(":")[1])
                        numa_node = numa_map.get(str(gpu_pci_idx), numa_map.get(gpu_pci_idx, -1))
                        w1_numa = self._make_raw_entry_numa(w1, s1, numa_node)
                        w3_numa = self._make_raw_entry_numa(w3, s3, numa_node)
                        w2_numa = self._make_raw_entry_numa(w2, s2, numa_node)
                        if w1_numa is not None:
                            gpu_cache.put(cache_key, w1_numa, w3_numa, w2_numa, pin=pin)
                    else:
                        gpu_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=pin)
        return self.expert_cache.get(cache_key)

    def _load_expert_raw(self, layer_idx, eid):
        raw = self._load_expert_weights(layer_idx, eid)
        if raw is None:
            return None
        return raw

    def _load_expert_deq(self, layer_idx, eid):
        raw = self._load_expert_weights(layer_idx, eid)
        if raw is None:
            return None
        deq = self.expert_cache.deq(f"{layer_idx}_{eid}")
        if deq is None:
            deq = raw
        return deq

    def _migrate_gpu_caches(self):
        """一次性迁移旧 _gpu_hot_experts / _gpu_bf16_cache 到 backend per-device state。
        只在第一次需要时执行。
        """
        if not getattr(self, '_gpu_hot_migrated', False):
            self._gpu_hot_migrated = True
            if hasattr(self, '_gpu_hot_experts') and self._backend is not None:
                dev0 = self._backend.devices[0]
                gpu_hot = self._backend.get_device_state(dev0).gpu_hot_experts
                if self._gpu_hot_experts is not gpu_hot:
                    for k, v in self._gpu_hot_experts.items():
                        if k not in gpu_hot:
                            gpu_hot[k] = v
                    self._gpu_hot_experts = gpu_hot
        if not getattr(self, '_gpu_bf16_migrated', False):
            self._gpu_bf16_migrated = True
            if hasattr(self, '_gpu_bf16_cache') and self._backend is not None:
                dev0 = self._backend.devices[0]
                gpu_bf16 = self._backend.get_device_state(dev0).gpu_bf16_cache
                if self._gpu_bf16_cache is not gpu_bf16:
                    for k, v in self._gpu_bf16_cache.items():
                        if k not in gpu_bf16:
                            gpu_bf16[k] = v
                    self._gpu_bf16_cache = gpu_bf16

    def _load_expert_fp4_raw(self, layer_idx, eid):
        self._ensure_backend()
        self._migrate_gpu_caches()
        hot_key = (layer_idx, eid)
        expert_dev = self._backend.resolve_expert_device(layer_idx, eid)
        n_gpu = self._backend.n_gpu()
        is_ep = self._is_multigpu and self._backend.strategy == "ep"

        # EP 亲和性: 搜索 all GPUs 的缓存, 但只返回当前 device 的 entry
        # (V21.11 fix: 跳过其他 GPU 的 entry, 避免 FP4 packed tensor
        # 在 GPU0 显存上、GPU1 的 Triton dequant 无法跨卡读取)
        if is_ep:
            cur_dev_idx = torch.cuda.current_device()
            for di, d in enumerate(self._backend.devices):
                if di != cur_dev_idx:
                    continue
                st = self._backend.get_device_state(d)
                if hot_key in st.gpu_hot_experts:
                    return st.gpu_hot_experts[hot_key]
                if hot_key in st.gpu_bf16_cache:
                    st.gpu_bf16_cache.move_to_end(hot_key)
                    return st.gpu_bf16_cache[hot_key]
            # 未在本地 GPU 缓存, 也从其他 GPU 接过 cached entry
            # 以使 page cache 保持温热
            # 未在任何 GPU 缓存: 加载并存在当前 device 上
            cur_dev_idx = torch.cuda.current_device()
            store_dev = self._backend.devices[cur_dev_idx] if cur_dev_idx < n_gpu else expert_dev
            store_state = self._backend.get_device_state(store_dev)
            gpu_hot = store_state.gpu_hot_experts
            gpu_bf16 = store_state.gpu_bf16_cache
            layer_dev = store_dev
        else:
            gpu_hot = self._backend.get_device_state(expert_dev).gpu_hot_experts
            gpu_bf16 = self._backend.get_device_state(expert_dev).gpu_bf16_cache
            layer_dev = expert_dev

            if hot_key in gpu_hot:
                return gpu_hot[hot_key]
            if hot_key in gpu_bf16:
                gpu_bf16.move_to_end(hot_key)
                return gpu_bf16[hot_key]

        raw = self._load_expert_raw(layer_idx, eid)
        if raw is None:
            return None

        # P2: NUMA-aware — prefer per-GPU cache entry (allocated on correct
        # NUMA node during _load_expert_weights), fall back to shared cache.
        if is_ep and self.hw_config.ep_numa_aware and self._expert_caches:
            layer_dev_str = str(layer_dev)
            target_gpu = int(layer_dev_str.split(":")[1]) if ":" in layer_dev_str else 0
            gpu_cache = self._expert_caches.get(target_gpu)
            if gpu_cache is not None:
                gpu_raw = gpu_cache.get(f"{layer_idx}_{eid}")
                if gpu_raw is not None:
                    raw = gpu_raw

        w1_entry, w3_entry, w2_entry = raw
        if w1_entry is None:
            return None

        w1_data, w1_scale, w1_fmt = w1_entry
        w3_data, w3_scale, w3_fmt = w3_entry
        w2_data, w2_scale, w2_fmt = w2_entry

        is_fp4_gpu = all(f in ("fp4_gpu", "fp4") for f in (w1_fmt, w3_fmt, w2_fmt))
        if not is_fp4_gpu:
            return None

        dev = torch.device(layer_dev)
        with torch.cuda.device(dev):
            w1_dev = w1_data.device if w1_data is not None else torch.device("cpu")
            if w1_data is not None and w1_dev.type != dev.type:
                w1_data = w1_data.to(dev, non_blocking=True)
                if w1_scale is not None:
                    w1_scale = w1_scale.to(dev, non_blocking=True)
            w3_dev = w3_data.device if w3_data is not None else torch.device("cpu")
            if w3_data is not None and w3_dev.type != dev.type:
                w3_data = w3_data.to(dev, non_blocking=True)
                if w3_scale is not None:
                    w3_scale = w3_scale.to(dev, non_blocking=True)
            w2_dev = w2_data.device if w2_data is not None else torch.device("cpu")
            if w2_data is not None and w2_dev.type != dev.type:
                w2_data = w2_data.to(dev, non_blocking=True)
                if w2_scale is not None:
                    w2_scale = w2_scale.to(dev, non_blocking=True)

            if w1_scale is not None:
                w1_scale = (w1_scale.to(torch.float32)
                            if w1_scale.dtype != torch.float32
                            else w1_scale)
            if w3_scale is not None:
                w3_scale = (w3_scale.to(torch.float32)
                            if w3_scale.dtype != torch.float32
                            else w3_scale)
            if w2_scale is not None:
                w2_scale = (w2_scale.to(torch.float32)
                            if w2_scale.dtype != torch.float32
                            else w2_scale)

            fp4_six = (w1_data, w1_scale, w3_data, w3_scale, w2_data, w2_scale)

            layer_hot = self._hot_expert_set_by_layer.get(layer_idx, self._hot_expert_set)
            if eid in layer_hot:
                if len(gpu_hot) >= self._max_hot_experts:
                    gpu_hot.pop(next(iter(gpu_hot)))
                gpu_hot[hot_key] = fp4_six
            else:
                if len(gpu_bf16) >= self._max_bf16_cache:
                    gpu_bf16.pop(next(iter(gpu_bf16)))
                gpu_bf16[hot_key] = fp4_six

        return fp4_six

    def _load_gpu_hot_expert_bf16(self, layer: int, eid: int):
        self._ensure_backend()
        key = (layer, eid)
        gpu_hot = self._backend.gpu_hot_experts(layer)
        if key not in gpu_hot and hasattr(self, '_gpu_hot_experts') and self._gpu_hot_experts is not gpu_hot:
            for k, v in self._gpu_hot_experts.items():
                if k not in gpu_hot:
                    gpu_hot[k] = v
            self._gpu_hot_experts = gpu_hot
        if key in gpu_hot:
            val = gpu_hot[key]
            if len(val) == 6:
                dev_idx = torch.cuda.current_device()
                ck = (eid, dev_idx)
                if ck in self._gpu_bf16_deq_cache:
                    return self._gpu_bf16_deq_cache[ck]
                bf16 = triton_dequantize_fp4_all(*val)
                self._gpu_bf16_deq_cache[ck] = bf16
                if len(self._gpu_bf16_deq_cache) > self._max_gpu_bf16_deq:
                    self._gpu_bf16_deq_cache.pop(next(iter(self._gpu_bf16_deq_cache)))
                return bf16
            return val
        return self._load_bf16_deq(layer, eid)

    def _load_bf16_deq(self, layer: int, eid: int) -> tuple | None:
        """加载专家权重并返回 BF16 3 元组, keyed by (eid, device).

        V21.11 fix: key by (eid, device_index) 防止 EP 多卡时 GPU1
        取到 GPU0 缓存的跨 device tensor 导致 matmul 报错.
        """
        dev_idx = torch.cuda.current_device()
        ck = (eid, dev_idx)
        if ck in self._gpu_bf16_deq_cache:
            self._gpu_bf16_deq_cache.move_to_end(ck)
            return self._gpu_bf16_deq_cache[ck]
        raw = self._load_expert_fp4_raw(layer, eid)
        if raw is None:
            return None
        if len(raw) == 3:
            return raw
        if len(raw) == 6:
            bf16 = triton_dequantize_fp4_all(*raw)
            self._gpu_bf16_deq_cache[ck] = bf16
            if len(self._gpu_bf16_deq_cache) > self._max_gpu_bf16_deq:
                self._gpu_bf16_deq_cache.pop(next(iter(self._gpu_bf16_deq_cache)))
            return bf16
        return None

    def _forward_ffn_hot_batched(self, flat_hidden, flat_topk_idx, flat_topk_w, layer_idx):
        B, D = flat_hidden.shape
        num_topk = flat_topk_idx.shape[1]
        valid_mask = flat_topk_idx >= 0
        valid_eids = flat_topk_idx[valid_mask]
        if valid_eids.numel() == 0:
            return torch.zeros_like(flat_hidden)
        all_eids = sorted(set(int(x) for x in valid_eids.tolist()))
        loaded = {}
        I_dim = None
        for eid in all_eids:
            weights = self._load_gpu_hot_expert_bf16(layer_idx, eid)
            if weights is None:
                return None
            w1, w3, w2 = weights
            if I_dim is None:
                I_dim = w1.shape[0]
            loaded[eid] = (w1, w3, w2)
        loaded_eids = sorted(loaded.keys())
        num_e = len(loaded_eids)
        w1 = torch.cat([loaded[eid][0] for eid in loaded_eids], dim=0)
        w3 = torch.cat([loaded[eid][1] for eid in loaded_eids], dim=0)
        w2 = torch.cat([loaded[eid][2] for eid in loaded_eids], dim=1)
        gate = flat_hidden.to(w1.dtype) @ w1.T
        up = flat_hidden.to(w3.dtype) @ w3.T
        g = gate.float().clamp(max=self.config.swiglu_limit)
        u = up.float().clamp(min=-self.config.swiglu_limit, max=self.config.swiglu_limit)
        activated = (g * g.sigmoid() * u).to(torch.bfloat16)
        # Vectorized routing scatter (no Python loop)
        expert_weights = torch.zeros(B, num_e, device=flat_hidden.device, dtype=torch.bfloat16)
        max_eid = max(loaded_eids)
        eid_to_idx_t = torch.full((max_eid + 1,), -1, device=flat_hidden.device, dtype=torch.int64)
        for i, eid in enumerate(loaded_eids):
            eid_to_idx_t[eid] = i
        clamped = flat_topk_idx.clamp(min=0).long()
        mapped = eid_to_idx_t[clamped]
        v = valid_mask & (mapped >= 0)
        b_idx = torch.arange(B, device=flat_hidden.device).unsqueeze(1).expand(-1, num_topk)
        if v.any():
            expert_weights[b_idx[v], mapped[v]] = flat_topk_w[v].to(torch.bfloat16)
        routing = expert_weights.unsqueeze(-1).expand(-1, -1, I_dim).reshape(B, -1)
        activated_weighted = activated * routing
        return activated_weighted.to(w2.dtype) @ w2.T

    def _all_routed_are_hot(self, topk_idx, layer_idx: int = 0) -> bool:
        layer_set = self._hot_expert_set_by_layer.get(layer_idx, self._hot_expert_set)
        flat = topk_idx.flatten().tolist()
        return all(eid < 0 or eid in layer_set for eid in flat)

    def _forward_ffn_m1_triton(self, hidden_states, flat_hidden, flat_topk_idx, flat_topk_w, layer_idx):
        eids = {}
        idx_row = flat_topk_idx[0].tolist()
        w_row = flat_topk_w[0].tolist()
        for k in range(flat_topk_idx.shape[1]):
            eid = int(idx_row[k])
            if eid >= 0:
                eids[eid] = eids.get(eid, 0.0) + float(w_row[k])
        if not eids:
            return None

        fp4_data = {}
        for eid in eids:
            raw = self._load_expert_fp4_raw(layer_idx, eid)
            if raw is None:
                return None
            fp4_data[eid] = raw

        try:
            result = self._fused_moe.forward_m1_fp4(flat_hidden, fp4_data, eids)
            return result.reshape(hidden_states.shape)
        except Exception:
            return None

    def _forward_ffn(self, hidden_states, lw, layer_idx, input_ids=None):
        if self._is_multigpu and getattr(self._backend, 'strategy', 'pp') == 'ep':
            return self._forward_ffn_ep(hidden_states, lw, layer_idx, input_ids)
        gate_w = self._deq("ffn.gate.weight", lw.get("ffn.gate.weight"), lw.get("ffn.gate.scale"), layer_idx)
        gate_bias = lw.get("ffn.gate.bias")
        tid2eid = lw.get("ffn.gate.tid2eid")

        if gate_w is None:
            return torch.zeros_like(hidden_states), set()

        B, T, D = hidden_states.shape
        total_tokens = B * T
        flat_hidden = hidden_states.reshape(total_tokens, D)

        with record_function(f"ffn_routing_L{layer_idx}"):
            if layer_idx < self.config.num_hash_layers and tid2eid is not None and input_ids is not None:
                if input_ids.device != hidden_states.device:
                    input_ids = input_ids.to(hidden_states.device)
                topk_idx, topk_w = self._compute_hash_experts(input_ids, layer_idx, tid2eid)
            else:
                topk_idx, topk_w = self._compute_routing_experts(flat_hidden, gate_w, gate_bias)

        self._last_routed_eids = sorted(set(
            int(x) for x in topk_idx.flatten().tolist() if x >= 0))

        ffn_out = torch.zeros_like(hidden_states)
        flat_topk_idx = topk_idx.reshape(total_tokens, self.config.num_experts_per_tok)
        flat_topk_w = topk_w.reshape(total_tokens, self.config.num_experts_per_tok)

        if self._hot_expert_set and total_tokens > 0:
            try:
                if self._all_routed_are_hot(flat_topk_idx, layer_idx):
                    with record_function(f"ffn_hot_batched_L{layer_idx}"):
                        hot_result = self._forward_ffn_hot_batched(
                            flat_hidden, flat_topk_idx, flat_topk_w, layer_idx)
                    if hot_result is not None:
                        ffn_out = hot_result.reshape(B, T, D)
                        shared_w = self._get_shared_expert(layer_idx)
                        if shared_w is not None:
                            w1_d, w3_d, w2_d = shared_w
                            with record_function(f"ffn_shared_L{layer_idx}"):
                                shared_out = self._shared_ffn.forward(
                                    hidden_states, w1_d, w3_d, w2_d, hidden_states.dtype)
                            ffn_out = ffn_out + shared_out
                        return ffn_out, set()
            except Exception:
                pass

        if total_tokens == 1 and self._fused_moe.use_triton:
            try:
                with record_function(f"ffn_m1_triton_L{layer_idx}"):
                    m1_result = self._forward_ffn_m1_triton(
                        hidden_states, flat_hidden, flat_topk_idx, flat_topk_w, layer_idx)
                if m1_result is not None:
                    ffn_out = m1_result
                    shared_w = self._get_shared_expert(layer_idx)
                    if shared_w is not None:
                        w1_d, w3_d, w2_d = shared_w
                        with record_function(f"ffn_shared_L{layer_idx}"):
                            shared_out = self._shared_ffn.forward(
                                hidden_states, w1_d, w3_d, w2_d, hidden_states.dtype)
                        ffn_out = ffn_out + shared_out
                    if hasattr(self, 'predictor'):
                        self.predictor.collect(layer_idx, flat_hidden, flat_topk_idx)
                    return ffn_out, set()
            except Exception:
                pass

        try:
            with record_function(f"ffn_fused_moe_L{layer_idx}"):
                def _bf16_load(layer, eid):
                    return self._load_bf16_deq(layer, eid)
                result = self._fused_moe.forward(
                    flat_hidden, flat_topk_idx, flat_topk_w,
                    _bf16_load,
                    layer_idx,
                )
            ffn_out = result.reshape(B, T, D)
        except Exception as _e:
            self._log(f"FusedMoE FP4 path failed at layer {layer_idx}: {type(_e).__name__}")
            try:
                def _deq_load(layer, eid):
                    return self._load_expert_deq(layer, eid)
                result = self._fused_moe.forward(
                    flat_hidden, flat_topk_idx, flat_topk_w,
                    _deq_load,
                    layer_idx,
                )
                ffn_out = result.reshape(B, T, D)
            except Exception as _e2:
                if self.verbose:
                    self._log(f"FusedMoE fallback at layer {layer_idx}: {type(_e2).__name__}")
                used_experts = set()
                for k in range(self.config.num_experts_per_tok):
                    expert_ids = flat_topk_idx[:, k]
                    weights = flat_topk_w[:, k]
                    unique_eids, inverse = torch.unique(expert_ids, return_inverse=True)
                    for eid_idx in range(unique_eids.shape[0]):
                        eid = unique_eids[eid_idx].item()
                        if eid < 0:
                            continue
                        used_experts.add(eid)
                        token_mask = inverse == eid_idx
                        if not token_mask.any():
                            continue
                        deq_triple = self._load_expert_deq(layer_idx, eid)
                        if deq_triple is None:
                            continue
                        w1_d, w3_d, w2_d = deq_triple
                        if w1_d is None:
                            continue
                        h_batch = flat_hidden[token_mask].to(w1_d.dtype)
                        gate_out = torch.matmul(h_batch, w1_d.t())
                        up_out = torch.matmul(h_batch, w3_d.t())
                        g = gate_out.float().clamp(max=self.config.swiglu_limit)
                        u = up_out.float().clamp(min=-self.config.swiglu_limit, max=self.config.swiglu_limit)
                        activated = (g * g.sigmoid() * u).to(w1_d.dtype)
                        out = torch.matmul(activated.to(w2_d.dtype), w2_d.t())
                        ffn_out.reshape(total_tokens, D)[token_mask] += out * weights[token_mask].unsqueeze(-1)

        shared_w = self._get_shared_expert(layer_idx)
        if shared_w is not None:
            w1_d, w3_d, w2_d = shared_w
            with record_function(f"ffn_shared_L{layer_idx}"):
                shared_out = self._shared_ffn.forward(
                    hidden_states, w1_d, w3_d, w2_d, hidden_states.dtype)
            ffn_out = ffn_out + shared_out

        if hasattr(self, 'predictor'):
            self.predictor.collect(layer_idx, flat_hidden, flat_topk_idx)

        return ffn_out, set()

    def _check_expert_cache_affinity(self, layer_idx: int, eid: int) -> int:
        """检查 expert 在哪个 GPU 上有缓存 (FP4 或 BF16 raw).
        返回 device index, 或 -1 (未在任何 GPU 缓存).
        """
        hot_key = (layer_idx, eid)
        for dev_idx, dev in enumerate(self._backend.devices):
            state = self._backend.get_device_state(dev)
            if hot_key in state.gpu_hot_experts or hot_key in state.gpu_bf16_cache:
                return dev_idx
        return -1

    def _check_expert_cache_affinity_batch(self, layer_idx: int, eids: list[int]) -> dict[int, int]:
        """Batch affinity check: all eids checked against all GPUs in one pass."""
        result: dict[int, int] = {}
        pending = list(eids)
        for dev_idx, dev in enumerate(self._backend.devices):
            state = self._backend.get_device_state(dev)
            hot = state.gpu_hot_experts
            bf16 = state.gpu_bf16_cache
            still_pending = []
            for eid in pending:
                if (layer_idx, eid) in hot or (layer_idx, eid) in bf16:
                    result[eid] = dev_idx
                else:
                    still_pending.append(eid)
            pending = still_pending
            if not pending:
                break
        for eid in pending:
            result[eid] = -1
        return result

    def _forward_ffn_ep(self, hidden_states, lw, layer_idx, input_ids=None):
        """EP: GPU0 负责 all, GPU1 负责 expert(eid%2==1). 双线程并行.

        V21.7: EP 亲和性调度 — 根据 expert 缓存的 GPU 动态分配,
        替代 eid%2 硬切分, 提升双卡 cache hit rate.
        """
        B, T, D = hidden_states.shape
        total_tokens = B * T
        flat_hidden = hidden_states.reshape(total_tokens, D)

        with record_function("ep_routing"):
            gate_w = self._deq("ffn.gate.weight", lw.get("ffn.gate.weight"), lw.get("ffn.gate.scale"), layer_idx)
            gate_bias = lw.get("gate.bias")
            tid2eid = lw.get("gate.tid2eid")
        if layer_idx < self.config.num_hash_layers and tid2eid is not None and input_ids is not None:
            if input_ids.device != hidden_states.device:
                input_ids = input_ids.to(hidden_states.device)
            topk_idx, topk_w = self._compute_hash_experts(input_ids, layer_idx, tid2eid)
        else:
            topk_idx, topk_w = self._compute_routing_experts(flat_hidden, gate_w, gate_bias)

        if topk_idx.numel() == 0:
            ffn_out = torch.zeros_like(hidden_states)
            shared_w = self._get_shared_expert(layer_idx)
            if shared_w is not None:
                shared_out = self._shared_ffn.forward(
                    hidden_states, shared_w[0], shared_w[1], shared_w[2], hidden_states.dtype
                )
                ffn_out = ffn_out + shared_out
            return ffn_out, set()

        n_gpu = self._backend.n_gpu()
        with record_function("ep_split"):
            unique_eids = sorted(set(int(x) for x in topk_idx.flatten().tolist() if x >= 0))
            self._last_routed_eids = unique_eids
            affinity = self._check_expert_cache_affinity_batch(layer_idx, unique_eids)
            eid_to_dev: dict[int, int] = {}
            for eid in unique_eids:
                dev_idx = affinity.get(eid, -1)
                if dev_idx < 0:
                    dev_idx = self._ep_affinity_rr % n_gpu
                    self._ep_affinity_rr += 1
                eid_to_dev[eid] = dev_idx

            topk_per_gpu = [topk_idx.clone().fill_(-1) for _ in range(n_gpu)]
            for eid, dev_idx in eid_to_dev.items():
                topk_per_gpu[dev_idx][topk_idx == eid] = eid

            topk_0 = topk_per_gpu[0]
            topk_1 = topk_per_gpu[1] if n_gpu > 1 else topk_idx.clone().fill_(-1)

        with record_function("ep_shared_expert"):
            shared_w = self._get_shared_expert(layer_idx)
            shared_out = (
                self._shared_ffn.forward(
                    hidden_states, shared_w[0], shared_w[1], shared_w[2], hidden_states.dtype
                ) if shared_w is not None else 0
            )

        # ── 启动 GPU1 工作线程 (惰性) ──
        self._backend.ep_start_worker(self)

        # ── 异步拷贝到 GPU1 (GPU0 default stream) ──
        has_gpu1_work = topk_1.max().item() >= 0 if topk_1.numel() > 0 else False

        gpu1_work_item = None
        if has_gpu1_work and self._backend.n_gpu() >= 2:
            with record_function("ep_async_copy_gpu1"):
                dev1 = self._backend.devices[1]
                h_1 = hidden_states.to(dev1, non_blocking=True).reshape(total_tokens, D)
                tk1 = topk_1.to(dev1, non_blocking=True)
                tw = topk_w.to(dev1, non_blocking=True)
                copy_ev = torch.cuda.Event()
                copy_ev.record(torch.cuda.current_stream(self._backend.devices[0]))
            gpu1_work_item = self._backend.ep_submit_work(
                h_1, tk1, tw, layer_idx, copy_ev)

        # ── GPU0: 本卡 expert 计算 ──
        with record_function("ep_gpu0_experts"):
            has_gpu0_work = topk_0.max().item() >= 0 if topk_0.numel() > 0 else False
            if has_gpu0_work:
                self._ep_ev_gpu0_start.record()
                try:
                    ffn_0 = self._fused_moe.forward(
                        flat_hidden, topk_0, topk_w,
                        lambda l, e: self._load_bf16_deq(l, e),
                        layer_idx,
                    )
                except Exception:
                    ffn_0 = torch.zeros(total_tokens, D, device=hidden_states.device, dtype=hidden_states.dtype)
                    for k in range(self.config.num_experts_per_tok):
                        eids_k = topk_0[:, k]
                        w_k = topk_w[:, k]
                        for tok_i in range(total_tokens):
                            eid = int(eids_k[tok_i].item())
                            if eid < 0:
                                continue
                            bf16 = self._load_bf16_deq(layer_idx, eid)
                            if bf16 is None:
                                continue
                            w1_d, w3_d, w2_d = bf16
                            h_tok = flat_hidden[tok_i:tok_i+1].to(w1_d.dtype)
                            gate_out = torch.matmul(h_tok, w1_d.t())
                            up_out = torch.matmul(h_tok, w3_d.t())
                            g = gate_out.float().clamp(max=self.config.swiglu_limit)
                            u = up_out.float().clamp(min=-self.config.swiglu_limit, max=self.config.swiglu_limit)
                            activated = (g * g.sigmoid() * u).to(w1_d.dtype)
                            out = torch.matmul(activated.to(w2_d.dtype), w2_d.t())
                            ffn_0[tok_i] += out[0] * w_k[tok_i]
                self._ep_ev_gpu0_end.record()
            else:
                ffn_0 = torch.zeros(total_tokens, D, device=hidden_states.device, dtype=hidden_states.dtype)
                self._ep_ev_gpu0_end.record()

        # ── 收 GPU1 结果 ──
        with record_function("ep_recv_combine"):
            self._ep_ev_copy_start.record()
            if gpu1_work_item is not None:
                ffn_1 = self._backend.ep_wait_result(gpu1_work_item)
            else:
                ffn_1 = torch.zeros(total_tokens, D, device=hidden_states.device, dtype=hidden_states.dtype)
            torch.cuda.synchronize()
            self._ep_ev_copy_end.record()
            ffn_out = (ffn_0 + ffn_1 + shared_out.reshape(total_tokens, D)).reshape(B, T, D)

        # ── Collect timing ──
        torch.cuda.synchronize()
        gpu0_ms = self._ep_ev_gpu0_start.elapsed_time(self._ep_ev_gpu0_end) if has_gpu0_work else 0
        copy_ms = self._ep_ev_copy_start.elapsed_time(self._ep_ev_copy_end) if has_gpu1_work else 0
        gpu1_ms = getattr(gpu1_work_item, 'gpu1_elapsed_ms', 0) if gpu1_work_item is not None else 0
        self._ep_timing = {'gpu0_ms': gpu0_ms, 'gpu1_ms': gpu1_ms, 'copy_ms': copy_ms}

        if hasattr(self, 'predictor'):
            self.predictor.collect(layer_idx, flat_hidden, topk_idx)
        return ffn_out, set()

    def _process_mhc_layer(self, hidden_4d, lw, prefix: str):
        hc_base = lw.get(f"{prefix}_base")
        hc_fn = lw.get(f"{prefix}_fn")
        hc_scale = lw.get(f"{prefix}_scale")
        if hc_base is None or hc_fn is None or hc_scale is None:
            return hidden_4d.sum(dim=2), None, None
        return self._forward_mhc(hidden_4d, hc_base.to(torch.bfloat16),
                                  hc_fn.to(torch.bfloat16), hc_scale.to(torch.bfloat16))

    def _process_mhc_post(self, hidden, residual, post, comb):
        if post is None or comb is None:
            return hidden
        B, S, D = hidden.shape
        # Use the same PyTorch path for all S (Triton mhc_post kernel always
        # fails with AssertionError in practice). Sequential T=1 verification
        # (used by MTP verify) ensures correct autoregressive causality.
        x_expanded = hidden.unsqueeze(2)
        term1 = post.unsqueeze(-1) * x_expanded
        residual_expanded = residual.unsqueeze(3)
        term2 = torch.sum(comb.unsqueeze(-1) * residual_expanded, dim=2)
        return (term1 + term2).to(hidden.dtype)

    def _hc_head(self, hidden_4d):
        B, T, hc_mult, D = hidden_4d.shape
        x = hidden_4d.reshape(B, T, hc_mult * D).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        dev = hidden_4d.device
        hc_fn = self.hc_head_fn.to(dev)
        mixes = torch.matmul(x, hc_fn.t()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale.to(dev) + self.hc_head_base.to(dev)) + self.config.hc_eps
        return (pre.unsqueeze(-1) * hidden_4d.float()).sum(dim=2).to(torch.bfloat16)

    def _load_shared_experts_gpu(self):
        self._ensure_backend()
        count = 0
        for layer_idx in range(self.config.num_hidden_layers):
            shared_prefix = f"layers.{layer_idx}.ffn.shared_experts"
            shared_keys = [f"{shared_prefix}.w1.weight", f"{shared_prefix}.w1.scale",
                           f"{shared_prefix}.w3.weight", f"{shared_prefix}.w3.scale",
                           f"{shared_prefix}.w2.weight", f"{shared_prefix}.w2.scale"]
            tensors = self.loader.get_weights(*shared_keys)
            w1 = tensors.get(shared_keys[0])
            if w1 is None:
                continue
            s1 = tensors.get(shared_keys[1])
            w3_t = tensors.get(shared_keys[2])
            s3 = tensors.get(shared_keys[3])
            w2_t = tensors.get(shared_keys[4])
            s2 = tensors.get(shared_keys[5])

            # PP: 共享专家在所属层的 device 上; EP: 复制到所有 device
            if self._is_multigpu and getattr(self._backend, 'strategy', 'pp') == 'ep':
                target_devs = self._backend.devices
            else:
                target_devs = (self._resolve_device(layer_idx),)
            for dev_name in target_devs:
                dev = torch.device(dev_name)
                dev_state = self._backend.get_device_state(dev_name)
                dev_state.shared_expert_weights[layer_idx] = (
                    w1.to(dev), s1.to(dev) if s1 is not None else None,
                    w3_t.to(dev) if w3_t is not None else None, s3.to(dev) if s3 is not None else None,
                    w2_t.to(dev) if w2_t is not None else None, s2.to(dev) if s2 is not None else None,
                    "fp8",
                )
            count += 1
        self._shared_experts_loaded = True
        self._log(f"Pre-loaded {count} shared experts as FP8 on GPU")

    def _get_shared_expert(self, layer_idx: int):
        self._ensure_backend()
        shared_cache = self._backend.shared_expert_cache(layer_idx)
        # 向后兼容: __new__ 测试桩可能写入了 self._shared_expert_weights
        if layer_idx not in shared_cache and hasattr(self, '_shared_expert_weights'):
            old = self._shared_expert_weights.get(layer_idx)
            if old is not None:
                shared_cache[layer_idx] = old
        cached = shared_cache.get(layer_idx)
        if cached is not None:
            if len(cached) == 3:
                return cached
            if len(cached) == 7 and cached[6] == "fp8":
                w1_fp8, w1_s, w3_fp8, w3_s, w2_fp8, w2_s, _ = cached
                if w1_fp8 is not None and w1_fp8.device.type != 'cuda':
                    layer_dev = torch.device(self._resolve_device(layer_idx))
                    w1_fp8 = w1_fp8.to(layer_dev)
                    if w1_s is not None:
                        w1_s = w1_s.to(layer_dev)
                    if w3_fp8 is not None:
                        w3_fp8 = w3_fp8.to(layer_dev)
                    if w3_s is not None:
                        w3_s = w3_s.to(layer_dev)
                    if w2_fp8 is not None:
                        w2_fp8 = w2_fp8.to(layer_dev)
                    if w2_s is not None:
                        w2_s = w2_s.to(layer_dev)
                w1_bf = load_fp8_weight(w1_fp8, w1_s) if w1_fp8 is not None else None
                w3_bf = load_fp8_weight(w3_fp8, w3_s) if w3_fp8 is not None else None
                w2_bf = load_fp8_weight(w2_fp8, w2_s) if w2_fp8 is not None else None
                if w1_bf is not None and w3_bf is not None and w2_bf is not None:
                    result = (w1_bf, w3_bf, w2_bf)
                    shared_cache[layer_idx] = result
                    if hasattr(self, '_shared_expert_weights'):
                        self._shared_expert_weights[layer_idx] = result
                    return result
                shared_cache[layer_idx] = None
                return None
        shared_prefix = f"layers.{layer_idx}.ffn.shared_experts"
        shared_keys = [f"{shared_prefix}.w1.weight", f"{shared_prefix}.w1.scale",
                       f"{shared_prefix}.w3.weight", f"{shared_prefix}.w3.scale",
                       f"{shared_prefix}.w2.weight", f"{shared_prefix}.w2.scale"]
        tensors = self.loader.get_weights(*shared_keys)
        w1 = tensors.get(shared_keys[0])
        if w1 is None:
            shared_cache[layer_idx] = None
            return None
        s1 = tensors.get(shared_keys[1])
        w3_t = tensors.get(shared_keys[2])
        s3 = tensors.get(shared_keys[3])
        w2_t = tensors.get(shared_keys[4])
        s2 = tensors.get(shared_keys[5])
        assert w1.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.int8), \
            f"Shared expert w1 unexpected dtype: {w1.dtype}"
        layer_dev = self._resolve_device(layer_idx)
        dev = torch.device(layer_dev)
        def _load_w(data, scale):
            if data is None:
                return None
            if data.dtype == torch.int8:
                return load_fp4_weight(data.to(dev), scale.to(dev) if scale is not None else None)
            return load_fp8_weight(data.to(dev), scale.to(dev) if scale is not None else None)
        w1_d = _load_w(w1, s1)
        w3_d = _load_w(w3_t, s3) if w3_t is not None else None
        w2_d = _load_w(w2_t, s2) if w2_t is not None else None
        if w1_d is not None and w3_d is not None and w2_d is not None:
            weights = (w1_d, w3_d, w2_d)
            shared_cache[layer_idx] = weights
            return weights
        shared_cache[layer_idx] = None
        return None

    def _load_mtp_weights(self):
        """Preload full MTP module weights (attention, projections, gate, experts, MHC)."""
        if not hasattr(self, 'loader') or self.loader is None:
            return
        mtp_keys = []
        # Attention
        for n in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]:
            mtp_keys.append(f"mtp.0.attn.{n}.weight")
            mtp_keys.append(f"mtp.0.attn.{n}.scale")
        for n in ["q_norm", "kv_norm"]:
            mtp_keys.append(f"mtp.0.attn.{n}.weight")
        mtp_keys.append("mtp.0.attn.attn_sink")
        # Norms
        for n in ["attn_norm", "ffn_norm", "norm", "enorm", "hnorm"]:
            mtp_keys.append(f"mtp.0.{n}.weight")
        # Projections
        for n in ["e_proj", "h_proj"]:
            mtp_keys.append(f"mtp.0.{n}.weight")
            mtp_keys.append(f"mtp.0.{n}.scale")
        # FFN gate
        mtp_keys.append("mtp.0.ffn.gate.weight")
        mtp_keys.append("mtp.0.ffn.gate.bias")
        # MHC
        for prefix in ["hc_attn", "hc_ffn", "hc_head"]:
            for suffix in ["base", "fn", "scale"]:
                mtp_keys.append(f"mtp.0.{prefix}_{suffix}")

        loader = self.loader
        tensors = loader.get_weights(*mtp_keys)
        weight_count = 0
        for full_key in mtp_keys:
            if full_key in tensors and tensors[full_key] is not None:
                t = tensors[full_key]
                scale_key = full_key.replace(".weight", ".scale")
                scale_t = tensors.get(scale_key)
                dev = self.device
                if t.dtype == torch.bfloat16 or t.dtype == torch.float32:
                    t_loaded = t.to(dev, non_blocking=True)
                elif t.dtype == torch.float8_e4m3fn:
                    t_dev = t.to(dev, non_blocking=True)
                    s_dev = scale_t.to(dev, non_blocking=True) if scale_t is not None else None
                    t_loaded = load_fp8_weight(t_dev, s_dev)
                elif t.dtype == torch.int8:
                    t_loaded = t.to(torch.bfloat16).to(dev, non_blocking=True)
                else:
                    t_loaded = t.to(dev, non_blocking=True)
                self._mtp_weights[full_key] = t_loaded
                weight_count += 1

        # Preload all 256 MTP experts into CPU ExpertWeightCache (pinned)
        expert_count = 0
        for eid in range(self.config.n_routed_experts):
            cache_key = f"mtp_0_{eid}"
            if self.expert_cache.get(cache_key) is not None:
                continue
            prefix = f"mtp.0.ffn.experts.{eid}"
            keys = [f"{prefix}.w1.weight", f"{prefix}.w1.scale",
                    f"{prefix}.w3.weight", f"{prefix}.w3.scale",
                    f"{prefix}.w2.weight", f"{prefix}.w2.scale"]
            tensors = loader.get_weights(*keys)
            w1 = tensors.get(keys[0]); s1 = tensors.get(keys[1])
            w3 = tensors.get(keys[2]); s3 = tensors.get(keys[3])
            w2 = tensors.get(keys[4]); s2 = tensors.get(keys[5])
            if w1 is None:
                continue
            s1_p = s1.to(torch.float32).to("cpu").pin_memory() if s1 is not None else None
            s3_p = s3.to(torch.float32).to("cpu").pin_memory() if s3 is not None else None
            s2_p = s2.to(torch.float32).to("cpu").pin_memory() if s2 is not None else None
            w1_entry = (w1.to("cpu").pin_memory(), s1_p, "fp4")
            w3_entry = (w3.to("cpu").pin_memory(), s3_p, "fp4")
            w2_entry = (w2.to("cpu").pin_memory(), s2_p, "fp4")
            self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=True)
            expert_count += 1

        if weight_count > 0 or expert_count > 0:
            self._log(f"MTP: {weight_count} layer weights + {expert_count} experts preloaded")

    def _get_mtp_weight(self, key: str):
        if key in self._mtp_weights:
            return self._mtp_weights[key]
        loader = getattr(self, 'loader', None)
        if loader is None:
            return None
        w = loader.get_weight(key, device=str(self.device))
        if w is not None:
            w = w.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w
            self._mtp_weights[key] = w
        return w

    def _mtp_load_expert(self, eid: int):
        cache_key = f"mtp_0_{eid}"
        cached = self.expert_cache.get(cache_key)
        if cached is not None:
            return cached
        loader = getattr(self, 'loader', None)
        if loader is None:
            return None
        prefix = f"mtp.0.ffn.experts.{eid}"
        keys = [f"{prefix}.w1.weight", f"{prefix}.w1.scale",
                f"{prefix}.w3.weight", f"{prefix}.w3.scale",
                f"{prefix}.w2.weight", f"{prefix}.w2.scale"]
        tensors = loader.get_weights(*keys)
        w1 = tensors.get(keys[0]); s1 = tensors.get(keys[1])
        w3 = tensors.get(keys[2]); s3 = tensors.get(keys[3])
        w2 = tensors.get(keys[4]); s2 = tensors.get(keys[5])
        if w1 is None:
            return None
        s1_p = s1.to(torch.float32).to("cpu").pin_memory() if s1 is not None else None
        s3_p = s3.to(torch.float32).to("cpu").pin_memory() if s3 is not None else None
        s2_p = s2.to(torch.float32).to("cpu").pin_memory() if s2 is not None else None
        w1_entry = (w1.to("cpu").pin_memory(), s1_p, "fp4")
        w3_entry = (w3.to("cpu").pin_memory(), s3_p, "fp4")
        w2_entry = (w2.to("cpu").pin_memory(), s2_p, "fp4")
        self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=True)
        return self.expert_cache.get(cache_key)

    def _mtp_attn_1tok(self, h, w, start_pos: int = 0):
        B, T, D = h.shape
        wq_a = w["mtp.0.attn.wq_a.weight"]; wq_b = w["mtp.0.attn.wq_b.weight"]
        wkv = w["mtp.0.attn.wkv.weight"]
        wo_a = w["mtp.0.attn.wo_a.weight"]; wo_b = w["mtp.0.attn.wo_b.weight"]
        q_norm = w.get("mtp.0.attn.q_norm.weight")
        kv_norm = w.get("mtp.0.attn.kv_norm.weight")

        q_latent = h.to(wq_a.dtype) @ wq_a.t()
        if q_norm is not None:
            q_latent = rms_norm(q_latent, q_norm.to(torch.bfloat16), self.config.rms_norm_eps)
        q = q_latent @ wq_b.t()
        q = q.view(B, T, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)

        kv_latent = h.to(wkv.dtype) @ wkv.t()
        if kv_norm is not None:
            kv_latent = rms_norm(kv_latent, kv_norm.to(torch.bfloat16), self.config.rms_norm_eps)

        rope_dim = self.config.qk_rope_head_dim
        freqs_cis = precompute_freqs_cis(rope_dim, T, theta=self.config.rope_theta, start_pos=start_pos).to(h.device)
        q = apply_rotary_emb(q, freqs_cis, rd=rope_dim)
        kv_latent = apply_rotary_emb(kv_latent, freqs_cis, rd=rope_dim)

        if self._use_gqa_fusion:
            from home_seek.gqa_attention import gqa_fused_attn

            k = kv_latent.unsqueeze(2).transpose(1, 2)
            k_cache = getattr(self, '_mtp_kv_cache', None)
            if k_cache is not None:
                k_cache.append(k)
                all_kv = torch.cat(k_cache, dim=-2)
            else:
                all_kv = k
            out = gqa_fused_attn(q.float(), all_kv).to(torch.bfloat16)
        else:
            n_kv = self.config.num_key_value_heads
            n_groups = self.config.num_attention_heads // n_kv
            k = kv_latent.unsqueeze(2).transpose(1, 2)
            k_exp = k.unsqueeze(1).expand(-1, n_groups, -1, -1, -1).reshape(B, -1, k.shape[-2], k.shape[-1])
            v_exp = k_exp

            k_cache = getattr(self, '_mtp_kv_cache', None)
            if k_cache is not None:
                k_cache.append((k_exp, v_exp))
                all_k = torch.cat([ck for ck, _ in k_cache], dim=-2)
                all_v = torch.cat([cv for _, cv in k_cache], dim=-2)
            else:
                all_k = k_exp
                all_v = v_exp

            scale_f = self.config.head_dim ** -0.5
            attn = torch.matmul(q.float() * scale_f, all_k.float().transpose(-2, -1))
            attn_p = torch.softmax(attn, dim=-1).to(all_v.dtype)
            out = torch.matmul(attn_p, all_v)

        out = apply_rotary_emb(out, freqs_cis, rd=rope_dim, inverse=True)
        out = out.transpose(1, 2).contiguous()

        if wo_a is not None and wo_b is not None:
            out_g = out.view(B, T, self.config.o_groups, -1)
            wo_a_g = wo_a.view(self.config.o_groups, self.config.o_lora_rank, -1)
            out_comb = torch.einsum('btgd,grd->btgr', out_g.to(wo_a_g.dtype), wo_a_g)
            out_comb = out_comb.reshape(B, T, -1)
            out = out_comb.to(wo_b.dtype) @ wo_b.t()
        else:
            out = out.view(B, T, -1)
        return out.to(h.dtype)

    def _mtp_forward_draft(self, h_4d, input_ids=None, start_pos: int = 0):
        """Full MTP transformer layer forward pass for 1 token.

        h_4d: [B=1, T=1, hc_mult=4, D=4096]
        start_pos: global position for RoPE computation
        Returns: [B, 1, hc_mult, D] hidden after MTP layer
        """
        w = self._mtp_weights
        B, T, hc, D = h_4d.shape

        lw_mtp = {
            "hc_attn_base": w.get("mtp.0.hc_attn_base"),
            "hc_attn_fn": w.get("mtp.0.hc_attn_fn"),
            "hc_attn_scale": w.get("mtp.0.hc_attn_scale"),
            "hc_ffn_base": w.get("mtp.0.hc_ffn_base"),
            "hc_ffn_fn": w.get("mtp.0.hc_ffn_fn"),
            "hc_ffn_scale": w.get("mtp.0.hc_ffn_scale"),
        }

        h_pre, post, comb = self._process_mhc_layer(h_4d, lw_mtp, "hc_attn")
        if w.get("mtp.0.attn_norm.weight") is not None:
            h_pre = rms_norm(h_pre, w["mtp.0.attn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)

        attn_out = self._mtp_attn_1tok(h_pre, w, start_pos)

        if post is not None and comb is not None:
            h = self._process_mhc_post(attn_out, h_4d, post, comb)
        else:
            h = h_4d + attn_out.unsqueeze(2).expand(-1, -1, hc, -1)

        residual = h
        h_pre, post, comb = self._process_mhc_layer(h, lw_mtp, "hc_ffn")
        if w.get("mtp.0.ffn_norm.weight") is not None:
            h_pre = rms_norm(h_pre, w["mtp.0.ffn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)

        gate_w = w.get("mtp.0.ffn.gate.weight")
        gate_bias = w.get("mtp.0.ffn.gate.bias")
        if gate_w is not None:
            def mtp_load(layer_idx, eid):
                raw = self._mtp_load_expert(eid)
                if raw is None:
                    return None
                w1_e, w3_e, w2_e = raw
                if w1_e is None:
                    return None
                w1_data, w1_s, w1_fmt = w1_e
                w3_data, w3_s, w3_fmt = w3_e
                w2_data, w2_s, w2_fmt = w2_e
                dev = self.device
                w1_b = (w1_data.to(dev, non_blocking=True) if w1_data.device.type != dev.type else w1_data)
                w3_b = (w3_data.to(dev, non_blocking=True) if w3_data.device.type != dev.type else w3_data)
                w2_b = (w2_data.to(dev, non_blocking=True) if w2_data.device.type != dev.type else w2_data)
                if w1_fmt == "fp4":
                    w1_s_gpu = (w1_s.to(dev, non_blocking=True) if w1_s is not None else None)
                    w3_s_gpu = (w3_s.to(dev, non_blocking=True) if w3_s is not None else None)
                    w2_s_gpu = (w2_s.to(dev, non_blocking=True) if w2_s is not None else None)
                    return (w1_b, w1_s_gpu, w3_b, w3_s_gpu, w2_b, w2_s_gpu)
                return (w1_b, w3_b, w2_b)

            h_2d = h_pre.reshape(B * T, D)
            topk_idx, topk_w = self._compute_routing_experts(h_2d, gate_w, gate_bias)
            ffn_result = self._fused_moe.forward(h_2d, topk_idx, topk_w, mtp_load, -1)
            ffn_out = ffn_result.reshape(B, T, D)
        else:
            ffn_out = torch.zeros_like(h_pre.reshape(B, T, D))

        if post is not None and comb is not None:
            h = self._process_mhc_post(ffn_out, residual, post, comb)
        else:
            h = residual + ffn_out.unsqueeze(2).expand(-1, -1, hc, -1)

        return h

    def _mtp_finalize(self, h_4d):
        """Apply sigmoid HC head + norm to get 2D hidden from MTP output.

        Matches reference ParallelHead.hc_head:
          pre = sigmoid(mixes * hc_scale + hc_base) + eps
          y = sum(pre * original_4D, dim=2)
        """
        hc_fn = self._mtp_weights.get("mtp.0.hc_head_fn")
        hc_base = self._mtp_weights.get("mtp.0.hc_head_base")
        hc_scale = self._mtp_weights.get("mtp.0.hc_head_scale")
        if all(x is not None for x in [hc_fn, hc_base, hc_scale]):
            B, T, hc, D = h_4d.shape
            x = h_4d.reshape(B, T, hc * D).float()
            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
            mixes = x @ hc_fn.t() * rsqrt
            pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.config.hc_eps
            h_3d = (pre.unsqueeze(-1) * h_4d.float()).sum(dim=2).to(torch.bfloat16)
        else:
            h_3d = h_4d.sum(dim=2)

        norm_w = self._mtp_weights.get("mtp.0.norm.weight")
        if norm_w is not None:
            h_3d = rms_norm(h_3d, norm_w.to(torch.bfloat16), self.config.rms_norm_eps)
        return h_3d

    def _conditional_preload(self):
        """Skipped: light preload not worth the startup time (136s for ~10% cache fill).
        On-demand loading during inference is fast enough with RAID 1.5 GB/s.
        """

    def _preload_all_experts(self):
        """Load expert pairs into ExpertWeightCache during init.

        Preloads hot + hash experts (pinned) first, then fills the remaining
        cache capacity with experts striped across all layers.
        Uses multiple threads to saturate the disk read bandwidth.
        """
        import concurrent.futures
        num_layers = self.config.num_hidden_layers
        num_experts = self.config.n_routed_experts
        cache_cap = self.expert_cache.max_experts

        pinned_count = len(self.expert_cache.pinned)
        fill_target = min(cache_cap - pinned_count, num_layers * num_experts)
        self._log(f"Preloading experts: {pinned_count} pinned, "
                  f"filling up to {fill_target} more (cache cap {cache_cap})...")

        t0 = time.time()
        count = [0]

        def load_one(args):
            li, ei = args
            if self._load_expert_weights(li, ei) is not None:
                count[0] += 1

        eids_per_layer = max(1, fill_target // num_layers)
        tasks = []
        for li in range(num_layers):
            for ei in range(min(eids_per_layer, num_experts)):
                tasks.append((li, ei))

        max_workers = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for _ in ex.map(load_one, tasks):
                if count[0] >= fill_target:
                    break

        elapsed = time.time() - t0
        rate = count[0] / elapsed if elapsed > 0 else 0
        self._log(f"Preload done: {count[0]} experts in {elapsed:.1f}s "
                  f"({rate:.0f} exp/s), cache: {len(self.expert_cache.cache)} entries")

    def _update_hot_usage(self, layer: int, eid: int):
        if layer not in self._hot_usage_counter:
            self._hot_usage_counter[layer] = {}
        self._hot_usage_counter[layer][eid] = self._hot_usage_counter[layer].get(eid, 0) + 1
        self._hot_usage_window.append((layer, eid))
        if len(self._hot_usage_window) > self._hot_usage_window_size:
            for _ in range(len(self._hot_usage_window) - self._hot_usage_window_size):
                old_layer, old_eid = self._hot_usage_window.pop(0)
                cnt = self._hot_usage_counter.get(old_layer, {}).get(old_eid, 0)
                if cnt > 1:
                    self._hot_usage_counter[old_layer][old_eid] = cnt - 1
                else:
                    self._hot_usage_counter[old_layer].pop(old_eid, None)

    def _refresh_gpu_hot_from_usage(self):
        if not self._hot_usage_counter:
            return
        for layer, eid_counts in self._hot_usage_counter.items():
            hot_layer = self._hot_expert_set_by_layer.get(layer, self._hot_expert_set)
            sorted_eids = sorted(eid_counts.items(), key=lambda x: -x[1])
            for eid, _ in sorted_eids[:8]:
                if eid not in hot_layer and (layer, eid) not in self._backend.gpu_hot_experts(layer):  # noqa: E501
                    continue

    def _async_prefetch_experts(self, next_layer: int, predicted_eids: list[int]):
        if not self._prefetch_enabled or not predicted_eids:
            return
        if self._prefetch_stream is None:
            return
        with torch.cuda.stream(self._prefetch_stream):
            for eid in predicted_eids:
                self._load_expert_fp4_raw(next_layer, eid)

    def _warmup(self) -> bool:
        if self._warmed_up:
            return True
        self._log("Warming up kernels...")

        self._warmed_up = True
        self._log("Warmup complete (tile_kernels removed)")
        return True

    @torch.no_grad()
    def _mtp_generate_draft(self, last_hidden, num_draft: int = 3, temperature: float = 0.6,
                            last_token_id: torch.Tensor | None = None):
        """Generate draft tokens using the full MTP module (attention + MoE FFN + MHC).

        Implements MTPBlock.forward from the reference:
          x = hnorm(x)                     # RMSNorm on 4D
          e = enorm(embed(input_ids))      # RMSNorm on 2D embedding
          x = e_proj(e).unsqueeze(2) + h_proj(x)   # Combine, preserve hc_mult
          x = Block.forward(x, ...)        # MHC_attn → attn → MHC_ffn → FFN
          logits = head(x, hc_head_fn, hc_head_scale, hc_head_base, norm)

        last_hidden: [B=1, 1, hc_mult, D] — main model's last hidden state.
        last_token_id: [B=1, 1] — the token ID at current position.
        Returns (draft_ids, n_draft) or (None, 0).
        """
        embed = self.embed
        lm_head = self.lm_head
        if embed is None or lm_head is None:
            return None, 0
        if not self._mtp_weights:
            return None, 0

        e_proj = self._mtp_weights.get("mtp.0.e_proj.weight")
        h_proj = self._mtp_weights.get("mtp.0.h_proj.weight")
        if e_proj is None or h_proj is None:
            return None, 0

        B, _, hc, D = last_hidden.shape

        def _bf16(t):
            return t.to(torch.bfloat16) if t is not None and t.dtype != torch.bfloat16 else t

        hnorm_w = _bf16(self._mtp_weights.get("mtp.0.hnorm.weight"))
        h_proj_bf16 = _bf16(h_proj)
        e_proj_bf16 = _bf16(e_proj)
        enorm_w = _bf16(self._mtp_weights.get("mtp.0.enorm.weight"))

        # h_main = hnorm(last_hidden) → h_proj  (preserves hc_mult)
        h_main_normed = rms_norm(last_hidden, hnorm_w, self.config.rms_norm_eps) if hnorm_w is not None else last_hidden
        h_main_proj_4d = F.linear(h_main_normed.to(h_proj_bf16.dtype), h_proj_bf16)

        draft_tokens: list[torch.Tensor] = []
        self._mtp_kv_cache = []
        try:
            for step in range(num_draft):
                if step == 0 and last_token_id is not None:
                    tok_ids = last_token_id
                elif step > 0:
                    tok_ids = draft_tokens[-1]
                else:
                    tok_ids = None

                if tok_ids is not None:
                    tok_emb = embed[tok_ids].to(torch.bfloat16)
                else:
                    tok_emb = torch.zeros(B, 1, D, device=last_hidden.device, dtype=torch.bfloat16)

                if enorm_w is not None:
                    tok_emb = rms_norm(tok_emb, enorm_w, self.config.rms_norm_eps)
                emb_proj_2d = tok_emb.to(e_proj_bf16.dtype) @ e_proj_bf16.t()
                emb_proj_4d = emb_proj_2d.unsqueeze(2)

                h_combined_4d = h_main_proj_4d + emb_proj_4d

                mtp_pos = getattr(self, '_global_pos', 0) + step + 1
                h_mtp_out = self._mtp_forward_draft(h_combined_4d, start_pos=mtp_pos)

                h_3d = self._mtp_finalize(h_mtp_out)

                logits = h_3d.to(lm_head.dtype) @ lm_head.t()
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1].float() / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1)
                else:
                    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
                draft_tokens.append(next_id)

            draft_ids = torch.cat(draft_tokens, dim=-1)
            return draft_ids, len(draft_tokens)
        finally:
            self._mtp_kv_cache = None

    @torch.no_grad()
    def _mtp_accept_drafts(self, input_ids, draft_ids, temperature=0.6):
        """Verify all draft tokens in ONE batched forward pass.

        Each draft token is processed sequentially one-by-one (not batched),
        because MHC kernels and compressor state only support T=1.

        Returns (n_accepted, bonus_logits) matching the main model's trajectory.
        """
        if draft_ids is None or draft_ids.shape[1] == 0:
            return 0, None

        T_draft = draft_ids.shape[1]
        n_accept = 0
        bonus_logits = None

        kv_snapshots = {}
        for layer_idx, state in self.layer_states.items():
            if state.kv_latent_cache is not None:
                kv_snapshots[layer_idx] = state.kv_latent_cache.clone()

        compressor_bak = {}
        for layer_idx, comp in self._compressors.items():
            if comp is not None:
                compressor_bak[layer_idx] = (comp.accumulated,
                    comp.kv_state.clone() if comp.kv_state is not None else None,
                    comp.score_state.clone() if comp.score_state is not None else None)

        pos_bak = self._global_pos

        try:
            for i in range(T_draft):
                self._global_pos += 1
                draft_token = draft_ids[:, i:i + 1]

                h = self.embed[draft_token].to(torch.bfloat16)
                h = h.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

                for layer_idx in range(self.config.num_hidden_layers):
                    lw = self._get_layer_weights(layer_idx)
                    h, _ = self._forward_layer(h, lw, layer_idx, draft_token)
                h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
                lm_head = self._get_per_device('lm_head', h_3d.device)
                norm_w = self._get_per_device('norm_weight', h_3d.device)
                if norm_w is not None:
                    h_3d = rms_norm(h_3d, norm_w, self.config.rms_norm_eps)
                logits = torch.matmul(h_3d.to(lm_head.dtype), lm_head.t())

                if i < T_draft - 1:
                    expected_next = draft_ids[:, i + 1]
                    pred_next = logits[:, -1].argmax(dim=-1)
                    if pred_next.item() != expected_next.item():
                        bonus_logits = logits[:, -1, :]
                        break
                    n_accept += 1
                else:
                    n_accept += 1

            if n_accept < T_draft:
                for layer_idx in kv_snapshots:
                    state = self.layer_states.get(layer_idx)
                    if state is not None and state.kv_latent_cache is not None:
                        orig_len = kv_snapshots[layer_idx].shape[1]
                        keep_len = orig_len + n_accept
                        if state.kv_latent_cache.dim() == 4:
                            state.kv_latent_cache = state.kv_latent_cache[:, :keep_len, :, :].contiguous()
                        else:
                            state.kv_latent_cache = state.kv_latent_cache[:, :keep_len, :].contiguous()
                for layer_idx, (acc, kv_s, sc_s) in compressor_bak.items():
                    comp = self._compressors.get(layer_idx)
                    if comp is not None:
                        comp.accumulated = acc
                        comp.kv_state = kv_s
                        comp.score_state = sc_s
        finally:
            self._global_pos = pos_bak + n_accept
            if n_accept == 0:
                for layer_idx, saved_kv in kv_snapshots.items():
                    state = self.layer_states.get(layer_idx)
                    if state is not None:
                        state.kv_latent_cache = saved_kv
                for layer_idx, (acc, kv_s, sc_s) in compressor_bak.items():
                    comp = self._compressors.get(layer_idx)
                    if comp is not None:
                        comp.accumulated = acc
                        comp.kv_state = kv_s
                        comp.score_state = sc_s

        return n_accept, bonus_logits

    @torch.no_grad()
    def _mtp_verify_batched(self, draft_ids, temperature=0.6, main_pred_id=None):
        """Verify all draft tokens in a single fused forward pass.

        Fuses old Step 1 (T=1, main_pred_id) + Step 2 (T=T_draft, draft_ids)
        into one forward (T=1+T_draft). Causal mask in _forward_attn prevents
        future token leakage within the batch.
        """
        if not isinstance(draft_ids, torch.Tensor) or draft_ids.shape[1] == 0:
            return 0, None

        T_draft = draft_ids.shape[1]

        kv_snapshots = {}
        compressed_data_bak = {}
        for layer_idx, state in self.layer_states.items():
            if state.kv_latent_cache is not None:
                kv_snapshots[layer_idx] = state.kv_latent_cache.clone()
                compressed_data_bak[layer_idx] = (
                    state.compressed_kv_data.clone() if state.compressed_kv_data is not None else None,
                    state.compressed_kv_idx.clone() if state.compressed_kv_idx is not None else None,
                )

        compressor_bak = {}
        compressors = getattr(self, '_compressors', {})
        for layer_idx, comp in compressors.items():
            if comp is not None:
                compressor_bak[layer_idx] = (comp.accumulated,
                    comp.kv_state.clone() if comp.kv_state is not None else None,
                    comp.score_state.clone() if comp.score_state is not None else None)

        pos_bak = self._global_pos

        n_accept = 0
        bonus_logits = None
        try:
            if main_pred_id is not None:
                combined_ids = torch.cat([main_pred_id, draft_ids], dim=-1)
                T_total = 1 + T_draft
            else:
                combined_ids = draft_ids
                T_total = T_draft

            self._global_pos = pos_bak + 1
            self._phase = "verify"
            h = self.embed[combined_ids].to(torch.bfloat16)
            h = h.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
            for layer_idx in range(self.config.num_hidden_layers):
                lw = self._get_layer_weights(layer_idx)
                h, _ = self._forward_layer(h, lw, layer_idx, combined_ids)
            h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
            if self.norm_weight is not None:
                h_3d = rms_norm(h_3d, self.norm_weight, self.config.rms_norm_eps)
            logits = torch.matmul(h_3d.to(self.lm_head.dtype), self.lm_head.t())

            # Comparison: logits[i] predicts the token after combined_ids[i].
            # Accept combined_ids[i+1] if argmax matches.
            for i in range(T_total):
                pred = logits[:, i].argmax(dim=-1)
                if i == T_total - 1:
                    bonus_logits = logits[:, i, :]
                    break
                expected = combined_ids[:, i + 1]
                if pred.item() == expected.item():
                    n_accept += 1
                else:
                    bonus_logits = logits[:, i, :]
                    break

            # Trim: keep orig + n_accept entries (n_accept = main_pred + matched drafts)
            if n_accept < T_total:
                for layer_idx in kv_snapshots:
                    state = self.layer_states.get(layer_idx)
                    if state is not None and state.kv_latent_cache is not None:
                        orig_len = kv_snapshots[layer_idx].shape[1]
                        keep_len = orig_len + n_accept
                        if state.kv_latent_cache.dim() == 4:
                            state.kv_latent_cache = state.kv_latent_cache[:, :keep_len, :, :].contiguous()
                        else:
                            state.kv_latent_cache = state.kv_latent_cache[:, :keep_len, :].contiguous()
                for layer_idx, (data, idx) in compressed_data_bak.items():
                    state = self.layer_states.get(layer_idx)
                    if state is not None:
                        state.compressed_kv_data = data
                        state.compressed_kv_idx = idx
                for layer_idx, (acc, kv_s, sc_s) in compressor_bak.items():
                    comp = self._compressors.get(layer_idx)
                    if comp is not None:
                        comp.accumulated = acc
                        comp.kv_state = kv_s
                        comp.score_state = sc_s
        finally:
            self._global_pos = pos_bak + n_accept
            if n_accept == 0:
                for layer_idx, saved_kv in kv_snapshots.items():
                    state = self.layer_states.get(layer_idx)
                    if state is not None:
                        state.kv_latent_cache = saved_kv
                for layer_idx, (data, idx) in compressed_data_bak.items():
                    state = self.layer_states.get(layer_idx)
                    if state is not None:
                        state.compressed_kv_data = data
                        state.compressed_kv_idx = idx
                for layer_idx, (acc, kv_s, sc_s) in compressor_bak.items():
                    comp = self._compressors.get(layer_idx)
                    if comp is not None:
                        comp.accumulated = acc
                        comp.kv_state = kv_s
                        comp.score_state = sc_s

        return n_accept, bonus_logits

    def save_session(self, session_id: str, session_dir: str = "sessions"):
        os.makedirs(session_dir, exist_ok=True)
        sdir = os.path.join(session_dir, session_id)
        os.makedirs(sdir, exist_ok=True)

        state = {"global_pos": self._global_pos}
        torch.save(state, os.path.join(sdir, "state.pt"))

        for layer_idx, ls in self.layer_states.items():
            ldir = os.path.join(sdir, f"layer_{layer_idx}")
            os.makedirs(ldir, exist_ok=True)
            if ls.kv_latent_cache is not None:
                torch.save(ls.kv_latent_cache.cpu(), os.path.join(ldir, "kv_latent.pt"))
            if ls.compressed_kv_data is not None:
                torch.save(ls.compressed_kv_data.cpu(), os.path.join(ldir, "compressed_data.pt"))
            if ls.compressed_kv_idx is not None:
                torch.save(ls.compressed_kv_idx.cpu(), os.path.join(ldir, "compressed_idx.pt"))
            cstate = {"compressed_count": ls.compressed_count,
                      "archived_len": ls.archived_len}
            torch.save(cstate, os.path.join(ldir, "counts.pt"))
            if ls.archived_kv is not None:
                torch.save(ls.archived_kv.cpu(), os.path.join(ldir, "archived_kv.pt"))

        for layer_idx, comp in self._compressors.items():
            if comp is None:
                continue
            cdir = os.path.join(sdir, f"compressor_{layer_idx}")
            os.makedirs(cdir, exist_ok=True)
            torch.save({"accumulated": comp.accumulated}, os.path.join(cdir, "state.pt"))
            if comp.kv_state is not None:
                torch.save(comp.kv_state.cpu(), os.path.join(cdir, "kv_state.pt"))
            if comp.score_state is not None:
                torch.save(comp.score_state.cpu(), os.path.join(cdir, "score_state.pt"))

        for layer_idx, hybrid in self._hybrid_kv.items():
            hdir = os.path.join(sdir, f"hybrid_{layer_idx}")
            os.makedirs(hdir, exist_ok=True)
            torch.save({"write_pos": hybrid.swa._write_pos,
                        "num_entries": hybrid.swa._num_entries},
                       os.path.join(hdir, "swa_state.pt"))
            torch.save(hybrid.swa.k_cache.cpu(), os.path.join(hdir, "k_cache.pt"))
            torch.save(hybrid.swa.v_cache.cpu(), os.path.join(hdir, "v_cache.pt"))

    def load_session(self, session_id: str, session_dir: str = "sessions"):
        sdir = os.path.join(session_dir, session_id)
        state = torch.load(os.path.join(sdir, "state.pt"), map_location="cpu", weights_only=True)
        self._global_pos = state["global_pos"]

        for layer_idx in range(self.config.num_hidden_layers):
            ldir = os.path.join(sdir, f"layer_{layer_idx}")
            if not os.path.isdir(ldir):
                continue
            ls = LayerState(device=str(self.device))
            kv_path = os.path.join(ldir, "kv_latent.pt")
            if os.path.exists(kv_path):
                ls.kv_latent_cache = torch.load(kv_path, map_location=self.device, weights_only=True)
            cd_path = os.path.join(ldir, "compressed_data.pt")
            if os.path.exists(cd_path):
                ls.compressed_kv_data = torch.load(cd_path, map_location=self.device, weights_only=True)
            ci_path = os.path.join(ldir, "compressed_idx.pt")
            if os.path.exists(ci_path):
                ls.compressed_kv_idx = torch.load(ci_path, map_location=self.device, weights_only=True)
            c_path = os.path.join(ldir, "counts.pt")
            if os.path.exists(c_path):
                cstate = torch.load(c_path, map_location="cpu", weights_only=True)
                ls.compressed_count = cstate.get("compressed_count", 0)
                ls.archived_len = cstate.get("archived_len", 0)
            av_path = os.path.join(ldir, "archived_kv.pt")
            if os.path.exists(av_path):
                ls.archived_kv = torch.load(av_path, map_location="cpu", weights_only=True)
            self.layer_states[layer_idx] = ls

        for layer_idx in range(self.config.num_hidden_layers):
            cdir = os.path.join(sdir, f"compressor_{layer_idx}")
            if not os.path.isdir(cdir):
                continue
            comp = self._compressors.get(layer_idx)
            if comp is None:
                continue
            cstate_path = os.path.join(cdir, "state.pt")
            if os.path.exists(cstate_path):
                cs = torch.load(cstate_path, map_location="cpu", weights_only=True)
                comp.accumulated = cs.get("accumulated", 0)
            kv_path = os.path.join(cdir, "kv_state.pt")
            if os.path.exists(kv_path):
                comp.kv_state = torch.load(kv_path, map_location=self.device, weights_only=True)
            sc_path = os.path.join(cdir, "score_state.pt")
            if os.path.exists(sc_path):
                comp.score_state = torch.load(sc_path, map_location=self.device, weights_only=True)

        for layer_idx in range(self.config.num_hidden_layers):
            hdir = os.path.join(sdir, f"hybrid_{layer_idx}")
            if not os.path.isdir(hdir):
                continue
            hybrid = self._hybrid_kv.get(layer_idx)
            if hybrid is None:
                continue
            swa_path = os.path.join(hdir, "swa_state.pt")
            if os.path.exists(swa_path):
                sws = torch.load(swa_path, map_location="cpu", weights_only=True)
                hybrid.swa._write_pos = sws.get("write_pos", 0)
                hybrid.swa._num_entries = sws.get("num_entries", 0)
            kc_path = os.path.join(hdir, "k_cache.pt")
            if os.path.exists(kc_path):
                hybrid.swa.k_cache = torch.load(kc_path, map_location=self.device, weights_only=True)
            vc_path = os.path.join(hdir, "v_cache.pt")
            if os.path.exists(vc_path):
                hybrid.swa.v_cache = torch.load(vc_path, map_location=self.device, weights_only=True)

        self._log(f"Loaded session {session_id}: pos={self._global_pos}, layers={len(self.layer_states)}")

    def _forward_layer(self, h: torch.Tensor, lw: dict, layer_idx: int,
                       input_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, set]:
        with record_function(f"forward_layer_{layer_idx}"):
            # 多 GPU: 自动路由到对应 device + Triton 需要切换 CUDA context
            if self._is_multigpu:
                target = self._resolve_device(layer_idx)
                prev_dev = h.device
                if prev_dev.type == 'cuda' and prev_dev.index != torch.device(target).index:
                    h = self._backend.transfer_hidden(h, target)
                torch.cuda.set_device(torch.device(target).index)
            residual_attn = h
            with record_function("mhc_attn"):
                h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_attn")
            if lw.get("attn_norm.weight") is not None:
                h_pre = rms_norm(h_pre, lw["attn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)
            with record_function("forward_attn"):
                attn_out = self._forward_attn(h_pre, lw, layer_idx)
            if post is not None and comb is not None:
                h = self._process_mhc_post(attn_out, residual_attn, post, comb)
            else:
                h = h + attn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

            residual = h
            with record_function("mhc_ffn"):
                h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_ffn")
            if lw.get("ffn_norm.weight") is not None:
                h_pre = rms_norm(h_pre, lw["ffn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)
            with record_function("forward_ffn"):
                ffn_out, used_experts = self._forward_ffn(h_pre, lw, layer_idx, input_ids)
            if post is not None and comb is not None:
                h = self._process_mhc_post(ffn_out, residual, post, comb)
            else:
                h = h + ffn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
            # Per-request cache: keep all layers cached during a generate() call.
            # Non-expert weights (~4.7GB GPU) are pinned across decode steps and
            # only cleaned at generate() entry.  Removing this was a 30× regression
            # (each decode step re-reads all 43 layers from mmap).
        return h, used_experts

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.6, stream_callback=None,
                 session_id=None, resume=False):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape

        skip_prefill = resume and session_id is not None

        if not skip_prefill:
            self.layer_states = {}
            self._deq_cache.clear()
            self.expert_cache.clear()
            for cache in self._expert_caches.values():
                cache.clear()
            self._layer_weight_cache.clear()
            clear_deq_cache()
            # Clear GPU caches to prevent VRAM accumulation across requests
            self._gpu_hot_experts.clear()
            self._gpu_bf16_cache.clear()
            self._gpu_bf16_deq_cache.clear()
            for state in self._backend.all_device_states():
                state.gpu_hot_experts.clear()
                state.gpu_bf16_cache.clear()
            # Clear persistent per-layer state that accumulates VRAM between generate() calls
            self._compressors.clear()
            self._indexers.clear()
            self._hybrid_kv.clear()
            torch.cuda.empty_cache()
            for compressor in self._compressors.values():
                compressor.reset()
            for indexer in self._indexers.values():
                indexer.reset()
            cache_total = len(self.expert_cache) + sum(len(c) for c in self._expert_caches.values())
            cache_max = self.expert_cache.max_experts + sum(c.max_experts for c in self._expert_caches.values())
            self._log(f"Generate: {T} prompt tokens, max_new={max_new_tokens}, "
                       f"cache={cache_total}/{cache_max}, "
                       f"gpu_hot={sum(len(s.gpu_hot_experts) for s in self._backend.all_device_states())}, "
                       f"gpu_bf16={sum(len(s.gpu_bf16_cache) for s in self._backend.all_device_states())}")
            for hybrid in self._hybrid_kv.values():
                hybrid.reset()
            self._global_pos = 0
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        for dev_name in self._backend.devices:
            torch.cuda.reset_peak_memory_stats(dev_name)
        start = time.time()

        if not skip_prefill:
            h = self._get_per_device('embed')[input_ids].to(torch.bfloat16)
            h = h.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

            self._phase = "prefill"
            for layer_idx in range(self.config.num_hidden_layers):
                lw = self._get_layer_weights(layer_idx)
                h, _ = self._forward_layer(h, lw, layer_idx, input_ids)

                if self._prefetch_enabled:
                    predicted = getattr(self, '_last_routed_eids', [])[:6]
                    for eid in predicted:
                        self._update_hot_usage(layer_idx, eid)

                if (layer_idx + 1) % 5 == 0:
                    mem = torch.cuda.memory_allocated(self._backend.devices[0]) / (1024**3)
                    if mem > self.hw_config.kv_offload_threshold_gb:
                        for li in range(max(0, layer_idx - 5), layer_idx + 1):
                            if li in self.layer_states and self.layer_states[li].kv_latent_cache is not None:
                                if self.layer_states[li].kv_latent_cache.shape[1] > self.config.sliding_window * 2:
                                    with torch.cuda.stream(self._kv_offload_stream):
                                        st = self.layer_states[li]
                                        kv = st.kv_latent_cache
                                        st.archived_kv = kv[:, :-self.config.sliding_window].contiguous()
                                        st.archived_kv = st.archived_kv.to("cpu", non_blocking=True)
                                        st.kv_latent_cache = kv[:, -self.config.sliding_window:].contiguous()
                                if self.layer_states[li].compressed_kv_data is not None:
                                    self.layer_states[li].compressed_kv_data = None
                                    self.layer_states[li].compressed_kv_idx = None
                        torch.cuda.empty_cache()
                        self._log(f"  Layer {layer_idx}: freed caches, mem={mem:.1f}GB")

            h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
            lm_head = self._get_per_device('lm_head', h_3d.device)
            norm_w = self._get_per_device('norm_weight', h_3d.device)
            if norm_w is not None:
                h_3d = rms_norm(h_3d, norm_w, self.config.rms_norm_eps)
            logits = torch.matmul(h_3d[:, -1:].to(lm_head.dtype), lm_head.t())

            self._global_pos = T  # prefill done, advance position

            if temperature > 0:
                probs = F.softmax(logits[:, -1].float() / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            next_id = next_id.to(self.embed.device)

            generated = [next_id]
            if stream_callback is not None:
                stream_callback(next_id.item())
            prefill_end = time.time()
        else:
            self._phase = "decode"
            next_id = input_ids[:, -1:]
            generated = []
            prefill_end = start
            T = self._global_pos
            self._log(f"Session resume: pos={T}, max_new={max_new_tokens}")

        mtp_num_draft = self._mtp_num_draft if self._mtp_loaded else 0
        _mtp_total_accepted = 0
        _mtp_total_drafts = 0
        _mtp_total_steps = 0

        self._phase = "decode"
        step = 0
        while step < max_new_tokens - 1:
            self._global_pos = T + step
            if next_id.device != self.embed.device:
                next_id = next_id.to(self.embed.device, non_blocking=True)
            with record_function("decode_embed"):
                h = self.embed[next_id].to(torch.bfloat16)
                h = h.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
            for layer_idx in range(self.config.num_hidden_layers):
                lw = self._get_layer_weights(layer_idx)
                h, _ = self._forward_layer(h, lw, layer_idx, next_id)

                if self._prefetch_enabled:
                    predicted = getattr(self, '_last_routed_eids', [])[:6]
                    for eid in predicted:
                        self._update_hot_usage(layer_idx, eid)

                if layer_idx == self.config.num_hidden_layers // 2:
                    mem = torch.cuda.memory_allocated(self._backend.devices[0]) / (1024**3)
                    if mem > self.hw_config.kv_offload_threshold_gb:
                        for li in range(layer_idx):
                            if li in self.layer_states and self.layer_states[li].kv_latent_cache is not None:
                                if self.layer_states[li].kv_latent_cache.shape[1] > self.config.sliding_window * 2:
                                    with torch.cuda.stream(self._kv_offload_stream):
                                        st = self.layer_states[li]
                                        kv = st.kv_latent_cache
                                        st.archived_kv = kv[:, :-self.config.sliding_window].contiguous()
                                        st.archived_kv = st.archived_kv.to("cpu", non_blocking=True)
                                        st.kv_latent_cache = kv[:, -self.config.sliding_window:].contiguous()

            last_h_for_mtp = h

            with record_function("decode_lm_head"):
                h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
                lm_head = self._get_per_device('lm_head', h_3d.device)
                norm_w = self._get_per_device('norm_weight', h_3d.device)
                if norm_w is not None:
                    h_3d = rms_norm(h_3d, norm_w, self.config.rms_norm_eps)
                logits = torch.matmul(h_3d.to(lm_head.dtype), lm_head.t())

            with record_function("decode_sample"):
                if temperature > 0:
                    probs = F.softmax(logits[:, -1].float() / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1)
                else:
                    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
                next_id = next_id.to(self.embed.device)
            generated.append(next_id)
            if stream_callback is not None:
                stream_callback(next_id.item())
            step += 1
            if next_id.item() in self._stop_token_ids:
                break

            if mtp_num_draft > 0 and step < max_new_tokens - 1:
                all_inputs = torch.cat(
                    [input_ids] + generated, dim=-1)
                last_token = all_inputs[:, -1:]
                mtp_temp = temperature if temperature > 0 else 0.0
                draft_ids, n_draft = self._mtp_generate_draft(
                    last_h_for_mtp, mtp_num_draft, mtp_temp, last_token_id=last_token)
                if draft_ids is not None and n_draft > 0:
                    if self._mtp_eager:
                        for i in range(n_draft):
                            generated.append(draft_ids[:, i:i+1])
                            if stream_callback is not None:
                                stream_callback(draft_ids[:, i:i+1].item())
                            step += 1
                            if draft_ids[:, i:i+1].item() in self._stop_token_ids:
                                break
                            if step >= max_new_tokens - 1:
                                break
                        else:
                            if step < max_new_tokens - 1:
                                next_id = draft_ids[:, -1:]
                            continue
                        break
                    n_acc, bonus_logits = self._mtp_verify_batched(
                        draft_ids, mtp_temp, main_pred_id=next_id)
                    _mtp_total_drafts += n_draft
                    _mtp_total_accepted += n_acc
                    _mtp_total_steps += 1
                    self._log(f"  MTP: n_draft={n_draft}, accepted={n_acc}, "
                              f"bonus={'yes' if bonus_logits is not None else 'no'}")
                    if n_acc > 0:
                        accepted_ids = draft_ids[:, :n_acc]
                        eos_hit = False
                        for i in range(n_acc):
                            generated.append(accepted_ids[:, i:i+1])
                            if stream_callback is not None:
                                stream_callback(accepted_ids[:, i:i+1].item())
                            step += 1
                            if accepted_ids[:, i:i+1].item() in self._stop_token_ids:
                                eos_hit = True
                                break
                            if step >= max_new_tokens - 1:
                                break
                        if eos_hit:
                            break
                        if step < max_new_tokens - 1:
                            if bonus_logits is not None and bonus_logits.dim() >= 1:
                                if temperature > 0:
                                    new_probs = F.softmax(bonus_logits.float() / temperature, dim=-1)
                                    next_id = torch.multinomial(new_probs, 1)
                                else:
                                    next_id = bonus_logits.argmax(dim=-1, keepdim=True)
                                generated.append(next_id)
                                if stream_callback is not None:
                                    stream_callback(next_id.item())
                                step += 1
                                if next_id.item() in self._stop_token_ids:
                                    break
                            elif n_acc == n_draft:
                                next_id = accepted_ids[:, -1:]
                        continue

        total_time = time.time() - start
        prefill_time = prefill_end - start
        decode_time = total_time - prefill_time
        if generated and generated[-1][0].item() in self._stop_token_ids:
            generated = generated[:-1]
        all_tokens = torch.cat([input_ids] + [g.to(input_ids.device) for g in generated], dim=-1)
        result = {
            "tokens": all_tokens,
            "total_time_s": total_time,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "new_tokens_per_second": len(generated) / total_time,
            "prefill_tokens_per_second": T / prefill_time if prefill_time > 0 else 0,
            "decode_tokens_per_second": len(generated) / decode_time if decode_time > 0 else 0,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "num_prompt_tokens": T,
            "num_generated_tokens": len(generated),
        }
        self._phase = "idle"
        from home_seek.fused_moe import deq_cache_stats
        hits, misses = deq_cache_stats()
        if hits + misses > 0:
            self._log(f"Deq cache: hits={hits} misses={misses} "
                      f"hit_rate={hits/(hits+misses)*100:.1f}%")
        if session_id is not None:
            self.save_session(session_id)
        self.loader.close()
        self._log(f"Done: {result['num_generated_tokens']} tokens in {total_time:.1f}s, "
                   f"peak mem: {result['peak_memory_gb']:.1f}GB")

        if _mtp_total_steps > 0:
            rate = _mtp_total_accepted / max(_mtp_total_drafts, 1) * 100
            self._log(f"MTP total: {_mtp_total_steps} steps, {_mtp_total_drafts} drafts, "
                      f"{_mtp_total_accepted} accepted ({rate:.0f}%), "
                      f"avg {_mtp_total_accepted / _mtp_total_steps:.2f} accepted/step")
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", default="weights")
    parser.add_argument("--prompt", default="Hello, world")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hot-experts", default="hot_experts.json", help="Hot experts JSON path")
    parser.add_argument("--use-mtp", action="store_true", help="Enable MTP speculative decoding")
    parser.add_argument("--use-gqa-fusion", action="store_true",
                        help="Use GQA fused attention kernel (experimental)")
    parser.add_argument("--mtp-eager", action="store_true",
                        help="Eager MTP: accept all drafts without verification (fast, risky)")
    parser.add_argument("--reprobe", action="store_true",
                        help="Force re-probe hardware profile, overwrite hw_profile.json")
    args = parser.parse_args()
    engine = HomeSeekInferenceEngine(args.weight_dir, verbose=args.verbose,
                                     hot_experts_path=args.hot_experts, reprobe=args.reprobe,
                                     use_gqa_fusion=args.use_gqa_fusion)
    if args.use_mtp:
        engine._mtp_loaded = True
    if args.mtp_eager:
        engine._mtp_loaded = True
        engine._mtp_eager = True

    tokenizer = None
    tokenizer_path = os.path.join(args.weight_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        try:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        except Exception:
            pass
    if tokenizer is not None:
        prompt_text = encode_messages([{"role": "user", "content": args.prompt}], thinking_mode="chat")
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(engine.device)
    else:
        input_ids = torch.randint(0, 100, (1, 8), device=engine.device)

    result = engine.generate(input_ids, max_new_tokens=args.max_tokens)
    if tokenizer is not None:
        out = tokenizer.decode(result["tokens"][0], skip_special_tokens=True)
        print(f"Output: {out}")
    print(f"Performance: {result['new_tokens_per_second']:.2f} t/s, "
          f"peak mem: {result['peak_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
