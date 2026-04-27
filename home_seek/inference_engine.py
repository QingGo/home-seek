import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from collections import defaultdict, OrderedDict

import math
from home_seek.utils import rms_norm
from home_seek.model_config import DeepSeekV4FlashConfig
from home_seek.mhc import mhc_split_sinkhorn
from home_seek.prefetch_worker import AsyncPrefetchWorker
from home_seek.fused_moe import FusedMoEFFN, SharedExpertFFN, clear_deq_cache, triton_dequantize_fp4_all
from home_seek.gpu_expert_store import AllExpertFP4Store
from home_seek.compressor import Compressor as NewCompressor
from home_seek.lightning_indexer import LightningIndexer
from home_seek.hybrid_kv_cache import HybridKVCache
from tile_reference import unpack_from_e2m1fn_x2
import tile_kernels

_current_dir = os.path.dirname(os.path.abspath(__file__))
_encoding_dir = os.path.join(_current_dir, '../weights/encoding')
sys.path.insert(0, os.path.abspath(_encoding_dir))
from encoding_dsv4 import encode_messages


def precompute_freqs_cis(dim: int, seqlen: int, theta: float = 10000.0,
                         original_seq_len: int = 0, factor: float = 1.0,
                         beta_fast: int = 32, beta_slow: int = 1) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0 and factor > 1.0:
        low = math.floor(dim * math.log(original_seq_len / (beta_fast * 2 * math.pi)) / (2 * math.log(theta)))
        high = math.ceil(dim * math.log(original_seq_len / (beta_slow * 2 * math.pi)) / (2 * math.log(theta)))
        low = max(low, 0)
        high = min(high, dim // 2 - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low + 1e-3)).clamp(0, 1)
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1.0 - smooth) + freqs * smooth
    t = torch.arange(seqlen, dtype=torch.float32)
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


def _ue8m0_to_f32(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype == torch.float32:
        return sf
    if sf.element_size() != 1:
        return sf.to(torch.float32)
    sf_u8 = sf.view(torch.uint8)
    sf_i32 = sf_u8.to(torch.int32)
    return (sf_i32 << 23).view(torch.float32)


def load_fp8_weight(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if data.dtype == torch.bfloat16 or data.dtype == torch.float32:
        return data.to(torch.bfloat16)
    if data.dtype == torch.float8_e4m3fn:
        sf = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32) if scale.element_size() == 1 else scale
        return tile_kernels.quant.cast_back((data, sf), 'bf16', (128, 128))
    return data.to(torch.bfloat16)


def load_fp4_weight(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if data.dtype == torch.bfloat16:
        return data
    deq = unpack_from_e2m1fn_x2(data)
    if scale.numel() == 1:
        deq = deq * scale.to(torch.float32).item()
    else:
        block_size = 32
        sf = scale.repeat_interleave(block_size, dim=1) if scale.dim() == 2 else scale
        deq = deq.to(torch.float32) * sf.to(torch.float32)
    return deq.to(torch.bfloat16)


class WeightLoader:
    def __init__(self, weight_dir: str, device: str = "cuda"):
        self.weight_dir = weight_dir
        self.device = torch.device(device)
        self.weight_map = {}
        self._mmap_cache = {}
        self._build_index()

    def _build_index(self):
        idx_path = os.path.join(self.weight_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                data = json.load(f)
            self.weight_map = data["weight_map"]
            self.total_size = data["metadata"]["total_size"]
            return

        safetensors_files = sorted([f for f in os.listdir(self.weight_dir) if f.endswith(".safetensors")])
        for fname in safetensors_files:
            fpath = os.path.join(self.weight_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with safe_open(fpath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        self.weight_map[key] = fname
            except Exception:
                pass

    def _open_mmap(self, fname: str):
        if fname not in self._mmap_cache:
            fpath = os.path.join(self.weight_dir, fname)
            self._mmap_cache[fname] = safe_open(fpath, framework="pt", device="cpu")
        return self._mmap_cache[fname]

    def get_weight(self, key: str, device: str = None):
        fname = self.weight_map.get(key)
        if fname is None:
            return None
        try:
            f = self._open_mmap(fname)
            tensor = f.get_tensor(key)
            if device is not None and str(device) != "cpu":
                return tensor.to(device, non_blocking=True)
            return tensor
        except Exception as e:
            self._log(f"Failed to get weight {key} from {fname}: {type(e).__name__}")
            return None

    def get_weights(self, *keys):
        results = {}
        file_keys = defaultdict(list)
        for k in keys:
            fname = self.weight_map.get(k)
            if fname:
                file_keys[fname].append(k)
        for fname, ks in file_keys.items():
            try:
                f = self._open_mmap(fname)
                for k in ks:
                    results[k] = f.get_tensor(k)
            except Exception as e:
                self._log(f"Failed batch load from {fname}: {type(e).__name__}")
        return results

    def close(self):
        self._mmap_cache.clear()

    def _safetensors_files(self):
        """Return list of all safetensor file paths in weight_dir."""
        weight_dir = self.weight_dir
        files = [os.path.join(weight_dir, f)
                 for f in sorted(os.listdir(weight_dir))
                 if f.endswith(".safetensors")]
        return [f for f in files if os.path.isfile(f)]

    def get_layer_weight(self, layer: int, weight_type: str):
        return self.get_weight(f"layers.{layer}.{weight_type}")

    def get_attn_weight(self, layer: int, name: str):
        return self.get_weight(f"layers.{layer}.attn.{name}")

    def get_ffn_weight(self, layer: int, name: str):
        return self.get_weight(f"layers.{layer}.ffn.{name}")


class LayerState:
    def __init__(self, device: str = "cuda", active_window: int = 32768):
        self.kv_latent_cache = None
        self.compressed_kv_data = None   # [N, head_dim] flat tensor (was CompressedKVCache.cache)
        self.compressed_kv_idx = None    # [N, idx_dim] flat tensor (was CompressedKVCache.indexer_keys)
        self.compressed_count = 0
        self.archived_kv = None
        self.archived_len = 0
        self.active_window = active_window
        self.device = torch.device(device)

    def append_kv(self, kv_latent: torch.Tensor):
        if self.kv_latent_cache is None:
            self.kv_latent_cache = kv_latent
            if self.archived_len > 0:
                self.kv_latent_cache = torch.cat([self._load_archived(), self.kv_latent_cache], dim=1)
                self.archived_kv = None
                self.archived_len = 0
            return
        self.kv_latent_cache = torch.cat([self.kv_latent_cache, kv_latent], dim=1)
        if self.kv_latent_cache.shape[1] > self.active_window + 1024:
            archive_len = self.kv_latent_cache.shape[1] - self.active_window
            archive_part = self.kv_latent_cache[:, :archive_len, :].contiguous()
            self.archived_len += archive_len
            if self.archived_kv is None:
                self.archived_kv = archive_part.to("cpu", non_blocking=True)
            else:
                self.archived_kv = torch.cat([self.archived_kv, archive_part.to("cpu", non_blocking=True)], dim=0)
            self.kv_latent_cache = self.kv_latent_cache[:, archive_len:, :].contiguous()

    def _load_archived(self) -> torch.Tensor:
        if self.archived_kv is None:
            return torch.zeros(1, 0, self.kv_latent_cache.shape[-1], device=self.device, dtype=torch.bfloat16)
        return self.archived_kv.to(self.device, non_blocking=True)

    def all_kv(self):
        if self.kv_latent_cache is None:
            return None
        if self.archived_kv is not None:
            arch = self._load_archived()
            result = torch.cat([arch, self.kv_latent_cache], dim=1)
            self.archived_kv = None
            self.archived_len = 0
            return result
        return self.kv_latent_cache


class ExpertWeightCache:
    def __init__(self, max_experts: int = 512, device: str = "cuda",
                 hot_deq_size: int = 0):
        self.max_experts = max_experts
        self.device = torch.device(device)
        self.cache = OrderedDict()
        self.pinned = set()

        self._hot_deq = OrderedDict()
        self._max_hot_deq = hot_deq_size

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def deq(self, key: str):
        if self._max_hot_deq > 0 and key in self._hot_deq:
            self._hot_deq.move_to_end(key)
            return self._hot_deq[key]

        entry = self.get(key)
        if entry is None:
            return None

        raw_w1, raw_w3, raw_w2 = entry
        w1 = self._dequantize_entry(raw_w1, self.device)
        w3 = self._dequantize_entry(raw_w3, self.device)
        w2 = self._dequantize_entry(raw_w2, self.device)

        if self._max_hot_deq > 0:
            self._hot_deq[key] = (w1, w3, w2)
            if len(self._hot_deq) > self._max_hot_deq:
                evicted = False
                for k in list(self._hot_deq.keys()):
                    if k not in self.pinned:
                        self._hot_deq.pop(k)
                        evicted = True
                        break
                if not evicted:
                    self._hot_deq.popitem(last=False)
        return (w1, w3, w2)

    def _dequantize_entry(self, entry, device=None):
        if entry is None:
            return None
        data, scale, fmt = entry
        dev = device if device is not None else data.device
        if fmt == "bf16":
            return data
        if fmt == "fp8":
            if str(data.device) != str(dev):
                data = data.to(dev, non_blocking=True)
                scale = scale.to(dev, non_blocking=True) if scale is not None else None
            return load_fp8_weight(data, scale)
        if fmt == "fp4_gpu" or fmt == "fp4":
            if str(data.device) != str(dev):
                data = data.to(dev, non_blocking=True)
                scale = scale.to(dev, non_blocking=True) if scale is not None else None
            return load_fp4_weight(data, scale)
        return data.to(torch.bfloat16)

    def put(self, key: str, w1_entry, w3_entry, w2_entry, pin: bool = False):
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        if not pin and len(self.cache) >= self.max_experts + len(self.pinned):
            for k, _ in list(self.cache.items()):
                if k not in self.pinned:
                    self.cache.pop(k)
                    if k in self._hot_deq:
                        del self._hot_deq[k]
                    break
        self.cache[key] = (w1_entry, w3_entry, w2_entry)
        if pin:
            self.pinned.add(key)

    def put_deq(self, key: str, w1_d, w3_d, w2_d, pin: bool = False):
        self.put(key, (w1_d, None, "bf16"), (w3_d, None, "bf16"), (w2_d, None, "bf16"), pin=pin)

    def clear(self):
        pinned_entries = {k: self.cache[k] for k in list(self.cache.keys()) if k in self.pinned}
        self.cache.clear()
        self.cache.update(pinned_entries)
        self._hot_deq.clear()
        self.pinned.clear()
        for k in pinned_entries:
            self.pinned.add(k)

    def trim(self, target_count: int = 64):
        excess = len(self.cache) - len(self.pinned) - target_count
        if excess <= 0:
            return
        for k in list(self.cache.keys()):
            if k in self.pinned:
                continue
            if excess <= 0:
                break
            self.cache.pop(k)
            self._hot_deq.pop(k, None)
            excess -= 1

    def __len__(self):
        return len(self.cache)


class HomeSeekInferenceEngine:
    def __init__(self, weight_dir: str = "weights", device: str = "cuda", verbose: bool = False,
                 hot_experts_path: str = "hot_experts.json", preload_all: bool = False,
                 reprobe: bool = False):
        config_path = os.path.join(weight_dir, "config.json")
        self.config = DeepSeekV4FlashConfig(config_path)
        self.weight_dir = weight_dir
        self.device = torch.device(device)
        self.verbose = verbose
        self.loader = WeightLoader(weight_dir, device)
        from home_seek.hw_profile import probe_hardware, load_profile
        self.hw_profile = probe_hardware(force=reprobe, weight_dir=weight_dir) if reprobe else load_profile()
        if self.hw_profile is None:
            self.hw_profile = probe_hardware(force=True, weight_dir=weight_dir)
        cache_size = 5120
        self.expert_cache = ExpertWeightCache(max_experts=cache_size, device=device, hot_deq_size=0)
        self.layer_states = {}
        self._deq_cache = OrderedDict()
        self._compressors = {}       # per-layer Compressor instances (lazy)
        self._indexers = {}          # per-layer LightningIndexer instances (lazy)
        self._hybrid_kv = {}         # per-layer HybridKVCache instances (lazy)
        self._global_pos = 0         # current global sequence position
        self._prefetch_worker = None
        self._load_global_weights()
        self._prefetch_enabled = False  # async prefetch adds GIL contention on shared disk
        from home_seek.expert_predictor import RecordingPredictor, HeuristicPredictor
        self.predictor = RecordingPredictor(
            HeuristicPredictor(self.config.num_hidden_layers, self.config.num_experts_per_tok))
        self._prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._cpu_fallback_enabled = True
        self._cpu_fallback_layers = set(range(min(3, self.config.num_hidden_layers))) | set(range(max(0, self.config.num_hidden_layers - 3), self.config.num_hidden_layers))
        self._hot_expert_ids = []
        self._hash_expert_ids = []
        self._hot_expert_set = set()
        self._hot_expert_set_by_layer = {}
        self._gpu_hot_experts = {}
        _vram_free = max(self.hw_profile.vram_free_gb, 8.0)
        _expert_bf16_gb = 48.0 / 1024
        self._max_hot_experts = max(16, min(int((_vram_free - 4) * 0.2 / _expert_bf16_gb), 64))
        self._gpu_bf16_cache = OrderedDict()
        _bf16_budget = max(1, _vram_free - 3)
        self._max_bf16_cache = max(16, min(int(_bf16_budget * 0.8 / _expert_bf16_gb), 100))
        self._preload_hot_experts(hot_experts_path)
        self._kv_offload_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._compress_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self._shared_expert_weights = {}
        self._shared_experts_loaded = False
        self._mtp_weights = {}
        self._mtp_loaded = False
        self._mtp_eager = False
        self._warmed_up = False
        self._layer_weight_cache = {}
        self._phase = "idle"

        self._gpu_expert_store = AllExpertFP4Store(
            weight_dir, self.config, str(self.device), max_experts=128)

        self._fused_moe = FusedMoEFFN(
            num_experts=self.config.n_routed_experts,
            intermediate_size=self.config.moe_intermediate_size,
            hidden_size=self.config.hidden_size,
            swiglu_limit=self.config.swiglu_limit,
            use_triton=True,
        )
        self._shared_ffn = SharedExpertFFN(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.shared_expert_intermediate_size,
            swiglu_limit=self.config.swiglu_limit,
        )

        self._load_shared_experts_gpu()
        self._load_mtp_weights()
        self._init_prefetch()
        if self._hot_expert_ids:
            self._preload_hot_experts_cpu_cache()
            self._preload_gpu_hot_experts()
        if preload_all:
            self._preload_all_experts()
        warmup_ok = self._warmup()
        if not warmup_ok:
            self._log("WARNING: Warmup had failures — first inference will be slow "
                      "due to JIT compilation on the critical path.")

    def _log(self, msg):
        if getattr(self, 'verbose', False):
            print(f"[inference] {msg}")

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

    def _preload_gpu_hot_experts(self):
        if not self._hot_expert_ids:
            return
        count = 0
        preload_layers = min(4, self.config.num_hidden_layers)
        for layer in range(preload_layers):
            layer_set = self._hot_expert_set_by_layer.get(layer, self._hot_expert_set)
            for eid in list(layer_set)[:16]:
                if (layer, eid) not in self._gpu_hot_experts:
                    self._load_expert_fp4_raw(layer, eid)
                    count += 1
                if len(self._gpu_hot_experts) >= self._max_hot_experts:
                    break
            if len(self._gpu_hot_experts) >= self._max_hot_experts:
                break
        self._log(f"Pre-loaded {count} hot expert BF16 weights on GPU")

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
                self._gpu_expert_store.cache_on_gpu(
                    layer_idx, eid, w1, s1, w3, s3, w2, s2)
                w1_entry = self._make_raw_entry(w1, s1)
                w3_entry = self._make_raw_entry(w3, s3)
                w2_entry = self._make_raw_entry(w2, s2)
                self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=True)
                count += 1
        if count > 0:
            self._log(f"Pre-loaded {count} hot expert raw entries into CPU cache "
                      f"({count * 48 // 1024} MB packed)")

    def _init_prefetch(self):
        if self._prefetch_enabled and self._prefetch_worker is None:
            self._prefetch_worker = AsyncPrefetchWorker(
                self.weight_dir, self.loader.weight_map, str(self.device),
                num_prefetch=32)

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

    def _get_layer_weights(self, layer_idx: int):
        if layer_idx in self._layer_weight_cache:
            return self._layer_weight_cache[layer_idx]
        lw = {}
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
        for full_key in keys_needed:
            short_key = full_key.replace(f"layers.{layer_idx}.", "")
            if full_key in present and present[full_key] is not None:
                lw[short_key] = present[full_key].to(self.device, non_blocking=True)
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

        # coff = 1 + overlap; overlap when compress_ratio == 4 (CSA)
        overlap = compress_ratio == 4
        coff = 2 if overlap else 1
        head_dim = self.config.head_dim  # 512

        # Handle weight dtypes
        if c_wkv.dtype not in (torch.bfloat16, torch.float32):
            c_wkv = c_wkv.to(torch.float32)
        if c_wgate.dtype not in (torch.bfloat16, torch.float32):
            c_wgate = c_wgate.to(torch.float32)
        if c_ape is not None and c_ape.dtype != torch.float32:
            c_ape = c_ape.to(torch.float32)

        compressor = NewCompressor(
            ratio=compress_ratio,
            head_dim=coff * head_dim // coff,  # head_dim (512)
            coff=coff,
            ape=c_ape,
            wkv=c_wkv.to(self.device) if c_wkv.device.type != self.device.type else c_wkv,
            wgate=c_wgate.to(self.device) if c_wgate.device.type != self.device.type else c_wgate,
            norm_w=c_norm,
            device=str(self.device),
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

        indexer = LightningIndexer(
            index_n_heads=self.config.index_n_heads,
            index_head_dim=self.config.index_head_dim,
            index_topk=self.config.index_topk,
            compress_ratio=compress_ratio,
            q_lora_rank=self.config.q_lora_rank,
            device=str(self.device),
        )
        indexer.set_weights(
            wq_b=idx_wq_b.to(self.device) if idx_wq_b.device.type != self.device.type else idx_wq_b,
            weights_proj=weights_proj.to(self.device) if weights_proj is not None and weights_proj.device.type != self.device.type else weights_proj,
            compressor_wkv=idx_c_wkv.to(self.device) if idx_c_wkv is not None and idx_c_wkv.device.type != self.device.type else idx_c_wkv,
            compressor_wgate=idx_c_wgate.to(self.device) if idx_c_wgate is not None and idx_c_wgate.device.type != self.device.type else idx_c_wgate,
            compressor_norm=idx_c_norm,
            compressor_ape=idx_c_ape.to(self.device) if idx_c_ape is not None and idx_c_ape.device.type != self.device.type else idx_c_ape,
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
        if scale is None:
            result = data.to(torch.bfloat16)
        elif data.dtype == torch.int8:
            result = load_fp4_weight(data, scale)
        else:
            result = load_fp8_weight(data, scale)
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
                device=str(self.device),
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
        try:
            hidden_contig = hidden_4d.contiguous()
            post_mix, comb_mix, layer_input = tile_kernels.modeling.mhc.ops.mhc_pre_big_fuse(
                hidden_contig, hc_fn.float(), hc_scale.to(torch.float32), hc_base.to(torch.float32),
                self.config.rms_norm_eps, self.config.hc_eps, self.config.hc_eps,
                2.0, self.config.hc_sinkhorn_iters)
            if apply_pre:
                hidden = layer_input.to(hidden_4d.dtype)
            else:
                hidden = hidden_4d.sum(dim=2)
            post = post_mix.squeeze(-1)
            comb = comb_mix
            return hidden, post, comb
        except Exception as e:
            self._log(f"mHC kernel fallback (not critical): {type(e).__name__}")
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
            return hidden, post, comb

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
        freqs_cis = precompute_freqs_cis(rope_dim, T, theta=rope_theta, original_seq_len=rope_original_seq_len, factor=rope_factor, beta_fast=rope_beta_fast, beta_slow=rope_beta_slow).to(q.device)
        q = apply_rotary_emb(q, freqs_cis, rd=self.config.qk_rope_head_dim)
        kv_latent = apply_rotary_emb(kv_latent, freqs_cis, rd=self.config.qk_rope_head_dim)

        state = self.layer_states.setdefault(layer_idx, LayerState(device=str(self.device)))
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
            v_all = torch.cat([v_sw, compressed_kv.to(v_sw.dtype)], dim=-2)
        else:
            k_all, v_all = k_sw, v_sw

        n_kv = self.config.num_key_value_heads
        n_groups = self.config.num_attention_heads // n_kv
        scale_f = self.config.head_dim ** -0.5

        k_expanded = k_all.unsqueeze(1).expand(-1, n_groups, -1, -1, -1).reshape(B, -1, k_all.shape[-2], k_all.shape[-1])
        v_expanded = v_all.unsqueeze(1).expand(-1, n_groups, -1, -1, -1).reshape(B, -1, v_all.shape[-2], v_all.shape[-1])

        attn = torch.matmul(q.float() * scale_f, k_expanded.float().transpose(-2, -1))

        if attn_sink is not None and attn_sink.numel() == self.config.num_attention_heads:
            # Virtual softmax entry: attn_sink absorbs probability mass without KV
            # Matches official demo's sparse_attn kernel behavior:
            #   sum_exp += exp(attn_sink - running_max)
            sink_val = attn_sink.view(1, -1, 1, 1).to(attn.dtype)
            attn_with_sink = torch.cat(
                [attn, sink_val.expand(-1, -1, T, -1)], dim=-1)
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
            data = data.to(dev)
            if scale is not None:
                scale = scale.to(torch.float32).to(dev) if scale.dtype != torch.float32 else scale.to(dev)
            return (data, scale, "fp4")
        data = data.to(self.device, non_blocking=True)
        return (data, scale, "bf16")

    def _load_expert_weights(self, layer_idx, eid):
        cache_key = f"{layer_idx}_{eid}"
        cached = self.expert_cache.get(cache_key)
        if cached is not None:
            return cached

        gpu_cached = self._gpu_expert_store.get_cache_key(layer_idx, eid)
        if gpu_cached is not None:
            gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale = gpu_cached
            w1_entry = self._make_raw_entry(gate_packed, gate_scale)
            w3_entry = self._make_raw_entry(up_packed, up_scale)
            w2_entry = self._make_raw_entry(down_packed, down_scale)
            layer_hot = self._hot_expert_set_by_layer.get(layer_idx, self._hot_expert_set)
            pin = layer_idx < self.config.num_hash_layers or eid in layer_hot
            self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=pin)
            return self.expert_cache.get(cache_key)

        # Check async prefetch cache before blocking disk read
        if getattr(self, '_prefetch_worker', None) is not None:
            prefetched = self._prefetch_worker.get(layer_idx, eid)
            if prefetched is not None:
                w1 = prefetched.get(("w1", "data"))
                s1 = prefetched.get(("w1", "scale"))
                w3 = prefetched.get(("w3", "data"))
                s3 = prefetched.get(("w3", "scale"))
                w2 = prefetched.get(("w2", "data"))
                s2 = prefetched.get(("w2", "scale"))
                if w1 is not None:
                    self._gpu_expert_store.cache_on_gpu(
                        layer_idx, eid, w1, s1, w3, s3, w2, s2)
                    w1_entry = self._make_raw_entry(w1, s1)
                    w3_entry = self._make_raw_entry(w3, s3)
                    w2_entry = self._make_raw_entry(w2, s2)
                    layer_hot = self._hot_expert_set_by_layer.get(layer_idx, self._hot_expert_set)
                    pin = layer_idx < self.config.num_hash_layers or eid in layer_hot
                    self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=pin)
                    return self.expert_cache.get(cache_key)

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

        self._gpu_expert_store.cache_on_gpu(
            layer_idx, eid, w1, s1, w3, s3, w2, s2)
        w1_entry = self._make_raw_entry(w1, s1)
        w3_entry = self._make_raw_entry(w3, s3)
        w2_entry = self._make_raw_entry(w2, s2)

        layer_hot = self._hot_expert_set_by_layer.get(layer_idx, self._hot_expert_set)
        pin = layer_idx < self.config.num_hash_layers or eid in layer_hot
        self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=pin)
        return self.expert_cache.get(cache_key)

    def _load_expert_raw(self, layer_idx, eid):
        raw = self._load_expert_weights(layer_idx, eid)
        if raw is None:
            if self._cpu_fallback_enabled and layer_idx in self._cpu_fallback_layers:
                raw = self._cpu_ffn_fallback(layer_idx, eid)
            if raw is None:
                return None
        return raw

    def _load_expert_deq(self, layer_idx, eid):
        raw = self._load_expert_weights(layer_idx, eid)
        if raw is None:
            if self._cpu_fallback_enabled and layer_idx in self._cpu_fallback_layers:
                raw = self._cpu_ffn_fallback(layer_idx, eid)
            if raw is None:
                return None
        deq = self.expert_cache.deq(f"{layer_idx}_{eid}")
        if deq is None:
            deq = raw
        return deq

    def _load_expert_fp4_raw(self, layer_idx, eid):
        hot_key = (layer_idx, eid)
        if hot_key in self._gpu_hot_experts:
            return self._gpu_hot_experts[hot_key]

        if hot_key in self._gpu_bf16_cache:
            self._gpu_bf16_cache.move_to_end(hot_key)
            return self._gpu_bf16_cache[hot_key]

        raw = self._load_expert_raw(layer_idx, eid)
        if raw is None:
            return None
        w1_entry, w3_entry, w2_entry = raw
        if w1_entry is None:
            return None

        w1_data, w1_scale, w1_fmt = w1_entry
        w3_data, w3_scale, w3_fmt = w3_entry
        w2_data, w2_scale, w2_fmt = w2_entry

        is_fp4_gpu = all(f in ("fp4_gpu", "fp4") for f in (w1_fmt, w3_fmt, w2_fmt))
        if not is_fp4_gpu:
            return None

        w1_dev = w1_data.device if w1_data is not None else torch.device("cpu")
        if w1_data is not None and w1_dev.type != self.device.type:
            w1_data = w1_data.to(self.device, non_blocking=True)
            if w1_scale is not None:
                w1_scale = w1_scale.to(self.device, non_blocking=True)
        w3_dev = w3_data.device if w3_data is not None else torch.device("cpu")
        if w3_data is not None and w3_dev.type != self.device.type:
            w3_data = w3_data.to(self.device, non_blocking=True)
            if w3_scale is not None:
                w3_scale = w3_scale.to(self.device, non_blocking=True)
        w2_dev = w2_data.device if w2_data is not None else torch.device("cpu")
        if w2_data is not None and w2_dev.type != self.device.type:
            w2_data = w2_data.to(self.device, non_blocking=True)
            if w2_scale is not None:
                w2_scale = w2_scale.to(self.device, non_blocking=True)

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

        w1_b, w3_b, w2_b = triton_dequantize_fp4_all(
            w1_data, w1_scale, w3_data, w3_scale, w2_data, w2_scale)

        layer_hot = self._hot_expert_set_by_layer.get(layer_idx, self._hot_expert_set)
        if eid in layer_hot:
            if len(self._gpu_hot_experts) >= self._max_hot_experts:
                self._gpu_hot_experts.pop(next(iter(self._gpu_hot_experts)))
            self._gpu_hot_experts[hot_key] = (w1_b, w3_b, w2_b)
        else:
            if len(self._gpu_bf16_cache) >= self._max_bf16_cache:
                self._gpu_bf16_cache.pop(next(iter(self._gpu_bf16_cache)))
            self._gpu_bf16_cache[hot_key] = (w1_b, w3_b, w2_b)

        return (w1_b, w3_b, w2_b)

    def _load_gpu_hot_expert_bf16(self, layer: int, eid: int):
        key = (layer, eid)
        if key in self._gpu_hot_experts:
            return self._gpu_hot_experts[key]
        result = self._load_expert_fp4_raw(layer, eid)
        if result is None:
            return None
        if len(result) == 3:
            return result
        if len(result) == 6:
            w1_d, w1_s, w3_d, w3_s, w2_d, w2_s = result
            w1_b = load_fp4_weight(w1_d, w1_s).to(torch.bfloat16).to(self.device)
            w3_b = load_fp4_weight(w3_d, w3_s).to(torch.bfloat16).to(self.device)
            w2_b = load_fp4_weight(w2_d, w2_s).to(torch.bfloat16).to(self.device)
            if len(self._gpu_hot_experts) >= self._max_hot_experts:
                self._gpu_hot_experts.pop(next(iter(self._gpu_hot_experts)))
            self._gpu_hot_experts[key] = (w1_b, w3_b, w2_b)
            return (w1_b, w3_b, w2_b)
        return None

    def _forward_ffn_hot_batched(self, flat_hidden, flat_topk_idx, flat_topk_w, layer_idx):
        B, D = flat_hidden.shape
        num_topk = flat_topk_idx.shape[1]
        all_eids = set()
        for b in range(B):
            for k in range(num_topk):
                eid = int(flat_topk_idx[b, k].item())
                if eid >= 0:
                    all_eids.add(eid)
        if not all_eids:
            return torch.zeros_like(flat_hidden)
        loaded = {}
        I_dim = None
        for eid in sorted(all_eids):
            weights = self._load_gpu_hot_expert_bf16(layer_idx, eid)
            if weights is None:
                return None
            w1, w3, w2 = weights
            if I_dim is None:
                I_dim = w1.shape[0]
            loaded[eid] = (w1, w3, w2)
        loaded_eids = sorted(loaded.keys())
        eid_to_idx = {eid: i for i, eid in enumerate(loaded_eids)}
        num_e = len(loaded_eids)
        w1 = torch.cat([loaded[eid][0] for eid in loaded_eids], dim=0)
        w3 = torch.cat([loaded[eid][1] for eid in loaded_eids], dim=0)
        w2 = torch.cat([loaded[eid][2] for eid in loaded_eids], dim=1)
        gate = flat_hidden.to(w1.dtype) @ w1.T
        up = flat_hidden.to(w3.dtype) @ w3.T
        g = gate.float().clamp(max=self.config.swiglu_limit)
        u = up.float().clamp(min=-self.config.swiglu_limit, max=self.config.swiglu_limit)
        activated = (g * g.sigmoid() * u).to(torch.bfloat16)
        routing = torch.zeros(B, num_e * I_dim, device=flat_hidden.device, dtype=torch.bfloat16)
        for b in range(B):
            for k in range(num_topk):
                eid = int(flat_topk_idx[b, k].item())
                if eid < 0:
                    continue
                idx = eid_to_idx.get(eid)
                if idx is None:
                    continue
                routing[b, idx * I_dim:(idx + 1) * I_dim] = flat_topk_w[b, k]
        activated_weighted = activated * routing
        return activated_weighted.to(w2.dtype) @ w2.T

    def _all_routed_are_hot(self, topk_idx) -> bool:
        for b in range(topk_idx.shape[0]):
            for k in range(topk_idx.shape[1]):
                eid = int(topk_idx[b, k].item())
                if eid >= 0 and eid not in self._hot_expert_set:
                    return False
        return True

    def _forward_ffn(self, hidden_states, lw, layer_idx, input_ids=None):
        gate_w = self._deq("ffn.gate.weight", lw.get("ffn.gate.weight"), lw.get("ffn.gate.scale"), layer_idx)
        gate_bias = lw.get("ffn.gate.bias")
        tid2eid = lw.get("ffn.gate.tid2eid")

        if gate_w is None:
            return torch.zeros_like(hidden_states), set()

        B, T, D = hidden_states.shape
        total_tokens = B * T
        flat_hidden = hidden_states.reshape(total_tokens, D)

        if layer_idx < self.config.num_hash_layers and tid2eid is not None and input_ids is not None:
            topk_idx, topk_w = self._compute_hash_experts(input_ids, layer_idx, tid2eid)
        else:
            topk_idx, topk_w = self._compute_routing_experts(flat_hidden, gate_w, gate_bias)

        used_experts = set()
        ffn_out = torch.zeros_like(hidden_states)
        flat_topk_idx = topk_idx.reshape(total_tokens, self.config.num_experts_per_tok)
        flat_topk_w = topk_w.reshape(total_tokens, self.config.num_experts_per_tok)

        if self._hot_expert_set and total_tokens > 0:
            try:
                if self._all_routed_are_hot(flat_topk_idx):
                    hot_result = self._forward_ffn_hot_batched(
                        flat_hidden, flat_topk_idx, flat_topk_w, layer_idx)
                    if hot_result is not None:
                        ffn_out = hot_result.reshape(B, T, D)
                        shared_w = self._get_shared_expert(layer_idx)
                        if shared_w is not None:
                            w1_d, w3_d, w2_d = shared_w
                            shared_out = self._shared_ffn.forward(
                                hidden_states, w1_d, w3_d, w2_d, hidden_states.dtype)
                            ffn_out = ffn_out + shared_out
                        return ffn_out, used_experts
            except Exception:
                pass

        try:
            def _fp4_load(layer, eid):
                return self._load_expert_fp4_raw(layer, eid)
            result = self._fused_moe.forward(
                flat_hidden, flat_topk_idx, flat_topk_w,
                _fp4_load,
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
            shared_out = self._shared_ffn.forward(
                hidden_states, w1_d, w3_d, w2_d, hidden_states.dtype)
            ffn_out = ffn_out + shared_out

        if hasattr(self, 'predictor'):
            self.predictor.collect(layer_idx, flat_hidden, flat_topk_idx)

        return ffn_out, used_experts

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
        try:
            post_4d = post.unsqueeze(-1) if post.dim() == 3 else post
            result = tile_kernels.modeling.mhc.ops.mhc_post(
                hidden.float(), residual.float(), post_4d.float(), comb.float())
            return result.to(hidden.dtype)
        except Exception as e:
            self._log(f"mHC post kernel fallback (not critical): {type(e).__name__}")
            B, S, D = hidden.shape
            x_expanded = hidden.unsqueeze(2)
            term1 = post.unsqueeze(-1) * x_expanded
            residual_expanded = residual.unsqueeze(3)
            term2 = torch.sum(comb.unsqueeze(-1) * residual_expanded, dim=2)
            y = term1 + term2
            return y.to(hidden.dtype)

    def _hc_head(self, hidden_4d):
        B, T, hc_mult, D = hidden_4d.shape
        x = hidden_4d.reshape(B, T, hc_mult * D).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = torch.matmul(x, self.hc_head_fn.t()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        return (pre.unsqueeze(-1) * hidden_4d.float()).sum(dim=2).to(torch.bfloat16)

    def _prefetch_next_layer(self, layer_idx, lw, hidden_states=None):
        if not self._prefetch_enabled or self._prefetch_worker is None:
            return
        next_idx = layer_idx + 1
        if next_idx >= self.config.num_hidden_layers:
            return
        next_lw = self._get_layer_weights(next_idx)
        next_gate_w = next_lw.get("ffn.gate.weight")
        if next_gate_w is not None:
            if hidden_states is not None:
                h_flat = hidden_states.reshape(-1, self.config.hidden_size)
                scores = torch.matmul(
                    h_flat.to(next_gate_w.dtype), next_gate_w.t())
                scores = F.softplus(scores).sqrt()
                _, topk_idx = torch.topk(
                    scores, self.config.num_experts_per_tok + 16, dim=-1)
                predicted_ids = list(set(int(x) for x in topk_idx[0].tolist()))
            else:
                sample_input = torch.randn(
                    1, self.config.hidden_size,
                    device=self.device, dtype=torch.bfloat16)
                scores = torch.matmul(
                    sample_input.to(next_gate_w.dtype), next_gate_w.t())
                scores = F.softplus(scores).sqrt()
                _, topk_idx = torch.topk(
                    scores, self.config.num_experts_per_tok + 16, dim=-1)
                predicted_ids = topk_idx[0].tolist()
            self._prefetch_worker.prefetch(next_idx, predicted_ids)

    def _predict_and_prefetch_next(self, layer_idx, hidden_states, input_ids):
        if not self._prefetch_enabled:
            return
        if not hasattr(self, 'predictor'):
            return
        next_idx = layer_idx + 1
        if next_idx >= self.config.num_hidden_layers:
            return
        h_flat = hidden_states.reshape(-1, self.config.hidden_size)
        pred = self.predictor.predict(h_flat, input_ids, layer_idx)
        if pred.expert_ids and len(pred.expert_ids) > 0:
            if self._prefetch_stream is not None:
                with torch.cuda.stream(self._prefetch_stream):
                    for eid in pred.expert_ids[:8]:
                        self._load_expert_raw(next_idx, eid)

    def _load_shared_experts_gpu(self):
        count = 0
        for layer_idx in range(self.config.num_hidden_layers):
            if layer_idx not in self._shared_expert_weights:
                shared_prefix = f"layers.{layer_idx}.ffn.shared_experts"
                shared_keys = [f"{shared_prefix}.w1.weight", f"{shared_prefix}.w1.scale",
                               f"{shared_prefix}.w3.weight", f"{shared_prefix}.w3.scale",
                               f"{shared_prefix}.w2.weight", f"{shared_prefix}.w2.scale"]
                tensors = self.loader.get_weights(*shared_keys)
                w1 = tensors.get(shared_keys[0])
                if w1 is None:
                    self._shared_expert_weights[layer_idx] = None
                    continue
                s1 = tensors.get(shared_keys[1])
                w3_t = tensors.get(shared_keys[2])
                s3 = tensors.get(shared_keys[3])
                w2_t = tensors.get(shared_keys[4])
                s2 = tensors.get(shared_keys[5])
                dev = self.device
                self._shared_expert_weights[layer_idx] = (
                    w1.to(dev), s1.to(dev) if s1 is not None else None,
                    w3_t.to(dev) if w3_t is not None else None, s3.to(dev) if s3 is not None else None,
                    w2_t.to(dev) if w2_t is not None else None, s2.to(dev) if s2 is not None else None,
                    "fp8",
                )
                count += 1
        self._shared_experts_loaded = True
        self._log(f"Pre-loaded {count} shared experts as FP8 on GPU")

    def _get_shared_expert(self, layer_idx: int):
        cached = self._shared_expert_weights.get(layer_idx)
        if cached is not None:
            if len(cached) == 3:
                return cached
            if len(cached) == 7 and cached[6] == "fp8":
                w1_fp8, w1_s, w3_fp8, w3_s, w2_fp8, w2_s, _ = cached
                w1_bf = load_fp8_weight(w1_fp8, w1_s) if w1_fp8 is not None else None
                w3_bf = load_fp8_weight(w3_fp8, w3_s) if w3_fp8 is not None else None
                w2_bf = load_fp8_weight(w2_fp8, w2_s) if w2_fp8 is not None else None
                if w1_bf is not None and w3_bf is not None and w2_bf is not None:
                    result = (w1_bf, w3_bf, w2_bf)
                    self._shared_expert_weights[layer_idx] = result
                    return result
                self._shared_expert_weights[layer_idx] = None
                return None
        shared_prefix = f"layers.{layer_idx}.ffn.shared_experts"
        shared_keys = [f"{shared_prefix}.w1.weight", f"{shared_prefix}.w1.scale",
                       f"{shared_prefix}.w3.weight", f"{shared_prefix}.w3.scale",
                       f"{shared_prefix}.w2.weight", f"{shared_prefix}.w2.scale"]
        tensors = self.loader.get_weights(*shared_keys)
        w1 = tensors.get(shared_keys[0])
        if w1 is None:
            self._shared_expert_weights[layer_idx] = None
            return None
        s1 = tensors.get(shared_keys[1])
        w3_t = tensors.get(shared_keys[2])
        s3 = tensors.get(shared_keys[3])
        w2_t = tensors.get(shared_keys[4])
        s2 = tensors.get(shared_keys[5])
        assert w1.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.int8), \
            f"Shared expert w1 unexpected dtype: {w1.dtype}"
        def _load_w(data, scale):
            if data is None:
                return None
            dev = self.device
            if data.dtype == torch.int8:
                return load_fp4_weight(data.to(dev), scale.to(dev) if scale is not None else None)
            return load_fp8_weight(data.to(dev), scale.to(dev) if scale is not None else None)
        w1_d = _load_w(w1, s1)
        w3_d = _load_w(w3_t, s3) if w3_t is not None else None
        w2_d = _load_w(w2_t, s2) if w2_t is not None else None
        if w1_d is not None and w3_d is not None and w2_d is not None:
            weights = (w1_d, w3_d, w2_d)
            self._shared_expert_weights[layer_idx] = weights
            return weights
        self._shared_expert_weights[layer_idx] = None
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
            w1_entry = (w1.to("cpu"), s1.to(torch.float32).to("cpu") if s1 is not None else None, "fp4")
            w3_entry = (w3.to("cpu"), s3.to(torch.float32).to("cpu") if s3 is not None else None, "fp4")
            w2_entry = (w2.to("cpu"), s2.to(torch.float32).to("cpu") if s2 is not None else None, "fp4")
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
        gpu_store = getattr(self, '_gpu_expert_store', None)
        if gpu_store is not None:
            gpu_store.cache_on_gpu(-1, eid, w1, s1, w3, s3, w2, s2)
        w1_entry = (w1.to("cpu"), s1.to(torch.float32).to("cpu") if s1 is not None else None, "fp4")
        w3_entry = (w3.to("cpu"), s3.to(torch.float32).to("cpu") if s3 is not None else None, "fp4")
        w2_entry = (w2.to("cpu"), s2.to(torch.float32).to("cpu") if s2 is not None else None, "fp4")
        self.expert_cache.put(cache_key, w1_entry, w3_entry, w2_entry, pin=True)
        return self.expert_cache.get(cache_key)

    def _mtp_attn_1tok(self, h, w):
        """MLA self-attention for 1 token — simplified (no KV cache needed)."""
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
        freqs_cis = precompute_freqs_cis(rope_dim, T, theta=self.config.rope_theta).to(h.device)
        q = apply_rotary_emb(q, freqs_cis, rd=rope_dim)
        kv_latent = apply_rotary_emb(kv_latent, freqs_cis, rd=rope_dim)

        n_kv = self.config.num_key_value_heads
        n_groups = self.config.num_attention_heads // n_kv
        k = kv_latent.unsqueeze(2).transpose(1, 2)
        k_exp = k.unsqueeze(1).expand(-1, n_groups, -1, -1, -1).reshape(B, -1, k.shape[-2], k.shape[-1])
        v_exp = k_exp

        scale_f = self.config.head_dim ** -0.5
        attn = torch.matmul(q.float() * scale_f, k_exp.float().transpose(-2, -1))
        attn_p = torch.softmax(attn, dim=-1).to(v_exp.dtype)
        out = torch.matmul(attn_p, v_exp)

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

    def _mtp_forward_draft(self, h_4d, input_ids=None):
        """Full MTP transformer layer forward pass for 1 token.

        h_4d: [B=1, T=1, hc_mult=4, D=4096]
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

        attn_out = self._mtp_attn_1tok(h_pre, w)

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

    def _warmup_page_cache(self):
        """Sequentially read all safetensors to warm the Linux page cache.

        With 150GB weights and 900GB available RAM, all weight files should
        fit in page cache after warmup.  Using 8 threads to saturate the RAID
        read bandwidth (~1.5 GB/s measured → ~100s for 150GB).
        """
        import concurrent.futures

        safetensors_files = self.loader._safetensors_files()
        if not safetensors_files:
            return

        total_gb = sum(os.path.getsize(f) for f in safetensors_files) / (1024**3)
        self._log(f"Warming page cache: {len(safetensors_files)} files, "
                  f"{total_gb:.1f} GB...")
        t0 = time.time()

        def read_file(fpath):
            try:
                with open(fpath, "rb") as f:
                    chunk_size = 16 * 1024 * 1024  # 16MB chunks
                    while f.read(chunk_size):
                        pass
            except Exception:
                pass

        max_workers = min(8, len(safetensors_files))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(read_file, safetensors_files))

        elapsed = time.time() - t0
        bw = total_gb / elapsed if elapsed > 0 else 0
        self._log(f"Page cache warmup done in {elapsed:.1f}s ({bw:.1f} GB/s)")

    def _preload_all_experts(self):
        """Load every (layer, expert) pair into ExpertWeightCache during init.

        Eliminates all file I/O during inference — trades init time (~115s at
        1.5 GB/s disk) for zero I/O during generate().  Uses multiple threads
        to saturate the disk read bandwidth.

        Memory cost: ~169 GB CPU RAM (all 11,008 expert pairs in FP4).
        The ExpertWeightCache is sized at 12,288 to hold everything.
        """
        import concurrent.futures
        num_layers = self.config.num_hidden_layers
        num_experts = self.config.n_routed_experts
        total = num_layers * num_experts
        self._log(f"Preloading all {total} expert pairs "
                  f"({num_layers}L × {num_experts}E)...")

        # Build the full task list — one (layer, eid) per expert
        tasks = [(li, ei) for li in range(num_layers)
                 for ei in range(num_experts)]

        t0 = time.time()
        count = [0]

        def load_one(args):
            li, ei = args
            self._load_expert_weights(li, ei)
            count[0] += 1

        # 8 threads is enough to saturate 1.5 GB/s disk on shared RAID
        max_workers = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(load_one, tasks))

        elapsed = time.time() - t0
        bw = total * 15.7 / 1024 / elapsed if elapsed > 0 else 0
        self._log(f"Preload done: {count[0]}/{total} experts in "
                  f"{elapsed:.1f}s ({bw:.1f} GB/s)")
        self._log(f"Expert cache entries: {len(self.expert_cache.cache)}")

    def _warmup(self) -> bool:
        if self._warmed_up:
            return True
        self._log("Warming up kernels...")

        cast_ok = False
        try:
            dummy = torch.zeros(1, 1, device=self.device, dtype=torch.bfloat16)
            s = torch.ones(1, 1, device=self.device)
            _ = tile_kernels.quant.cast_back((dummy, s), 'bf16', (128, 128))
            cast_ok = True
            self._log("cast_back kernel warmup OK")
        except Exception as e:
            self._log(f"Warmup: cast_back kernel failed ({type(e).__name__}), "
                      f"quantized ops may JIT-compile on first use")
        finally:
            del dummy
            try:
                del s
            except NameError:
                pass

        moe_ok = False
        try:
            from tile_kernels.moe import get_fused_mapping, expand_to_fused, reduce_fused
            from tile_kernels.torch.moe import inplace_unique_group_indices
            dw = torch.randn(16, 4096, device=self.device, dtype=torch.bfloat16)
            ti = torch.randint(0, 256, (16, 6), device=self.device, dtype=torch.int64)
            inplace_unique_group_indices(ti, 256)
            m = get_fused_mapping(ti, 256, 4096, 32)
            ex = expand_to_fused(dw, m[3], m[0])
            tw = torch.randn(16, 6, device=self.device, dtype=torch.float32)
            _ = reduce_fused(ex, tw, m[3])
            torch.cuda.synchronize(self.device)
            moe_ok = True
            self._log("MoE kernels warmup OK")
        except Exception as e:
            self._log(f"Warmup: MoE kernels failed ({type(e).__name__}), "
                      f"fused routing may JIT-compile on first use")
        finally:
            try:
                del dw, ti, ex, tw, m
            except NameError:
                pass
            torch.cuda.empty_cache()

        all_ok = cast_ok and moe_ok
        self._warmed_up = all_ok
        status = "OK" if all_ok else "partial failure"
        self._log(f"Warmup complete: {status} (cast={cast_ok}, moe={moe_ok})")
        return all_ok

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

        draft_tokens = []
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

            # enorm → e_proj (2D → 2D) → unsqueeze to 4D
            if enorm_w is not None:
                tok_emb = rms_norm(tok_emb, enorm_w, self.config.rms_norm_eps)
            emb_proj_2d = tok_emb.to(e_proj_bf16.dtype) @ e_proj_bf16.t()
            emb_proj_4d = emb_proj_2d.unsqueeze(2)  # [B,1,1,D] broadcasts with [B,1,hc,D]

            h_combined_4d = h_main_proj_4d + emb_proj_4d

            # Full MTP Block forward: MHC_attn → attn → MHC_ffn → FFN
            h_mtp_out = self._mtp_forward_draft(h_combined_4d)

            # hc_head → norm → lm_head
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

                    residual_attn = h
                    h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_attn")
                    if lw.get("attn_norm.weight") is not None:
                        h_pre = rms_norm(h_pre, lw["attn_norm.weight"].to(torch.bfloat16),
                                         self.config.rms_norm_eps)
                    attn_out = self._forward_attn(h_pre, lw, layer_idx)
                    if post is not None and comb is not None:
                        h = self._process_mhc_post(attn_out, residual_attn, post, comb)
                    else:
                        h = h + attn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

                    residual = h
                    h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_ffn")
                    if lw.get("ffn_norm.weight") is not None:
                        h_pre = rms_norm(h_pre, lw["ffn_norm.weight"].to(torch.bfloat16),
                                         self.config.rms_norm_eps)
                    ffn_out, _ = self._forward_ffn(h_pre, lw, layer_idx, draft_token)
                    if post is not None and comb is not None:
                        h = self._process_mhc_post(ffn_out, residual, post, comb)
                    else:
                        h = h + ffn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

                h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
                if self.norm_weight is not None:
                    h_3d = rms_norm(h_3d, self.norm_weight, self.config.rms_norm_eps)
                logits = torch.matmul(h_3d.to(self.lm_head.dtype), self.lm_head.t())

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
    def generate(self, input_ids, max_new_tokens=50, temperature=0.6):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        self._log(f"Generate: {T} prompt tokens, max_new={max_new_tokens}")

        self.layer_states = {}
        self._deq_cache.clear()
        self.expert_cache.clear()
        self._layer_weight_cache.clear()
        clear_deq_cache()
        for compressor in self._compressors.values():
            compressor.reset()
        for indexer in self._indexers.values():
            indexer.reset()
        for hybrid in self._hybrid_kv.values():
            hybrid.reset()
        self._global_pos = 0
        if self._prefetch_worker is not None:
            self._prefetch_worker.clear()
        self._init_prefetch()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.time()

        h = self.embed[input_ids].to(torch.bfloat16)
        h = h.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

        self._phase = "prefill"
        for layer_idx in range(self.config.num_hidden_layers):
            lw = self._get_layer_weights(layer_idx)

            residual_attn = h
            h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_attn")
            if lw.get("attn_norm.weight") is not None:
                h_pre = rms_norm(h_pre, lw["attn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)
            attn_out = self._forward_attn(h_pre, lw, layer_idx)
            if post is not None and comb is not None:
                h = self._process_mhc_post(attn_out, residual_attn, post, comb)
            else:
                h = h + attn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

            residual = h
            h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_ffn")
            if lw.get("ffn_norm.weight") is not None:
                h_pre = rms_norm(h_pre, lw["ffn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)
            ffn_out, used_experts = self._forward_ffn(h_pre, lw, layer_idx, input_ids)
            if post is not None and comb is not None:
                h = self._process_mhc_post(ffn_out, residual, post, comb)
            else:
                h = h + ffn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

            self._predict_and_prefetch_next(layer_idx, h_pre, input_ids)
            if self._prefetch_worker is not None:
                self._prefetch_next_layer(layer_idx, lw, hidden_states=h_pre[:, -1:, :])

            if (layer_idx + 1) % 5 == 0:
                mem = torch.cuda.memory_allocated() / (1024**3)
                if mem > 18:
                    if self._prefetch_worker is not None:
                        self._prefetch_worker.clear()
                    for li in range(max(0, layer_idx - 5), layer_idx + 1):
                        if li in self.layer_states and self.layer_states[li].kv_latent_cache is not None:
                            if self.layer_states[li].kv_latent_cache.shape[1] > self.config.sliding_window * 2:
                                with torch.cuda.stream(self._kv_offload_stream):
                                    self.layer_states[li].archived_kv = self.layer_states[li].kv_latent_cache[:, :-self.config.sliding_window].contiguous().to("cpu", non_blocking=True)
                                    self.layer_states[li].kv_latent_cache = self.layer_states[li].kv_latent_cache[:, -self.config.sliding_window:].contiguous()
                            if self.layer_states[li].compressed_kv_data is not None:
                                self.layer_states[li].compressed_kv_data = None
                                self.layer_states[li].compressed_kv_idx = None
                    self._gpu_expert_store.resize(32)
                    torch.cuda.empty_cache()
                    self._log(f"  Layer {layer_idx}: freed caches, mem={mem:.1f}GB")

        h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
        if self.norm_weight is not None:
            h_3d = rms_norm(h_3d, self.norm_weight, self.config.rms_norm_eps)
        logits = torch.matmul(h_3d[:, -1:].to(self.lm_head.dtype), self.lm_head.t())

        self._global_pos = T  # prefill done, advance position

        if temperature > 0:
            probs = F.softmax(logits[:, -1].float() / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        else:
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)

        generated = [next_id]
        prefill_end = time.time()

        mtp_num_draft = 3 if self._mtp_loaded else 0

        self._phase = "decode"
        step = 0
        while step < max_new_tokens - 1:
            self._global_pos = T + step
            h = self.embed[next_id].to(torch.bfloat16)
            h = h.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
            for layer_idx in range(self.config.num_hidden_layers):
                lw = self._get_layer_weights(layer_idx)

                residual_attn = h
                h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_attn")
                if lw.get("attn_norm.weight") is not None:
                    h_pre = rms_norm(h_pre, lw["attn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)
                attn_out = self._forward_attn(h_pre, lw, layer_idx)
                if post is not None and comb is not None:
                    h = self._process_mhc_post(attn_out, residual_attn, post, comb)
                else:
                    h = h + attn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

                residual = h
                h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_ffn")
                if lw.get("ffn_norm.weight") is not None:
                    h_pre = rms_norm(h_pre, lw["ffn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)
                ffn_out, used_experts = self._forward_ffn(h_pre, lw, layer_idx, next_id)
                if post is not None and comb is not None:
                    h = self._process_mhc_post(ffn_out, residual, post, comb)
                else:
                    h = h + ffn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

                self._predict_and_prefetch_next(layer_idx, h_pre, next_id)
                if self._prefetch_worker is not None:
                    self._prefetch_next_layer(layer_idx, lw, hidden_states=h_pre[:, -1:, :])

                if layer_idx == self.config.num_hidden_layers // 2:
                    mem = torch.cuda.memory_allocated() / (1024**3)
                    if mem > 18:
                        for li in range(layer_idx):
                            if li in self.layer_states and self.layer_states[li].kv_latent_cache is not None:
                                if self.layer_states[li].kv_latent_cache.shape[1] > self.config.sliding_window * 2:
                                    with torch.cuda.stream(self._kv_offload_stream):
                                        self.layer_states[li].archived_kv = self.layer_states[li].kv_latent_cache[:, :-self.config.sliding_window].contiguous().to("cpu", non_blocking=True)
                                        self.layer_states[li].kv_latent_cache = self.layer_states[li].kv_latent_cache[:, -self.config.sliding_window:].contiguous()

            last_h_for_mtp = h

            h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
            if self.norm_weight is not None:
                h_3d = rms_norm(h_3d, self.norm_weight, self.config.rms_norm_eps)
            logits = torch.matmul(h_3d.to(self.lm_head.dtype), self.lm_head.t())

            if temperature > 0:
                probs = F.softmax(logits[:, -1].float() / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_id)
            step += 1

            if mtp_num_draft > 0 and step < max_new_tokens - 1:
                all_inputs = torch.cat(
                    [input_ids] + generated, dim=-1)
                last_token = all_inputs[:, -1:]
                draft_ids, n_draft = self._mtp_generate_draft(
                    last_h_for_mtp, mtp_num_draft, temperature, last_token_id=last_token)
                if draft_ids is not None and n_draft > 0:
                    if self._mtp_eager:
                        for i in range(n_draft):
                            generated.append(draft_ids[:, i:i+1])
                            step += 1
                            if step >= max_new_tokens - 1:
                                break
                        if step < max_new_tokens - 1:
                            next_id = draft_ids[:, -1:]
                        continue
                    n_acc, bonus_logits = self._mtp_accept_drafts(
                        all_inputs, draft_ids, temperature)
                    if n_acc > 0:
                        accepted_ids = draft_ids[:, :n_acc]
                        for i in range(n_acc):
                            generated.append(accepted_ids[:, i:i+1])
                            step += 1
                            if step >= max_new_tokens - 1:
                                break
                        if step < max_new_tokens - 1:
                            if n_acc < n_draft and bonus_logits is not None and bonus_logits.dim() >= 1:
                                if temperature > 0:
                                    new_probs = F.softmax(bonus_logits.float() / temperature, dim=-1)
                                    next_id = torch.multinomial(new_probs, 1)
                                else:
                                    next_id = bonus_logits.argmax(dim=-1, keepdim=True)
                                generated.append(next_id)
                                step += 1
                            elif n_acc == n_draft:
                                next_id = accepted_ids[:, -1:]
                        continue

        total_time = time.time() - start
        prefill_time = prefill_end - start
        decode_time = total_time - prefill_time
        all_tokens = torch.cat([input_ids] + generated, dim=-1)
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
        self.loader.close()
        self._log(f"Done: {result['num_generated_tokens']} tokens in {total_time:.1f}s, "
                  f"peak mem: {result['peak_memory_gb']:.1f}GB")
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", default="weights")
    parser.add_argument("--prompt", default="Hello, world")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prefetch", action="store_true", help="Enable expert prefetch (may increase memory)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable CPU fallback")
    parser.add_argument("--hot-experts", default="hot_experts.json", help="Hot experts JSON path")
    parser.add_argument("--use-mtp", action="store_true", help="Enable MTP speculative decoding")
    parser.add_argument("--mtp-eager", action="store_true",
                        help="Eager MTP: accept all drafts without verification (fast, risky)")
    parser.add_argument("--reprobe", action="store_true",
                        help="Force re-probe hardware profile, overwrite hw_profile.json")
    args = parser.parse_args()
    engine = HomeSeekInferenceEngine(args.weight_dir, verbose=args.verbose,
                                     hot_experts_path=args.hot_experts, reprobe=args.reprobe)
    if args.prefetch:
        engine._prefetch_enabled = True
    if args.no_fallback:
        engine._cpu_fallback_enabled = False
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
