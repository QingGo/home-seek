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
from home_seek.fused_moe import FusedMoEFFN, SharedExpertFFN, clear_deq_cache
from home_seek.gpu_expert_store import AllExpertFP4Store
from home_seek.compressor import Compressor as NewCompressor
from home_seek.lightning_indexer import LightningIndexer
from home_seek.hybrid_kv_cache import HybridKVCache, SWACache
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
        self.cache.clear()
        self._hot_deq.clear()
        self.pinned.clear()

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
                 hot_experts_path: str = "hot_experts.json"):
        config_path = os.path.join(weight_dir, "config.json")
        self.config = DeepSeekV4FlashConfig(config_path)
        self.weight_dir = weight_dir
        self.device = torch.device(device)
        self.verbose = verbose
        self.loader = WeightLoader(weight_dir, device)
        self.expert_cache = ExpertWeightCache(max_experts=512, device=device, hot_deq_size=0)
        self.layer_states = {}
        self._deq_cache = OrderedDict()
        self._compressors = {}       # per-layer Compressor instances (lazy)
        self._indexers = {}          # per-layer LightningIndexer instances (lazy)
        self._hybrid_kv = {}         # per-layer HybridKVCache instances (lazy)
        self._global_pos = 0         # current global sequence position
        self._prefetch_worker = None
        self._load_global_weights()
        self._prefetch_enabled = True
        self._cpu_fallback_enabled = True
        self._cpu_fallback_layers = set(range(min(3, self.config.num_hidden_layers))) | set(range(max(0, self.config.num_hidden_layers - 3), self.config.num_hidden_layers))
        self._hot_expert_ids = []
        self._hash_expert_ids = []
        self._preload_hot_experts(hot_experts_path)
        self._kv_offload_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._compress_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self._shared_expert_weights = {}
        self._shared_experts_loaded = False
        self._mtp_weights = {}
        self._mtp_loaded = False
        self._warmed_up = False

        self._gpu_expert_store = AllExpertFP4Store(
            weight_dir, self.config, str(self.device), max_experts=512)

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
        self._hot_expert_ids = data.get("top_16_hot_experts", [])
        hash_ids = data.get("hash_layer_expert_ids", [])
        self._hash_expert_ids = hash_ids[:18] if len(hash_ids) > 18 else hash_ids
        self._log(f"Hot experts: {len(self._hot_expert_ids)} IDs, "
                  f"Hash experts: {len(self._hash_expert_ids)} IDs")

    def _init_prefetch(self):
        if self._prefetch_enabled and self._prefetch_worker is None:
            self._prefetch_worker = AsyncPrefetchWorker(
                self.weight_dir, self.loader.weight_map, str(self.device),
                num_prefetch=12)

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

        head_dim = self.config.head_dim

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
            sw_len = k_sw.shape[-2]
            attn[:, :, :, sw_len - 1:sw_len] = attn[:, :, :, sw_len - 1:sw_len] + attn_sink.view(1, -1, 1, 1)

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
        scores = torch.matmul(hidden_states.to(gate_w.dtype), gate_w.t())
        if gate_bias is not None:
            scores = scores + gate_bias.to(scores.dtype)
        scores = F.softplus(scores).sqrt()
        topk_w, topk_idx = torch.topk(scores, self.config.num_experts_per_tok, dim=-1)
        topk_sum = topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        topk_w = topk_w / topk_sum * self.config.routed_scaling_factor
        return topk_idx, topk_w

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
            pin = layer_idx < self.config.num_hash_layers or eid in self._hot_expert_ids
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
            if self._prefetch_worker is not None:
                prefetched = self._prefetch_worker.get(layer_idx, eid)
                if prefetched is not None:
                    w1 = prefetched.get(("w1", "data"))
                    s1 = prefetched.get(("w1", "scale"))
                    w3 = prefetched.get(("w3", "data"))
                    s3 = prefetched.get(("w3", "scale"))
                    w2 = prefetched.get(("w2", "data"))
                    s2 = prefetched.get(("w2", "scale"))
            if w1 is None:
                return None

        self._gpu_expert_store.cache_on_gpu(
            layer_idx, eid, w1, s1, w3, s3, w2, s2)
        w1_entry = self._make_raw_entry(w1, s1)
        w3_entry = self._make_raw_entry(w3, s3)
        w2_entry = self._make_raw_entry(w2, s2)

        pin = layer_idx < self.config.num_hash_layers or eid in self._hot_expert_ids
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

        return (w1_data, w1_scale, w3_data, w3_scale, w2_data, w2_scale)

    def _forward_ffn(self, hidden_states, lw, layer_idx, input_ids=None):
        gate_w = self._deq("ffn.gate.weight", lw.get("ffn.gate.weight"), lw.get("ffn.gate.scale"), layer_idx)
        gate_bias = lw.get("ffn.gate.bias")
        tid2eid = lw.get("ffn.gate.tid2eid")

        if gate_w is None:
            return torch.zeros_like(hidden_states), set()

        if layer_idx < self.config.num_hash_layers and tid2eid is not None and input_ids is not None:
            topk_idx, topk_w = self._compute_hash_experts(input_ids, layer_idx, tid2eid)
        else:
            topk_idx, topk_w = self._compute_routing_experts(hidden_states, gate_w, gate_bias)

        used_experts = set()
        ffn_out = torch.zeros_like(hidden_states)
        B, T, D = hidden_states.shape
        total_tokens = B * T
        flat_hidden = hidden_states.reshape(total_tokens, D)
        flat_topk_idx = topk_idx.reshape(total_tokens, self.config.num_experts_per_tok)
        flat_topk_w = topk_w.reshape(total_tokens, self.config.num_experts_per_tok)

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
            hc = comb.shape[-1]
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
                    scores, self.config.num_experts_per_tok + 4, dim=-1)
                predicted_ids = list(set(int(x) for x in topk_idx[0].tolist()))
            else:
                sample_input = torch.randn(
                    1, self.config.hidden_size,
                    device=self.device, dtype=torch.bfloat16)
                scores = torch.matmul(
                    sample_input.to(next_gate_w.dtype), next_gate_w.t())
                scores = F.softplus(scores).sqrt()
                _, topk_idx = torch.topk(
                    scores, self.config.num_experts_per_tok + 4, dim=-1)
                predicted_ids = topk_idx[0].tolist()
            self._prefetch_worker.prefetch(next_idx, predicted_ids)

    def _load_shared_experts_gpu(self):
        self._shared_experts_loaded = True
        self._log("Shared expert lazy loading enabled (loaded on first use)")

    def _get_shared_expert(self, layer_idx: int):
        if layer_idx in self._shared_expert_weights:
            return self._shared_expert_weights[layer_idx]
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
        self._mtp_loaded = True
        self._log("MTP lazy loading enabled")

    def _get_mtp_weight(self, key: str):
        if key in self._mtp_weights:
            return self._mtp_weights[key]
        w = self.loader.get_weight(key, device=str(self.device))
        if w is not None:
            w = w.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w
            self._mtp_weights[key] = w
        return w

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
    def _mtp_generate_draft(self, last_hidden, num_draft: int = 3, temperature: float = 0.6):
        device = self.device
        mtp_embed = self._get_mtp_weight("mtp.0.embed.weight")
        mtp_norm = self._get_mtp_weight("mtp.0.norm.weight")
        mtp_head = self._get_mtp_weight("mtp.0.head.weight")
        if mtp_head is None:
            mtp_head = self.lm_head
        mtp_head_norm = self._get_mtp_weight("mtp.0.head_norm.weight")
        if mtp_embed is None:
            return None, 0

        draft_tokens = []
        h = last_hidden
        for step in range(num_draft):
            h_flat = h[:, -1:, :]
            if mtp_norm is not None:
                h_flat = rms_norm(h_flat, mtp_norm.to(torch.bfloat16))
            logits = h_flat.to(mtp_head.dtype) @ mtp_head.t()
            if mtp_head_norm is not None:
                logits = rms_norm(logits, mtp_head_norm.to(torch.bfloat16))
            if temperature > 0:
                probs = torch.softmax(logits[:, -1].float() / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            draft_tokens.append(next_id)

            next_embed = mtp_embed[next_id].to(torch.bfloat16)
            h = torch.cat([h, next_embed], dim=1)

        draft_ids = torch.cat(draft_tokens, dim=-1)
        return draft_ids, len(draft_tokens)

    @torch.no_grad()
    def _mtp_accept_drafts(self, input_ids, draft_ids, temperature=0.6):
        if draft_ids is None or draft_ids.shape[1] == 0:
            return 0, None

        full_ids = torch.cat([input_ids, draft_ids], dim=-1)
        B, T_full = full_ids.shape

        h = self.embed[full_ids].to(torch.bfloat16)
        h = h.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

        layer_states_bak = {}
        for k, v in self.layer_states.items():
            if v.kv_latent_cache is not None:
                state_copy = LayerState(device=str(self.device))
                state_copy.kv_latent_cache = v.kv_latent_cache.clone()
                state_copy.archived_kv = v.archived_kv
                state_copy.archived_len = v.archived_len
                state_copy.compressed_kv_data = v.compressed_kv_data
                state_copy.compressed_kv_idx = v.compressed_kv_idx
                layer_states_bak[k] = state_copy

        self.layer_states = {}
        try:
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
                ffn_out, _ = self._forward_ffn(h_pre, lw, layer_idx, full_ids)
                if post is not None and comb is not None:
                    h = self._process_mhc_post(ffn_out, residual, post, comb)
                else:
                    h = h + ffn_out.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)

            h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
            if self.norm_weight is not None:
                h_3d = rms_norm(h_3d, self.norm_weight, self.config.rms_norm_eps)

            all_logits = h_3d.to(self.lm_head.dtype) @ self.lm_head.t()
            draft_len = draft_ids.shape[1]

            n_accept = 0
            for i in range(draft_len):
                expected = draft_ids[:, i]
                pred_logits = all_logits[:, input_ids.shape[1] + i - 1, :]
                pred_id = pred_logits.argmax(dim=-1)
                if pred_id.item() == expected.item():
                    n_accept += 1
                else:
                    break
            return n_accept, all_logits[:, input_ids.shape[1] - 1 + n_accept, :]
        finally:
            self.layer_states = layer_states_bak

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.6):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        self._log(f"Generate: {T} prompt tokens, max_new={max_new_tokens}")

        self.layer_states = {}
        self._deq_cache.clear()
        self.expert_cache.clear()
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

            if self._prefetch_worker is not None:
                self._prefetch_next_layer(layer_idx, lw, hidden_states=h_pre[:, -1:, :])

            if layer_idx in self._cpu_fallback_layers and self._prefetch_worker is not None:
                self._prefetch_worker.clear()

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
                    self.expert_cache.trim(64)
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

        mtp_num_draft = 3 if self._mtp_loaded else 0
        verify_extra_seen = 0
        last_h_for_mtp = None

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

                if layer_idx == self.config.num_hidden_layers // 2:
                    mem = torch.cuda.memory_allocated() / (1024**3)
                    if mem > 18:
                        for li in range(layer_idx):
                            if li in self.layer_states and self.layer_states[li].kv_latent_cache is not None:
                                if self.layer_states[li].kv_latent_cache.shape[1] > self.config.sliding_window * 2:
                                    with torch.cuda.stream(self._kv_offload_stream):
                                        self.layer_states[li].archived_kv = self.layer_states[li].kv_latent_cache[:, :-self.config.sliding_window].contiguous().to("cpu", non_blocking=True)
                                        self.layer_states[li].kv_latent_cache = self.layer_states[li].kv_latent_cache[:, -self.config.sliding_window:].contiguous()

            if layer_idx == self.config.num_hidden_layers - 1:
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
                draft_ids, n_draft = self._mtp_generate_draft(
                    last_h_for_mtp, mtp_num_draft, temperature)
                if draft_ids is not None and n_draft > 0:
                    n_acc, verify_logits = self._mtp_accept_drafts(
                        all_inputs, draft_ids, temperature)
                    if n_acc > 0:
                        accepted_ids = draft_ids[:, :n_acc]
                        for i in range(n_acc):
                            generated.append(accepted_ids[:, i:i+1])
                            step += 1
                            if step >= max_new_tokens - 1:
                                break
                        if step < max_new_tokens - 1:
                            if n_acc < n_draft and verify_logits is not None and verify_logits.dim() >= 1:
                                if temperature > 0:
                                    new_probs = F.softmax(verify_logits.float() / temperature, dim=-1)
                                    next_id = torch.multinomial(new_probs, 1)
                                else:
                                    next_id = verify_logits.argmax(dim=-1, keepdim=True)
                                generated.append(next_id)
                                step += 1
                            elif n_acc == n_draft:
                                next_id = accepted_ids[:, -1:]
                        continue

        total_time = time.time() - start
        all_tokens = torch.cat([input_ids] + generated, dim=-1)
        result = {
            "tokens": all_tokens,
            "total_time_s": total_time,
            "new_tokens_per_second": len(generated) / total_time,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "num_prompt_tokens": T,
            "num_generated_tokens": len(generated),
        }
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
    args = parser.parse_args()
    engine = HomeSeekInferenceEngine(args.weight_dir, verbose=args.verbose, hot_experts_path=args.hot_experts)
    if args.prefetch:
        engine._prefetch_enabled = True
    if args.no_fallback:
        engine._cpu_fallback_enabled = False

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
