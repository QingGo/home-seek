import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from collections import defaultdict, OrderedDict

import math
from home_seek.model_config import DeepSeekV4FlashConfig
from home_seek.mhc import mhc_split_sinkhorn
from home_seek.prefetch_worker import PrefetchWorker
from tile_reference import cast_back, unpack_from_e2m1fn_x2, swiglu_forward

_current_dir = os.path.dirname(os.path.abspath(__file__))
_encoding_dir = os.path.join(_current_dir, '../weights/encoding')
sys.path.insert(0, os.path.abspath(_encoding_dir))
from encoding_dsv4 import encode_messages


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x_normed).to(x.dtype)


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
        data_f32 = data.to(torch.float32)
        scale_f32 = _ue8m0_to_f32(scale)
        if scale_f32.shape != data_f32.shape:
            bh = data_f32.shape[0] // scale_f32.shape[0]
            bw = data_f32.shape[1] // scale_f32.shape[1]
            scale_f32 = scale_f32.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)
        return (data_f32 * scale_f32).to(torch.bfloat16)
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


class ExpertWeightCache:
    def __init__(self, max_experts: int = 64):
        self.max_experts = max_experts
        self.cache = OrderedDict()
        self.pinned = set()

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, w1, w3, w2, pin: bool = False):
        if key in self.cache:
            self.cache.move_to_end(key)
            return
        if not pin and len(self.cache) >= self.max_experts + len(self.pinned):
            for k, _ in list(self.cache.items()):
                if k not in self.pinned:
                    self.cache.pop(k)
                    break
        self.cache[key] = (w1, w3, w2)
        if pin:
            self.pinned.add(key)


class WeightLoader:
    def __init__(self, weight_dir: str, device: str = "cuda"):
        self.weight_dir = weight_dir
        self.device = torch.device(device)
        self.weight_map = {}
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

    def get_weight(self, key: str):
        fname = self.weight_map.get(key)
        if fname is None:
            return None
        fpath = os.path.join(self.weight_dir, fname)
        if not os.path.exists(fpath):
            return None
        try:
            with safe_open(fpath, framework="pt", device=str(self.device)) as f:
                return f.get_tensor(key)
        except Exception:
            return None

    def get_weights(self, *keys):
        results = {}
        file_keys = defaultdict(list)
        for k in keys:
            fname = self.weight_map.get(k)
            if fname:
                file_keys[fname].append(k)
        for fname, ks in file_keys.items():
            fpath = os.path.join(self.weight_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with safe_open(fpath, framework="pt", device=str(self.device)) as f:
                    for k in ks:
                        results[k] = f.get_tensor(k)
            except Exception:
                pass
        return results

    def get_layer_weight(self, layer: int, weight_type: str):
        return self.get_weight(f"layers.{layer}.{weight_type}")

    def get_attn_weight(self, layer: int, name: str):
        return self.get_weight(f"layers.{layer}.attn.{name}")

    def get_ffn_weight(self, layer: int, name: str):
        return self.get_weight(f"layers.{layer}.ffn.{name}")


class LayerState:
    def __init__(self, device: str = "cuda", active_window: int = 32768):
        self.kv_latent_cache = None
        self.compressed_kv = None
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


class CompressedKVCache:
    def __init__(self, compress_ratio: int, dim: int, device: str):
        self.compress_ratio = compress_ratio
        self.dim = dim
        self.device = torch.device(device)
        self.cache = torch.zeros(0, dim, device=self.device, dtype=torch.bfloat16)

    def append(self, compressed: torch.Tensor):
        self.cache = torch.cat([self.cache, compressed], dim=0)

    def get(self):
        return self.cache

    def clear(self):
        self.cache = torch.zeros(0, self.dim, device=self.device, dtype=torch.bfloat16)


class HomeSeekInferenceEngine:
    def __init__(self, weight_dir: str = "weights", device: str = "cuda", verbose: bool = False,
                 hot_experts_path: str = "hot_experts.json"):
        config_path = os.path.join(weight_dir, "config.json")
        self.config = DeepSeekV4FlashConfig(config_path)
        self.weight_dir = weight_dir
        self.device = torch.device(device)
        self.verbose = verbose
        self.loader = WeightLoader(weight_dir, device)
        self.expert_cache = ExpertWeightCache(max_experts=128)
        self.layer_states = {}
        self._deq_cache = OrderedDict()
        self._load_global_weights()
        self._prefetch_enabled = False
        self._prefetch_worker = PrefetchWorker(self.loader, device)
        self._prefetch_a = {}
        self._prefetch_b = {}
        self._prefetch_buffer = 'a'
        self._cpu_fallback_enabled = True
        self._cpu_fallback_layers = set(range(min(3, self.config.num_hidden_layers))) | set(range(max(0, self.config.num_hidden_layers - 3), self.config.num_hidden_layers))
        self._hot_expert_ids = []
        self._hash_expert_ids = []
        self._preload_hot_experts(hot_experts_path)

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

    def _load_global_weights(self):
        def safe(key):
            return self.loader.get_weight(key)

        self.embed = safe("embed.weight")
        if self.embed is not None and not isinstance(self.embed, torch.Tensor):
            self.embed = None
        self.embed = self.embed.to(torch.bfloat16) if self.embed is not None else None
        self._log(f"embed.weight: {self.embed.shape if self.embed is not None else 'missing'}")

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
        weight_types = [
            "attn_norm.weight", "ffn_norm.weight",
            "attn.wq_a.weight", "attn.wq_a.scale",
            "attn.wq_b.weight", "attn.wq_b.scale",
            "attn.wkv.weight", "attn.wkv.scale",
            "attn.wo_a.weight", "attn.wo_a.scale",
            "attn.wo_b.weight", "attn.wo_b.scale",
            "attn.q_norm.weight", "attn.kv_norm.weight",
            "attn.attn_sink",
            "ffn.gate.weight", "ffn.gate.scale",
            "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
            "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
        ]
        for t in weight_types:
            key = f"layers.{layer_idx}.{t}"
            lw[t] = self.loader.get_weight(key)

        ffn_has_bias = self.loader.get_weight(f"layers.{layer_idx}.ffn.gate.bias")
        if ffn_has_bias is not None:
            lw["ffn.gate.bias"] = ffn_has_bias

        tid2eid = self.loader.get_weight(f"layers.{layer_idx}.ffn.gate.tid2eid")
        if tid2eid is not None:
            lw["ffn.gate.tid2eid"] = tid2eid

        c_wkv = self.loader.get_weight(f"layers.{layer_idx}.attn.compressor.wkv.weight")
        if c_wkv is not None:
            lw["attn.compressor.wkv.weight"] = c_wkv
            lw["attn.compressor.wgate.weight"] = self.loader.get_weight(f"layers.{layer_idx}.attn.compressor.wgate.weight")
            lw["attn.compressor.norm.weight"] = self.loader.get_weight(f"layers.{layer_idx}.attn.compressor.norm.weight")
            lw["attn.compressor.ape"] = self.loader.get_weight(f"layers.{layer_idx}.attn.compressor.ape")

            idx_wq_b = self.loader.get_weight(f"layers.{layer_idx}.attn.indexer.wq_b.weight")
            if idx_wq_b is not None:
                lw["attn.indexer.wq_b.weight"] = idx_wq_b
                lw["attn.indexer.wq_b.scale"] = self.loader.get_weight(f"layers.{layer_idx}.attn.indexer.wq_b.scale")
                lw["attn.indexer.weights_proj.weight"] = self.loader.get_weight(f"layers.{layer_idx}.attn.indexer.weights_proj.weight")
                lw["attn.indexer.compressor.wkv.weight"] = self.loader.get_weight(f"layers.{layer_idx}.attn.indexer.compressor.wkv.weight")
                lw["attn.indexer.compressor.wgate.weight"] = self.loader.get_weight(f"layers.{layer_idx}.attn.indexer.compressor.wgate.weight")
                lw["attn.indexer.compressor.norm.weight"] = self.loader.get_weight(f"layers.{layer_idx}.attn.indexer.compressor.norm.weight")
                lw["attn.indexer.compressor.ape"] = self.loader.get_weight(f"layers.{layer_idx}.attn.indexer.compressor.ape")

        return lw

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
            if len(self._deq_cache) >= 4:
                for k in list(self._deq_cache.keys()):
                    del self._deq_cache[k]
                    break
            self._deq_cache[cache_key] = result
        return result

    def _compress_kv(self, hidden: torch.Tensor, lw: dict, layer_idx: int, state: LayerState):
        compress_ratio = self.config.get_compress_ratio(layer_idx)
        if compress_ratio == 0:
            return state

        c_wkv = lw.get("attn.compressor.wkv.weight")
        c_wgate = lw.get("attn.compressor.wgate.weight")
        c_norm = lw.get("attn.compressor.norm.weight")
        c_ape = lw.get("attn.compressor.ape")

        if c_wkv is None:
            return state

        B, T, D = hidden.shape
        c_dim = c_wkv.shape[0]

        c_hidden = torch.matmul(hidden.to(c_wkv.dtype), c_wkv.t())
        c_gate = torch.matmul(hidden.to(c_wgate.dtype), c_wgate.t())
        c_gate = torch.sigmoid(c_gate.float()).to(c_hidden.dtype)
        c_hidden = c_hidden * c_gate

        if c_hidden.shape[1] >= compress_ratio:
            c_hidden = c_hidden[:, :c_hidden.shape[1] // compress_ratio * compress_ratio, :]
            c_hidden = c_hidden.view(B, -1, compress_ratio, c_dim).mean(dim=2)
        else:
            c_hidden = torch.zeros(B, 0, c_dim, device=c_hidden.device, dtype=c_hidden.dtype)

        if c_ape is not None and c_ape.shape[-1] == c_dim and c_hidden.shape[1] > 0:
            ape_seq = min(c_ape.shape[0], c_hidden.shape[1])
            c_hidden[:, :ape_seq, :] = c_hidden[:, :ape_seq, :] + c_ape[:ape_seq, :].to(c_hidden.dtype)

        norm_dim = c_norm.shape[-1] if c_norm is not None else c_dim
        if c_dim != norm_dim:
            if c_hidden.shape[1] > 0:
                c_hidden = c_hidden.view(B, -1, c_dim // norm_dim, norm_dim).mean(dim=2)
            else:
                c_hidden = torch.zeros(B, 0, norm_dim, device=c_hidden.device, dtype=c_hidden.dtype)
            c_dim = norm_dim

        if c_norm is not None and c_hidden.shape[1] > 0:
            c_hidden = rms_norm(c_hidden, c_norm.to(torch.bfloat16))

        if state.compressed_kv is None:
            state.compressed_kv = CompressedKVCache(compress_ratio, c_dim, str(self.device))
        state.compressed_kv.append(c_hidden.squeeze(0))
        return state

    def _get_compressed_attention_kv(self, state: LayerState, lw: dict):
        compressed_kv = state.compressed_kv.get() if state.compressed_kv is not None else None
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
        compressed_kv = compressed_kv.unsqueeze(0).unsqueeze(2).contiguous()
        return compressed_kv  # [B=1, n_kv_head=1, seq, head_dim]

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

        state = self._compress_kv(hidden_states, lw, layer_idx, state)

        T_kv = all_kv_latent.shape[1]
        sw = min(self.config.sliding_window, T_kv)
        kv_sw = all_kv_latent[:, -sw:, :]
        k_sw, v_sw = self._expand_kv(kv_sw)

        compressed_kv = self._get_compressed_attention_kv(state, lw)
        if compressed_kv is not None:
            if compressed_kv.dim() < k_sw.dim():
                compressed_kv = compressed_kv.unsqueeze(1).expand(-1, k_sw.shape[1], -1, -1)
            if compressed_kv.shape[1] != k_sw.shape[1]:
                compressed_kv = compressed_kv.expand(-1, k_sw.shape[1], -1, -1)
            if compressed_kv.shape[3] != k_sw.shape[3]:
                compressed_kv = compressed_kv[:, :, :, :k_sw.shape[3]]
            min_seq = min(k_sw.shape[2], compressed_kv.shape[2])
            k_all = torch.cat([k_sw, compressed_kv.to(k_sw.dtype)], dim=-2)
            v_all = torch.cat([v_sw, compressed_kv.to(v_sw.dtype)], dim=-2)
        else:
            k_all, v_all = k_sw, v_sw

        n_kv = self.config.num_key_value_heads
        n_groups = self.config.num_attention_heads // n_kv
        scale_f = self.config.head_dim ** -0.5

        attn = torch.zeros(B, self.config.num_attention_heads, T, k_all.shape[-2],
                           device=q.device, dtype=torch.float32)
        for g in range(n_groups):
            q_g = q[:, g * n_kv:(g + 1) * n_kv]
            with torch.no_grad():
                scores = torch.matmul(q_g.float() * scale_f, k_all.float().transpose(-2, -1))
            attn[:, g * n_kv:(g + 1) * n_kv] = scores

        if attn_sink is not None and attn_sink.numel() == self.config.num_attention_heads:
            sw_len = k_sw.shape[-2]
            for h in range(self.config.num_attention_heads):
                attn[:, h, :, sw_len - 1:sw_len] = attn[:, h, :, sw_len - 1:sw_len] + attn_sink[h]

        attn_p = F.softmax(attn, dim=-1).to(v_all.dtype)
        out = torch.zeros(B, self.config.num_attention_heads, T, self.config.head_dim,
                          device=q.device, dtype=v_all.dtype)
        for g in range(n_groups):
            a_g = attn_p[:, g * n_kv:(g + 1) * n_kv]
            out[:, g * n_kv:(g + 1) * n_kv] = torch.matmul(a_g, v_all)

        # Inverse RoPE on attention output (remove rotation from V)
        out = apply_rotary_emb(out, freqs_cis, rd=self.config.qk_rope_head_dim, inverse=True)

        out = out.transpose(1, 2).contiguous()

        if wo_a is not None and wo_b is not None:
            out_g = out.view(B, T, self.config.o_groups, -1)
            wo_a_g = wo_a.view(self.config.o_groups, self.config.o_lora_rank, self.config.hidden_size)
            projected_parts = []
            for g in range(self.config.o_groups):
                part = torch.matmul(out_g[:, :, g, :].to(wo_a_g.dtype), wo_a_g[g].t())
                projected_parts.append(part)
            out_combined = torch.cat(projected_parts, dim=-1)
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

    def _load_expert_weights(self, layer_idx, eid):
        cache_key = f"{layer_idx}_{eid}"
        cached = self.expert_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._prefetch_enabled and self._prefetch_worker is not None:
            prefetched = self._prefetch_worker.get_cached(layer_idx, eid)
            if prefetched is not None:
                w1_d, w3_d, w2_d = prefetched
                pin = layer_idx < self.config.num_hash_layers or eid in self._hot_expert_ids
                self.expert_cache.put(cache_key, w1_d, w3_d, w2_d, pin=pin)
                return (w1_d, w3_d, w2_d)

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

        w1_d = load_fp8_weight(w1, s1) if w1.dtype != torch.int8 else load_fp4_weight(w1, s1)
        w3_d = load_fp8_weight(w3, s3) if w3.dtype != torch.int8 else load_fp4_weight(w3, s3)
        w2_d = load_fp8_weight(w2, s2) if w2.dtype != torch.int8 else load_fp4_weight(w2, s2)

        pin = layer_idx < self.config.num_hash_layers
        self.expert_cache.put(cache_key, w1_d, w3_d, w2_d, pin=pin)
        return (w1_d, w3_d, w2_d)

    def _cpu_ffn_fallback(self, layer_idx: int, eid: int):
        cache_key = f"cpu_{layer_idx}_{eid}"
        cached = self.expert_cache.get(cache_key)
        if cached is not None:
            return cached
        prefix = f"layers.{layer_idx}.ffn.experts.{eid}"
        keys = [f"{prefix}.w1.weight", f"{prefix}.w1.scale",
                f"{prefix}.w3.weight", f"{prefix}.w3.scale",
                f"{prefix}.w2.weight", f"{prefix}.w2.scale"]
        start = time.time()
        tensors = self.loader.get_weights(*keys)
        w1 = tensors.get(keys[0]); s1 = tensors.get(keys[1])
        w3 = tensors.get(keys[2]); s3 = tensors.get(keys[3])
        w2 = tensors.get(keys[4]); s2 = tensors.get(keys[5])
        if w1 is None:
            return None
        w1_d = load_fp8_weight(w1, s1) if w1.dtype != torch.int8 else load_fp4_weight(w1, s1)
        w3_d = load_fp8_weight(w3, s3) if w3.dtype != torch.int8 else load_fp4_weight(w3, s3)
        w2_d = load_fp8_weight(w2, s2) if w2.dtype != torch.int8 else load_fp4_weight(w2, s2)
        self.expert_cache.put(cache_key, w1_d, w3_d, w2_d, pin=False)
        elapsed = time.time() - start
        if elapsed > 0.1:
            self._log(f"CPU fallback: layer {layer_idx} expert {eid} loaded in {elapsed*1000:.0f}ms")
        return (w1_d, w3_d, w2_d)

    def _forward_single_expert(self, h, w1_d, w3_d, w2_d):
        gate_out = torch.matmul(h, w1_d.t())
        up_out = torch.matmul(h, w3_d.t())
        x = torch.cat([gate_out, up_out], dim=-1)
        x_2d = x.view(-1, x.shape[-1]).contiguous()
        activated = swiglu_forward(x_2d, swiglu_clamp_value=self.config.swiglu_limit)
        activated = activated.view(*x.shape[:-1], -1)
        out = torch.matmul(activated.to(w2_d.dtype), w2_d.t())
        return out

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
                expert_w = self._load_expert_weights(layer_idx, eid)
                if expert_w is None:
                    if self._cpu_fallback_enabled and layer_idx in self._cpu_fallback_layers:
                        expert_w = self._cpu_ffn_fallback(layer_idx, eid)
                    if expert_w is None:
                        continue
                w1_d, w3_d, w2_d = expert_w
                h_batch = flat_hidden[token_mask].to(w1_d.dtype)
                gate_out = torch.matmul(h_batch, w1_d.t())
                up_out = torch.matmul(h_batch, w3_d.t())
                x = torch.cat([gate_out, up_out], dim=-1)
                activated = swiglu_forward(x.contiguous(), swiglu_clamp_value=self.config.swiglu_limit)
                out = torch.matmul(activated.to(w2_d.dtype), w2_d.t())
                ffn_out.reshape(total_tokens, D)[token_mask] += out * weights[token_mask].unsqueeze(-1)

        shared_prefix = f"layers.{layer_idx}.ffn.shared_experts"
        shared_keys = [f"{shared_prefix}.w1.weight", f"{shared_prefix}.w1.scale",
                       f"{shared_prefix}.w3.weight", f"{shared_prefix}.w3.scale",
                       f"{shared_prefix}.w2.weight", f"{shared_prefix}.w2.scale"]
        shared_tensors = self.loader.get_weights(*shared_keys)
        shared_w1 = shared_tensors.get(shared_keys[0])
        if shared_w1 is not None:
            def _load_w(data, scale):
                if data is None:
                    return None
                if data.dtype == torch.int8:
                    return load_fp4_weight(data, scale)
                return load_fp8_weight(data, scale)

            s1 = shared_tensors.get(shared_keys[1])
            w3 = shared_tensors.get(shared_keys[2])
            s3 = shared_tensors.get(shared_keys[3])
            w2 = shared_tensors.get(shared_keys[4])
            s2 = shared_tensors.get(shared_keys[5])
            w1_d = _load_w(shared_w1, s1)
            w3_d = _load_w(w3, s3)
            w2_d = _load_w(w2, s2)

            if w3_d is not None and w2_d is not None:
                h = hidden_states.to(w1_d.dtype)
                shared_out = self._forward_single_expert(h, w1_d, w3_d, w2_d)
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

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.6):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        self._log(f"Generate: {T} prompt tokens, max_new={max_new_tokens}")

        self.layer_states = {}
        self._deq_cache.clear()
        for k in list(self.expert_cache.cache.keys()):
            if k not in self.expert_cache.pinned:
                del self.expert_cache.cache[k]
        if self._prefetch_worker is not None:
            self._prefetch_worker.clear()
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

            if layer_idx in self._cpu_fallback_layers:
                self._prefetch_worker.clear()

            if (layer_idx + 1) % 10 == 0:
                mem = torch.cuda.memory_allocated() / (1024**3)
                if mem > 19:
                    if self._prefetch_worker is not None:
                        self._prefetch_worker.clear()
                    for li in range(max(0, layer_idx - 3), layer_idx + 1):
                        if li in self.layer_states:
                            self.layer_states[li].kv_latent_cache = None
                            self.layer_states[li].archived_kv = None
                            self.layer_states[li].archived_len = 0
                            self.layer_states[li].compressed_kv = None
                    self.expert_cache.cache.clear()
                    torch.cuda.empty_cache()
                    self._log(f"  Layer {layer_idx}: freed caches, mem={mem:.1f}GB")

        h_3d = self._hc_head(h) if self.hc_head_fn is not None else h.sum(dim=2)
        if self.norm_weight is not None:
            h_3d = rms_norm(h_3d, self.norm_weight, self.config.rms_norm_eps)
        logits = torch.matmul(h_3d[:, -1:].to(self.lm_head.dtype), self.lm_head.t())

        if temperature > 0:
            probs = F.softmax(logits[:, -1].float() / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
        else:
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)

        generated = [next_id]
        for step in range(max_new_tokens - 1):
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

                if self._prefetch_enabled and self._prefetch_worker is not None and layer_idx + 1 < self.config.num_hidden_layers:
                    if len(used_experts) > 0:
                        self._prefetch_worker.prefetch_next_layer(layer_idx + 1, list(used_experts))

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
