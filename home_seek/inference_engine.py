import os
import json
import time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from collections import defaultdict, OrderedDict

from home_seek.model_config import DeepSeekV4FlashConfig
from home_seek.mhc import mhc_split_sinkhorn
from tile_reference import cast_back, unpack_from_e2m1fn_x2, swiglu_forward


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x_normed).to(x.dtype)


def load_fp8_weight(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if data.dtype == torch.bfloat16 or data.dtype == torch.float32:
        return data.to(torch.bfloat16)
    if data.dtype == torch.float8_e4m3fn:
        data_f32 = data.to(torch.float32)
        scale_f32 = scale.to(torch.float32)
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

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, w1, w3, w2):
        if len(self.cache) >= self.max_experts:
            self.cache.popitem(last=False)
        self.cache[key] = (w1, w3, w2)


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
    def __init__(self):
        self.kv_latent_cache = None
        self.compressed_kv = None
        self.compressed_count = 0


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
    def __init__(self, weight_dir: str = "weights", device: str = "cuda", verbose: bool = False):
        config_path = os.path.join(weight_dir, "config.json")
        self.config = DeepSeekV4FlashConfig(config_path)
        self.weight_dir = weight_dir
        self.device = torch.device(device)
        self.verbose = verbose
        self.loader = WeightLoader(weight_dir, device)
        self.expert_cache = ExpertWeightCache(max_experts=128)
        self.layer_states = {}
        self._load_global_weights()

    def _log(self, msg):
        if getattr(self, 'verbose', False):
            print(f"[inference] {msg}")

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

    def _deq(self, name, data, scale):
        if data is None:
            return None
        if scale is None:
            return data.to(torch.bfloat16)
        if data.dtype == torch.int8:
            return load_fp4_weight(data, scale)
        return load_fp8_weight(data, scale)

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

        if c_norm is not None:
            c_hidden = rms_norm(c_hidden, c_norm.to(torch.bfloat16))

        c_hidden = c_hidden.view(B, -1, compress_ratio, c_dim).mean(dim=2)

        if c_ape is not None:
            ape_len = min(c_ape.shape[0], c_hidden.shape[1])
            c_hidden[:, :ape_len, :] = c_hidden[:, :ape_len, :] + c_ape[:ape_len, :].to(c_hidden.dtype)

        if state.compressed_kv is None:
            state.compressed_kv = CompressedKVCache(compress_ratio, c_dim, str(self.device))
        state.compressed_kv.append(c_hidden.squeeze(0))
        return state

    def _get_compressed_attention_kv(self, state: LayerState, lw: dict):
        compressed_kv = state.compressed_kv.get() if state.compressed_kv is not None else None
        if compressed_kv is None or compressed_kv.shape[0] == 0:
            return None
        c_dim = compressed_kv.shape[-1]
        if c_dim != self.config.head_dim:
            compressed_kv_proj = torch.zeros(compressed_kv.shape[0], self.config.head_dim,
                                              device=compressed_kv.device, dtype=compressed_kv.dtype)
            min_dim = min(c_dim, self.config.head_dim)
            compressed_kv_proj[:, :min_dim] = compressed_kv[:, :min_dim]
            compressed_kv = compressed_kv_proj
        compressed_kv = compressed_kv.unsqueeze(0).unsqueeze(2)
        return compressed_kv

    def _expand_kv(self, kv_latent):
        B, T, _ = kv_latent.shape
        k = kv_latent.unsqueeze(2).transpose(1, 2)
        return k, k

    def _forward_mhc(self, hidden: torch.Tensor, hc_base: torch.Tensor,
                     hc_fn: torch.Tensor, hc_scale: torch.Tensor, apply_pre: bool = True):
        B, T, D = hidden.shape
        hc_mult = self.config.hc_mult
        fn_in_features = hc_fn.shape[-1]
        expected_in = D * hc_mult
        if fn_in_features != expected_in:
            self._log(f"mHC skip: hc_fn needs {fn_in_features}-d input, hidden is {D}-d "
                      f"(expected expanded to {expected_in})")
            return hidden, None, None
        hidden_expanded = hidden.unsqueeze(2).expand(-1, -1, hc_mult, -1)
        hidden_flat = hidden_expanded.reshape(B, T, expected_in)
        mixes = torch.matmul(hidden_flat.to(hc_fn.dtype), hc_fn.t())
        mixes = mixes.float()
        pre, post, comb = mhc_split_sinkhorn(
            mixes, hc_scale.to(torch.float32), hc_base.to(torch.float32),
            hc_mult=hc_mult, sinkhorn_iters=self.config.hc_sinkhorn_iters, eps=self.config.hc_eps,
        )
        if apply_pre:
            hidden_g = hidden.view(B, T, hc_mult, D // hc_mult)
            scaled = hidden_g * pre.unsqueeze(-1)
            hidden = scaled.view(B, T, D)
        return hidden, post, comb

    def _forward_attn(self, hidden_states, lw, layer_idx):
        B, T, D = hidden_states.shape

        wq_a = self._deq("wq_a", lw.get("attn.wq_a.weight"), lw.get("attn.wq_a.scale"))
        wq_b = self._deq("wq_b", lw.get("attn.wq_b.weight"), lw.get("attn.wq_b.scale"))
        wkv = self._deq("wkv", lw.get("attn.wkv.weight"), lw.get("attn.wkv.scale"))
        wo_a = self._deq("wo_a", lw.get("attn.wo_a.weight"), lw.get("attn.wo_a.scale"))
        wo_b = self._deq("wo_b", lw.get("attn.wo_b.weight"), lw.get("attn.wo_b.scale"))
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

        kv_latent = torch.matmul(hidden_states.to(wkv.dtype), wkv.t())
        if kv_norm is not None:
            kv_latent = rms_norm(kv_latent, kv_norm)

        state = self.layer_states.setdefault(layer_idx, LayerState())

        if state.kv_latent_cache is not None:
            all_kv_latent = torch.cat([state.kv_latent_cache, kv_latent], dim=1)
        else:
            all_kv_latent = kv_latent
        state.kv_latent_cache = all_kv_latent

        state = self._compress_kv(hidden_states, lw, layer_idx, state)

        T_kv = all_kv_latent.shape[1]
        sw = min(self.config.sliding_window, T_kv)
        kv_sw = all_kv_latent[:, -sw:, :]
        k_sw, v_sw = self._expand_kv(kv_sw)

        total_kv_len = k_sw.shape[-2]
        compressed_kv = self._get_compressed_attention_kv(state, lw)
        if compressed_kv is not None:
            total_kv_len += compressed_kv.shape[-2]

        attn = torch.zeros(B, self.config.num_attention_heads, T, total_kv_len, device=q.device, dtype=torch.float32)
        scale = self.config.head_dim ** -0.5
        num_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        sw_len = k_sw.shape[-2]
        for g in range(num_groups):
            q_g = q[:, g * self.config.num_key_value_heads:(g + 1) * self.config.num_key_value_heads]
            scores_sw = torch.matmul(q_g.float() * scale, k_sw.float().transpose(-2, -1))
            if compressed_kv is not None:
                scores_c = torch.matmul(q_g.float() * scale, compressed_kv.float().transpose(-2, -1))
                scores = torch.cat([scores_c, scores_sw], dim=-1)
            else:
                scores = scores_sw
            attn[:, g * self.config.num_key_value_heads:(g + 1) * self.config.num_key_value_heads] = scores

        if attn_sink is not None and attn_sink.numel() == self.config.num_attention_heads:
            for h in range(self.config.num_attention_heads):
                attn[:, h, :, -1:] = attn[:, h, :, -1:] + attn_sink[h]

        attn_p = F.softmax(attn, dim=-1).to(v_sw.dtype)
        out = torch.zeros(B, self.config.num_attention_heads, T, self.config.head_dim, device=q.device, dtype=v_sw.dtype)
        for g in range(num_groups):
            a_g = attn_p[:, g * self.config.num_key_value_heads:(g + 1) * self.config.num_key_value_heads]
            if compressed_kv is not None:
                c_len = compressed_kv.shape[-2]
                a_c = a_g[:, :, :, :c_len]
                a_sw = a_g[:, :, :, c_len:]
                v_c = compressed_kv.expand(-1, a_g.shape[1], -1, -1)
                v_part = torch.matmul(a_c, v_c) + torch.matmul(a_sw, v_sw)
            else:
                v_part = torch.matmul(a_g, v_sw)
            out[:, g * self.config.num_key_value_heads:(g + 1) * self.config.num_key_value_heads] = v_part
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

        w1 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w1.weight")
        s1 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w1.scale")
        w3 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w3.weight")
        s3 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w3.scale")
        w2 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w2.weight")
        s2 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w2.scale")

        if w1 is None:
            return None

        w1_d = load_fp4_weight(w1, s1) if w1.dtype == torch.int8 else load_fp8_weight(w1, s1)
        w3_d = load_fp4_weight(w3, s3) if w3 is not None and w3.dtype == torch.int8 else load_fp8_weight(w3, s3) if w3 is not None else w1_d
        w2_d = load_fp8_weight(w2, s2) if w2 is not None else w1_d

        self.expert_cache.put(cache_key, w1_d, w3_d, w2_d)
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
        gate_w = self._deq("ffn.gate", lw.get("ffn.gate.weight"), lw.get("ffn.gate.scale"))
        gate_bias = lw.get("ffn.gate.bias")
        tid2eid = lw.get("ffn.gate.tid2eid")

        if gate_w is None:
            return torch.zeros_like(hidden_states)

        if layer_idx < self.config.num_hash_layers and tid2eid is not None and input_ids is not None:
            topk_idx, topk_w = self._compute_hash_experts(input_ids, layer_idx, tid2eid)
        else:
            topk_idx, topk_w = self._compute_routing_experts(hidden_states, gate_w, gate_bias)

        ffn_out = torch.zeros_like(hidden_states)
        B, T, D = hidden_states.shape
        for b in range(B):
            for t in range(T):
                for k in range(self.config.num_experts_per_tok):
                    eid = topk_idx[b, t, k].item()
                    wgt = topk_w[b, t, k].item()
                    if eid < 0:
                        continue

                    expert_w = self._load_expert_weights(layer_idx, eid)
                    if expert_w is None:
                        continue
                    w1_d, w3_d, w2_d = expert_w

                    h = hidden_states[b:b+1, t:t+1].to(w1_d.dtype)
                    out = self._forward_single_expert(h, w1_d, w3_d, w2_d)
                    ffn_out[b:b+1, t:t+1] += out * wgt

        shared_w1 = self.loader.get_weight(f"layers.{layer_idx}.ffn.shared_experts.w1.weight")
        if shared_w1 is not None:
            s1 = self.loader.get_weight(f"layers.{layer_idx}.ffn.shared_experts.w1.scale")
            w3 = self.loader.get_weight(f"layers.{layer_idx}.ffn.shared_experts.w3.weight")
            s3 = self.loader.get_weight(f"layers.{layer_idx}.ffn.shared_experts.w3.scale")
            w2 = self.loader.get_weight(f"layers.{layer_idx}.ffn.shared_experts.w2.weight")
            s2 = self.loader.get_weight(f"layers.{layer_idx}.ffn.shared_experts.w2.scale")

            w1_d = load_fp8_weight(shared_w1, s1)
            w3_d = load_fp8_weight(w3, s3) if w3 is not None else None
            w2_d = load_fp8_weight(w2, s2) if w2 is not None else None

            if w3_d is not None and w2_d is not None:
                h = hidden_states.to(w1_d.dtype)
                shared_out = self._forward_single_expert(h, w1_d, w3_d, w2_d)
                ffn_out = ffn_out + shared_out

        return ffn_out

    def _process_mhc_layer(self, hidden, lw, prefix: str):
        hc_base = lw.get(f"{prefix}_base")
        hc_fn = lw.get(f"{prefix}_fn")
        hc_scale = lw.get(f"{prefix}_scale")
        if hc_base is None or hc_fn is None or hc_scale is None:
            return hidden, None, None
        return self._forward_mhc(hidden, hc_base.to(torch.bfloat16),
                                  hc_fn.to(torch.bfloat16), hc_scale.to(torch.bfloat16))

    def _process_mhc_post(self, hidden, residual, post, comb):
        if post is None or comb is None:
            return hidden
        B, S, D = hidden.shape
        hc_mult = comb.shape[-1]
        hidden_g = hidden.view(B, S, hc_mult, D // hc_mult)
        mixed = torch.matmul(comb.transpose(-2, -1).float(), hidden_g.float())
        result = mixed * post.unsqueeze(-1)
        return result.to(hidden.dtype).view(B, S, D)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.0):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        self._log(f"Generate: {T} prompt tokens, max_new={max_new_tokens}")

        self.layer_states = {}
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.time()

        h = self.embed[input_ids].to(torch.bfloat16)
        for layer_idx in range(self.config.num_hidden_layers):
            lw = self._get_layer_weights(layer_idx)
            if lw.get("attn_norm.weight") is not None:
                h = rms_norm(h, lw["attn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)

            h_mhc, _, _ = self._process_mhc_layer(h, lw, "hc_attn")
            attn_out = self._forward_attn(h if h_mhc is None else h_mhc, lw, layer_idx)
            h = h + attn_out

            if lw.get("ffn_norm.weight") is not None:
                h = rms_norm(h, lw["ffn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)

            residual = h
            h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_ffn")
            h_ffn_in = h_pre if h_pre is not None else h

            ffn_out = self._forward_ffn(h_ffn_in, lw, layer_idx, input_ids)
            if post is not None and comb is not None:
                ffn_out = self._process_mhc_post(ffn_out, residual, post, comb)
            h = h + ffn_out

            if (layer_idx + 1) % 10 == 0:
                mem = torch.cuda.memory_allocated() / (1024**3)
                if mem > 21:
                    for li in range(max(0, layer_idx - 3), layer_idx + 1):
                        if li in self.layer_states:
                            self.layer_states[li].kv_latent_cache = None
                            self.layer_states[li].compressed_kv = None
                    torch.cuda.empty_cache()
                    self._log(f"  Layer {layer_idx}: freed KV cache, mem={mem:.1f}GB")

        if self.norm_weight is not None:
            h = rms_norm(h, self.norm_weight, self.config.rms_norm_eps)
        logits = torch.matmul(h[:, -1:].to(self.lm_head.dtype), self.lm_head.t())

        if temperature > 0:
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).unsqueeze(0)
        else:
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)

        generated = [next_id]
        for step in range(max_new_tokens - 1):
            h = self.embed[next_id].to(torch.bfloat16)
            for layer_idx in range(self.config.num_hidden_layers):
                lw = self._get_layer_weights(layer_idx)
                st = self.layer_states.get(layer_idx)
                if st is not None and st.kv_latent_cache is not None and st.kv_latent_cache.shape[1] > 2000000:
                    keep = self.config.sliding_window + 128
                    st.kv_latent_cache = st.kv_latent_cache[:, -keep:, :]
                    if st.compressed_kv is not None:
                        st.compressed_kv = None
                    self._log(f"  Layer {layer_idx}: truncated KV latent to {keep}")

                if lw.get("attn_norm.weight") is not None:
                    h = rms_norm(h, lw["attn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)

                h_mhc, _, _ = self._process_mhc_layer(h, lw, "hc_attn")
                attn_out = self._forward_attn(h if h_mhc is None else h_mhc, lw, layer_idx)
                h = h + attn_out

                if lw.get("ffn_norm.weight") is not None:
                    h = rms_norm(h, lw["ffn_norm.weight"].to(torch.bfloat16), self.config.rms_norm_eps)

                residual = h
                h_pre, post, comb = self._process_mhc_layer(h, lw, "hc_ffn")
                h_ffn_in = h_pre if h_pre is not None else h

                ffn_out = self._forward_ffn(h_ffn_in, lw, layer_idx, next_id)
                if post is not None and comb is not None:
                    ffn_out = self._process_mhc_post(ffn_out, residual, post, comb)
                h = h + ffn_out

            if self.norm_weight is not None:
                h = rms_norm(h, self.norm_weight, self.config.rms_norm_eps)
            logits = torch.matmul(h.to(self.lm_head.dtype), self.lm_head.t())

            if temperature > 0:
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).unsqueeze(0)
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
    args = parser.parse_args()
    engine = HomeSeekInferenceEngine(args.weight_dir, verbose=args.verbose)

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    except Exception:
        pass
    if tokenizer is not None:
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(engine.device)
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
