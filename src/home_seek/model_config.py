from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepSeekV4FlashConfig:
    num_hidden_layers: int = 43
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512
    vocab_size: int = 129280
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    sliding_window: int = 128
    swiglu_limit: float = 10.0
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    topk_method: str = "noaux_tc"
    num_hash_layers: int = 3
    shared_expert_intermediate_size: int = 2048
    compress_ratios: Optional[list[int]] = None
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False
    kv_lora_rank: int = 512
    v_head_dim: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling_factor: float = 16.0
    rope_scaling_beta_fast: int = 32
    rope_scaling_beta_slow: int = 1
    rope_scaling_original_max_position_embeddings: int = 65536
    num_hash_experts: int = 18
    num_hot_experts: int = 16
    num_swa_layers: int = 2

    def __post_init__(self):
        if self.compress_ratios is None:
            ratios = []
            for i in range(self.num_hidden_layers):
                if i < 2:
                    ratios.append(0)
                elif i == self.num_hidden_layers - 1:
                    ratios.append(0)
                else:
                    ratios.append(4 if (i - 2) % 2 == 0 else 128)
            self.compress_ratios = ratios
            self.num_swa_layers = sum(1 for r in ratios if r == 0)

    @classmethod
    def from_json(cls, config_path: str) -> DeepSeekV4FlashConfig:
        with open(config_path) as f:
            cfg = json.load(f)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict) -> DeepSeekV4FlashConfig:
        head_dim = cfg.get("head_dim", 512)
        index_head_dim = cfg.get("index_head_dim", 128)
        num_layers = cfg.get("num_hidden_layers", 43)
        num_experts_per_tok = cfg.get("num_experts_per_tok", 6)
        rope_scaling = cfg.get("rope_scaling", {})

        compress_ratios = cfg.get("compress_ratios")
        if compress_ratios is None:
            compress_ratios = []
            for i in range(num_layers):
                if i < 2:
                    compress_ratios.append(0)
                elif i == num_layers - 1:
                    compress_ratios.append(0)
                else:
                    compress_ratios.append(4 if (i - 2) % 2 == 0 else 128)

        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in ("compress_ratios", "num_hash_experts", "num_swa_layers",
                              "v_head_dim", "qk_nope_head_dim",
                              "rope_scaling_factor", "rope_scaling_beta_fast",
                              "rope_scaling_beta_slow",
                              "rope_scaling_original_max_position_embeddings"):
                continue
            if field_name in cfg:
                kwargs[field_name] = cfg[field_name]

        return cls(
            compress_ratios=compress_ratios,
            v_head_dim=cfg.get("v_head_dim", head_dim),
            qk_nope_head_dim=cfg.get("qk_nope_head_dim", index_head_dim),
            rope_scaling_factor=rope_scaling.get("factor", 16.0),
            rope_scaling_beta_fast=rope_scaling.get("beta_fast", 32),
            rope_scaling_beta_slow=rope_scaling.get("beta_slow", 1),
            rope_scaling_original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 65536),
            num_hash_experts=num_layers * num_experts_per_tok,
            num_swa_layers=sum(1 for r in compress_ratios if r == 0),
            **kwargs,
        )

    def get_compress_ratio(self, layer_idx: int) -> int:
        ratios = self.compress_ratios
        if ratios is None:
            return 0
        return ratios[layer_idx] if layer_idx < len(ratios) else 0
