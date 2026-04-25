import json


class DeepSeekV4FlashConfig:
    def __init__(self, config_path: str = None):
        if config_path:
            with open(config_path) as f:
                cfg = json.load(f)
            self._load_from_dict(cfg)
        else:
            self._load_from_dict({})

    def _load_from_dict(self, cfg: dict):
        self.num_hidden_layers = cfg.get("num_hidden_layers", 43)
        self.hidden_size = cfg.get("hidden_size", 4096)
        self.moe_intermediate_size = cfg.get("moe_intermediate_size", 2048)
        self.num_attention_heads = cfg.get("num_attention_heads", 64)
        self.num_key_value_heads = cfg.get("num_key_value_heads", 1)
        self.head_dim = cfg.get("head_dim", 512)
        self.q_lora_rank = cfg.get("q_lora_rank", 1024)
        self.o_lora_rank = cfg.get("o_lora_rank", 1024)
        self.o_groups = cfg.get("o_groups", 8)
        self.index_head_dim = cfg.get("index_head_dim", 128)
        self.index_n_heads = cfg.get("index_n_heads", 64)
        self.index_topk = cfg.get("index_topk", 512)
        self.vocab_size = cfg.get("vocab_size", 129280)
        self.max_position_embeddings = cfg.get("max_position_embeddings", 1048576)
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)
        self.sliding_window = cfg.get("sliding_window", 128)
        self.swiglu_limit = cfg.get("swiglu_limit", 10.0)

        self.n_routed_experts = cfg.get("n_routed_experts", 256)
        self.n_shared_experts = cfg.get("n_shared_experts", 1)
        self.num_experts_per_tok = cfg.get("num_experts_per_tok", 6)
        self.routed_scaling_factor = cfg.get("routed_scaling_factor", 1.5)
        self.scoring_func = cfg.get("scoring_func", "sqrtsoftplus")
        self.norm_topk_prob = cfg.get("norm_topk_prob", True)
        self.topk_method = cfg.get("topk_method", "noaux_tc")
        self.num_hash_layers = cfg.get("num_hash_layers", 3)
        self.shared_expert_intermediate_size = cfg.get("shared_expert_intermediate_size", 2048)
        self.moe_intermediate_size = cfg.get("moe_intermediate_size", 2048)

        self.compress_ratios = cfg.get("compress_ratios", None)
        if not self.compress_ratios:
            ratios = []
            for i in range(self.num_hidden_layers):
                if i < 2:
                    ratios.append(0)
                elif i == self.num_hidden_layers - 1:
                    ratios.append(0)
                else:
                    ratios.append(4 if (i - 2) % 2 == 0 else 128)
            self.compress_ratios = ratios

        self.hc_mult = cfg.get("hc_mult", 4)
        self.hc_sinkhorn_iters = cfg.get("hc_sinkhorn_iters", 20)
        self.hc_eps = cfg.get("hc_eps", 1e-6)

        self.num_nextn_predict_layers = cfg.get("num_nextn_predict_layers", 1)
        self.tie_word_embeddings = cfg.get("tie_word_embeddings", False)

        self.kv_lora_rank = cfg.get("kv_lora_rank", 512)
        self.v_head_dim = cfg.get("v_head_dim", self.head_dim)
        self.qk_rope_head_dim = cfg.get("qk_rope_head_dim", 64)
        self.qk_nope_head_dim = cfg.get("qk_nope_head_dim", self.index_head_dim)

        self.num_hash_experts = self.num_hash_layers * self.num_experts_per_tok
        self.num_hot_experts = 16
        self.num_swa_layers = sum(1 for r in self.compress_ratios if r == 0)

    def get_compress_ratio(self, layer_idx: int) -> int:
        return self.compress_ratios[layer_idx] if layer_idx < len(self.compress_ratios) else 0
