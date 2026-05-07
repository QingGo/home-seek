"""L1 测试: torch.compile attention 输出正确性 (cosine similarity)."""

import torch
import pytest
from home_seek.model_config import DeepSeekV4FlashConfig
from home_seek.inference_engine.engine import HomeSeekInferenceEngine
from tests._engine_stub import make_engine


def _setup_attn_weights(eng: HomeSeekInferenceEngine, layer_idx: int, B=1, T=1):
    """Populate a layer's weight dict with real random tensors for attention testing."""
    cfg = eng.config
    D = cfg.hidden_size
    H = cfg.num_attention_heads
    hd = cfg.head_dim
    D_q = cfg.q_lora_rank
    d_qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

    lw = {}
    lw["attn.wq_a.weight"] = torch.randn(D_q, D, device=eng.device, dtype=torch.bfloat16)
    lw["attn.wq_b.weight"] = torch.randn(H * hd, D_q, device=eng.device, dtype=torch.bfloat16)
    lw["attn.wkv.weight"] = torch.randn(H * d_qk, D, device=eng.device, dtype=torch.bfloat16)
    lw["attn.wo_a.weight"] = torch.randn(cfg.o_groups * cfg.o_lora_rank, cfg.hidden_size,
                                          device=eng.device, dtype=torch.bfloat16)
    lw["attn.wo_b.weight"] = torch.randn(cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank,
                                          device=eng.device, dtype=torch.bfloat16)
    lw["attn.q_norm.weight"] = torch.randn(D_q, device=eng.device, dtype=torch.bfloat16)
    lw["attn.kv_norm.weight"] = torch.randn(H * d_qk, device=eng.device, dtype=torch.bfloat16)
    lw["attn.attn_sink"] = None
    return lw


class TestCompiledAttention:
    """Verify torch.compile attention output matches uncompiled (cosine ~1.0)."""

    def test_compiled_attn_cosine_similarity(self):
        """T=1 decode: compiled vs uncompiled attention output must match."""
        eng = make_engine(
            head_dim=128,
            num_attention_heads=64,
            num_key_value_heads=1,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            sliding_window=4096,
        )
        eng.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eng._use_gqa_fusion = True

        lw = _setup_attn_weights(eng, layer_idx=0)
        hidden = torch.randn(1, 1, eng.config.hidden_size, device=eng.device, dtype=torch.bfloat16)

        # Warm up the KV cache with a couple of tokens
        from home_seek.inference_engine.layer_state import LayerState
        eng.layer_states = {}
        eng._global_pos = 0

        # Pre-populate KV cache
        for pos in range(5):
            eng._global_pos = pos
            kv_dummy = torch.randn(1, 1, 8192, device=eng.device, dtype=torch.bfloat16)
            state = eng.layer_states.setdefault(0, LayerState(device=str(eng.device)))
            state.append_kv(kv_dummy)

        # Run uncompiled reference
        eng._global_pos = 5
        eng._compiled_attn_enabled = False
        torch.manual_seed(42)
        with torch.no_grad():
            ref_out = eng._forward_attn(hidden.clone(), lw, layer_idx=0)

        # Run compiled
        eng._compiled_attn_enabled = True
        torch.manual_seed(42)
        with torch.no_grad():
            # Restore KV cache
            eng.layer_states.clear()
            for pos in range(5):
                eng._global_pos = pos
                kv_dummy = torch.randn(1, 1, 8192, device=eng.device, dtype=torch.bfloat16)
                state = eng.layer_states.setdefault(0, LayerState(device=str(eng.device)))
                state.append_kv(kv_dummy)
            eng._global_pos = 5
            compiled_out = eng._forward_attn(hidden.clone(), lw, layer_idx=0)

        cos_sim = torch.nn.functional.cosine_similarity(
            ref_out.float().flatten(), compiled_out.float().flatten(), dim=0
        )
        assert cos_sim > 0.99, f"cosine={cos_sim.item():.6f} — compiled output diverges from uncompiled"

    def test_compiled_attn_t4_prefill_reference(self):
        """T=4 prefill: compiled should fall back to uncompiled path."""
        eng = make_engine(
            head_dim=128,
            num_attention_heads=64,
            num_key_value_heads=1,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            sliding_window=4096,
        )
        eng.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eng._use_gqa_fusion = True

        lw = _setup_attn_weights(eng, layer_idx=0)
        hidden = torch.randn(1, 4, eng.config.hidden_size, device=eng.device, dtype=torch.bfloat16)

        from home_seek.inference_engine.layer_state import LayerState
        eng.layer_states = {}
        eng._global_pos = 0

        eng._compiled_attn_enabled = True
        with torch.no_grad():
            out = eng._forward_attn(hidden, lw, layer_idx=0)

        assert out.shape == (1, 4, eng.config.hidden_size), f"unexpected shape: {out.shape}"
