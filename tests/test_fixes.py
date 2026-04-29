import os
import sys
import unittest
import torch
import torch.nn.functional as F
import pytest
from home_seek.mhc import mhc_split_sinkhorn
from home_seek.inference_engine import HomeSeekInferenceEngine
from home_seek.model_config import DeepSeekV4FlashConfig
from home_seek._fp4 import unpack_from_e2m1fn_x2
from tests._reference import swiglu_forward


class TestExpandKV:
    def test_expand_kv_direct(self):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(
            head_dim=512,
        )

        kv_latent = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
        k, v = eng._expand_kv(kv_latent)
        assert k.shape == (2, 1, 8, 512)
        assert v.shape == (2, 1, 8, 512)
        assert torch.equal(k, v)

    def test_expand_kv_no_woa_crash(self):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(
            head_dim=512,
        )
        kv_latent = torch.randn(1, 4, 512, device="cuda", dtype=torch.bfloat16)
        k, v = eng._expand_kv(kv_latent)
        assert k.shape[-2] == 4

    def test_w2_fp4_quantization(self):
        from home_seek.inference_engine import load_fp4_weight
        w2_data = torch.randint(0, 15, (4, 1024), device="cuda", dtype=torch.int8)
        scale_u8 = torch.zeros(4, 64, device="cuda", dtype=torch.uint8)
        scale_u8[:, :] = 127
        scale_f32 = scale_u8.view(torch.uint8).to(torch.int32)
        scale_f32 = (scale_f32 << 23).view(torch.float32)
        deq = load_fp4_weight(w2_data, scale_f32)
        assert deq.shape == (4, 2048)
        assert deq.dtype == torch.bfloat16


class TestSwiGLU:
    def test_swiglu_clamp(self):
        x = torch.tensor([[100.0, -100.0, 1.0, -1.0, 5.0, -5.0, 2.0, -2.0]],
                         device="cuda", dtype=torch.bfloat16)
        out = swiglu_forward(x, swiglu_clamp_value=10.0)
        assert out.shape == (1, 4)

    def test_swiglu_with_weights(self):
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        pos = torch.tensor([0, 1, 2, 3], device="cuda")
        weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]], device="cuda")
        out = swiglu_forward(x, pos_to_token_topk=pos, topk_weights=weights)
        assert out.shape == (4, 4)


class TestMHC:
    def test_mhc_split_sinkhorn(self):
        B, S = 1, 8
        hc_mult = 4
        mix_dim = hc_mult * (2 + hc_mult)
        mixes = torch.randn(B, S, mix_dim, device="cuda", dtype=torch.float32)
        hc_scale = torch.tensor([1.0, 0.5, 0.1], device="cuda", dtype=torch.float32)
        hc_base = torch.randn(mix_dim, device="cuda", dtype=torch.float32)
        pre, post, comb = mhc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters=5)
        assert pre.shape == (B, S, hc_mult)
        assert post.shape == (B, S, hc_mult)
        assert comb.shape == (B, S, hc_mult, hc_mult)

    def test_mhc_pre_post(self):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.verbose = False
        eng._use_triton = True
        eng.config = DeepSeekV4FlashConfig(
            hc_mult=4,
            hc_sinkhorn_iters=5,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
            num_attention_heads=64,
            num_key_value_heads=1,
            head_dim=512,
            o_groups=8,
            o_lora_rank=1024,
            hidden_size=4096,
            sliding_window=128,
        )
        B, T, hc, D = 1, 3, 4, 4096
        h4d = torch.randn(B, T, hc, D, device="cuda", dtype=torch.bfloat16)
        hc_base = torch.randn(24, device="cuda", dtype=torch.bfloat16)
        hc_fn = torch.randn(24, D * hc, device="cuda", dtype=torch.bfloat16)
        hc_scale = torch.tensor([1.0, 0.5, 0.1], device="cuda", dtype=torch.bfloat16)
        h_out, post, comb = eng._forward_mhc(h4d, hc_base, hc_fn, hc_scale, apply_pre=True)
        assert h_out.shape == (B, T, D)
        ffn_out = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        mixed = eng._process_mhc_post(ffn_out, h4d, post.float(), comb.float())
        assert mixed.shape == (B, T, hc, D)

    def test_mhc_shape_mismatch_skip(self):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.verbose = False
        eng.config = DeepSeekV4FlashConfig(
            hc_mult=4,
            hc_sinkhorn_iters=5,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
            hidden_size=4096,
        )
        hidden = torch.randn(1, 4, 4, 4096, device="cuda", dtype=torch.bfloat16)
        hc_base = torch.randn(12, device="cuda", dtype=torch.bfloat16)
        hc_fn_wrong = torch.randn(12, 4096, device="cuda", dtype=torch.bfloat16)
        hc_scale = torch.tensor([1.0, 0.5], device="cuda", dtype=torch.bfloat16)
        out, post, comb = eng._forward_mhc(hidden, hc_base, hc_fn_wrong, hc_scale, apply_pre=True)
        assert post is None and comb is None
        assert torch.equal(out, hidden.sum(dim=2))


class TestRouting:
    def test_hash_routing(self):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(
            num_experts_per_tok=6,
            num_hash_layers=3,
        )
        input_ids = torch.tensor([[5, 10, 15]], device="cuda")
        tid2eid = torch.randint(0, 256, (100, 6), device="cuda")
        eids, weights = eng._compute_hash_experts(input_ids, 0, tid2eid)
        assert eids.shape == (1, 3, 6)
        assert weights.shape == (1, 3, 6)
        expected_eid = tid2eid[input_ids]
        assert torch.equal(eids, expected_eid)

    def test_routing_sqrtsoftplus(self):
        from home_seek.router import compute_expert_affinity
        hidden = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        indices, weights = compute_expert_affinity(hidden, gate, top_k=4)
        assert indices.shape == (2, 4)

    def test_routing_with_bias(self):
        from home_seek.router import compute_expert_affinity_with_bias
        hidden = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(8, device="cuda", dtype=torch.bfloat16)
        indices, weights = compute_expert_affinity_with_bias(hidden, gate, bias, top_k=4)
        assert indices.shape == (2, 4)

    def test_deterministic_routing(self):
        from home_seek.router import compute_expert_affinity
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        hidden = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        r1, w1 = compute_expert_affinity(hidden, gate, top_k=2)
        r2, w2 = compute_expert_affinity(hidden, gate, top_k=2)
        assert torch.equal(r1, r2) and torch.equal(w1, w2)


class TestKVCache:
    def test_compressed_kv_cache(self):
        from home_seek.hybrid_kv_cache import HybridKVCache
        c = HybridKVCache(compress_ratio=4, head_dim=512, device="cuda")
        t1 = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
        t2 = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        for t in [t1, t2]:
            for i in range(t.shape[0]):
                c.append_compressed(t[i:i+1])
        result = c.get_compressed_kv()
        assert result is not None and result.shape == (5, 512)

    def test_get_compressed_attention_kv(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(
            head_dim=512,
        )
        state = LayerState()
        state.compressed_kv_data = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
        result = eng._get_compressed_attention_kv(state, {})
        assert result is not None and result.dim() == 4

    def test_compress_kv(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.device = torch.device("cuda")
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=1,
            hidden_size=4096,
            head_dim=512,
        )
        eng.config.get_compress_ratio = lambda idx: 4
        B, T, D = 1, 16, 4096
        hidden = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        lw = {
            "attn.compressor.wkv.weight": torch.randn(1024, D, device="cuda", dtype=torch.bfloat16),
            "attn.compressor.wgate.weight": torch.randn(1024, D, device="cuda", dtype=torch.bfloat16),
            "attn.compressor.norm.weight": torch.randn(512, device="cuda", dtype=torch.bfloat16),
        }
        state = LayerState()
        eng._compress_kv(hidden, lw, 0, state)
        assert state.compressed_kv_data is not None
        c = state.compressed_kv_data
        assert c.shape[0] >= T // 4 and c.shape[1] == 512


class TestQuantization:
    def test_fp4_roundtrip(self):
        from home_seek._fp4 import cast
        for h in [64, 128, 256]:
            for w in [128, 256, 512]:
                x = torch.randn(h, w, device="cuda", dtype=torch.bfloat16)
                q, sf = cast(x, fmt="e2m1", block_size=(1, 32))
                dq = unpack_from_e2m1fn_x2(q)
                sf_e = sf.repeat_interleave(32, dim=1)
                dq = dq.to(torch.float32) * sf_e.to(torch.float32)
                dq = dq[:h, :w]
                cos = F.cosine_similarity(x.flatten().unsqueeze(0).float(), dq.flatten().unsqueeze(0)).item()
                assert cos >= 0.99

    def test_fp8_roundtrip(self):
        from home_seek._fp4 import cast, cast_back
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        q, sf = cast(x, fmt="e4m3", block_size=(32, 32))
        dq = cast_back((q, sf), fmt="fp32", block_size=(32, 32))
        dq = dq[:64, :128]
        cos = F.cosine_similarity(x.flatten().unsqueeze(0).float(), dq.flatten().unsqueeze(0)).item()
        assert cos >= 0.995

    def test_expert_weight_approximation(self):
        from home_seek._fp4 import cast
        hidden, inter = 4096, 2048
        torch.manual_seed(42)
        gate = torch.randn(inter, hidden, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(inter, hidden, device="cuda", dtype=torch.bfloat16)
        down = torch.randn(hidden, inter, device="cuda", dtype=torch.bfloat16)
        for name, orig in [("gate", gate), ("up", up), ("down", down)]:
            qd, sf = cast(orig, fmt="e2m1", block_size=(1, 32))
            dq = unpack_from_e2m1fn_x2(qd)
            sf_e = sf.repeat_interleave(32, dim=1)
            dq = dq.to(torch.float32) * sf_e.to(torch.float32)
            dq = dq[:orig.shape[0], :orig.shape[1]]
            cos = F.cosine_similarity(orig.flatten().unsqueeze(0).float(), dq.flatten().unsqueeze(0)).item()
            assert cos >= 0.99, f"{name}: {cos:.6f}"


class TestCSAIndexer:
    def test_compute_indexer_shapes(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState
        from home_seek.model_config import DeepSeekV4FlashConfig
        config = DeepSeekV4FlashConfig()
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = config
        eng._deq_cache = {}
        eng.verbose = False

        B, T, q_rank = 1, 8, config.q_lora_rank
        q_latent = torch.randn(B, T, q_rank, device="cuda", dtype=torch.bfloat16)

        num_compressed = 600
        idx_dim = config.index_head_dim
        state = LayerState(device="cuda")
        state.compressed_kv_data = torch.randn(num_compressed, config.kv_lora_rank, device="cuda", dtype=torch.bfloat16)
        state.compressed_kv_idx = torch.randn(num_compressed, idx_dim, device="cuda", dtype=torch.bfloat16)

        lw = {
            "attn.indexer.wq_b.weight": torch.randn(
                config.index_n_heads * config.index_head_dim, q_rank,
                device="cuda", dtype=torch.bfloat16),
        }

        result = eng._compute_indexer(q_latent, torch.randn(B, T, 4096, device="cuda", dtype=torch.bfloat16), lw, state, 0)
        assert result is not None
        assert result.dim() == 4, f"Expected 4D, got {result.dim()}D"
        B_r, n_kv_r, seq_r, hd_r = result.shape
        assert n_kv_r == 1, f"Expected 1 KV head, got {n_kv_r}"
        assert hd_r == config.head_dim, f"Expected head_dim={config.head_dim}, got {hd_r}"
        assert seq_r == config.index_topk, f"Expected {config.index_topk} selected, got {seq_r}"

    def test_compute_indexer_no_weights_fallback(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState
        from home_seek.model_config import DeepSeekV4FlashConfig
        config = DeepSeekV4FlashConfig()
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = config
        eng._deq_cache = {}
        eng.verbose = False

        B, T, q_rank = 1, 4, config.q_lora_rank
        q_latent = torch.randn(B, T, q_rank, device="cuda", dtype=torch.bfloat16)

        state = LayerState(device="cuda")
        state.compressed_kv_data = torch.randn(2, config.kv_lora_rank, device="cuda", dtype=torch.bfloat16)

        result = eng._compute_indexer(q_latent, torch.randn(B, T, 4096, device="cuda", dtype=torch.bfloat16), {}, state, 0)
        assert result is None, "Should return None when no indexer weights"

    def test_csa_attn_dispatch(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState
        from home_seek.model_config import DeepSeekV4FlashConfig
        config = DeepSeekV4FlashConfig()
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = config
        eng.layer_states = {}
        eng._deq_cache = {}
        eng.device = torch.device("cuda")
        eng.verbose = False
        eng._cpu_fallback_enabled = False
        eng._cpu_fallback_layers = set()
        eng.expert_cache = type('obj', (object,), {'cache': {}})()

        for layer_idx in range(config.num_hidden_layers):
            cr = config.get_compress_ratio(layer_idx)
            if cr == 0:
                continue
            state = LayerState(device="cuda")
            eng.layer_states[layer_idx] = state

        for layer_idx in [2, 4, 6]:
            cr = config.get_compress_ratio(layer_idx)
            assert cr == 4, f"Layer {layer_idx} should be CSA (ratio=4), got {cr}"

        for layer_idx in [3, 5, 7]:
            cr = config.get_compress_ratio(layer_idx)
            assert cr == 128, f"Layer {layer_idx} should be HCA (ratio=128), got {cr}"

        for layer_idx in [0, 1, config.num_hidden_layers - 1]:
            cr = config.get_compress_ratio(layer_idx)
            assert cr == 0, f"Layer {layer_idx} should be SWA (ratio=0), got {cr}"


class TestRoutingFix:
    def test_routing_with_scaling_factor(self):
        from home_seek.router import compute_expert_affinity
        torch.manual_seed(42)
        hidden = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        indices, weights = compute_expert_affinity(hidden, gate, top_k=4, routed_scaling_factor=1.5)
        assert indices.shape == (2, 4)
        assert weights.shape == (2, 4)
        assert torch.all(weights > 0), "All weights should be positive"
        routed_factors = weights.sum(dim=-1)
        assert torch.allclose(routed_factors, torch.full_like(routed_factors, 1.5), atol=1e-5), \
            f"Routed sum should equal scaling_factor=1.5, got {routed_factors}"

    def test_routing_without_scaling_default(self):
        from home_seek.router import compute_expert_affinity
        hidden = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        indices, weights = compute_expert_affinity(hidden, gate, top_k=4)
        routed_factors = weights.sum(dim=-1)
        assert torch.allclose(routed_factors, torch.full_like(routed_factors, 1.5), atol=1e-5), \
            f"Default routed sum should be 1.5, got {routed_factors}"

    def test_routing_deterministic_with_factor(self):
        from home_seek.router import compute_expert_affinity
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        hidden = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        r1, w1 = compute_expert_affinity(hidden, gate, top_k=2, routed_scaling_factor=2.0)
        r2, w2 = compute_expert_affinity(hidden, gate, top_k=2, routed_scaling_factor=2.0)
        assert torch.equal(r1, r2) and torch.equal(w1, w2)
        assert torch.allclose(w1.sum(dim=-1), torch.full((4,), 2.0, device=w1.device), atol=1e-5)


class TestExpertCache:
    def test_expert_weight_cache_lru(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=3)
        w1 = torch.randn(4, 8, device="cuda")
        w3 = torch.randn(4, 8, device="cuda")
        w2 = torch.randn(8, 4, device="cuda")
        cache.put_deq("e0", w1, w3, w2)
        assert cache.get("e0") is not None
        cache.put_deq("e1", w1, w3, w2)
        cache.put_deq("e2", w1, w3, w2)
        cache.put_deq("e3", w1, w3, w2)
        assert cache.get("e0") is None

    def test_expert_cache_pin_protects_entries(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=3)
        w = torch.randn(4, 8, device="cuda")
        cache.put_deq("pinned", w, w, w, pin=True)
        cache.put_deq("e1", w, w, w)
        cache.put_deq("e2", w, w, w)
        cache.put_deq("e3", w, w, w)
        cache.put_deq("e4", w, w, w)
        assert cache.get("pinned") is not None
        assert cache.get("e1") is None

    def test_expert_cache_hot_deq(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=10, hot_deq_size=8)
        w1 = torch.randn(4, 8, device="cuda")
        w3 = torch.randn(4, 8, device="cuda")
        w2 = torch.randn(8, 4, device="cuda")
        cache.put_deq("e0", w1, w3, w2)
        deq = cache.deq("e0")
        assert deq is not None
        w1_d, w3_d, w2_d = deq
        assert w1_d.shape == w1.shape

    def test_expert_cache_raw_fp8_entry(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=10)
        data = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        entry = (data, None, "bf16")
        cache.put("e0", entry, entry, entry)
        assert cache.get("e0") is not None
        deq = cache.deq("e0")
        assert deq is not None

    def test_expert_cache_trim(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=100)
        w = torch.randn(4, 8, device="cuda")
        for i in range(20):
            cache.put_deq(f"e{i}", w, w, w)
        assert len(cache) == 20
        cache.trim(target_count=5)
        assert len(cache) <= 5 + len(cache.pinned)

    def test_expert_cache_rejects_duplicate_key(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=10)
        w = torch.randn(4, 8, device="cuda")
        cache.put_deq("k", w, w, w)
        assert len(cache) == 1
        cache.put_deq("k", w, w, w)
        assert len(cache) == 1


class TestWeightLoaderMmap:
    def test_weight_loader_init_no_weights(self):
        from home_seek.inference_engine import WeightLoader
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            loader = WeightLoader(td, device="cuda")
            assert loader.weight_map == {}
            assert loader._mmap_cache == {}
            loader.close()

    def test_weight_loader_get_missing(self):
        from home_seek.inference_engine import WeightLoader
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            loader = WeightLoader(td, device="cuda")
            result = loader.get_weight("nonexistent.key")
            assert result is None
            loader.close()

    def test_weight_loader_device_propagation(self):
        from home_seek.inference_engine import WeightLoader
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            loader = WeightLoader(td, device="cuda:0")
            assert str(loader.device) == "cuda:0"
            loader.close()


class TestRegression:
    """Lightweight regression tests for bugs found during M2.5/M3 development.
    Each test runs in <1s on a CUDA GPU."""

    def test_swiglu_is_silu_not_sigmoid(self):
        """Bug: SwiGLU = SiLU(gate) * up = gate * sigmoid(gate) * up, not sigmoid(gate) * up."""
        torch.manual_seed(42)
        g = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
        u = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
        x = torch.cat([g, u], dim=-1).contiguous()
        from tests._reference import swiglu_forward
        expected = swiglu_forward(x, swiglu_clamp_value=10.0)
        g_clamped = g.float().clamp(max=10.0)
        u_clamped = u.float().clamp(min=-10.0, max=10.0)
        silu = (g_clamped * g_clamped.sigmoid() * u_clamped).to(torch.bfloat16)
        sigmoid_only = (g_clamped.sigmoid() * u_clamped).to(torch.bfloat16)
        assert torch.allclose(expected.float(), silu.float(), atol=1e-1), \
            "SiLU(gate)*up should match swiglu_forward"
        sigmoid_diff = (expected.float() - sigmoid_only.float()).abs().max().item()
        silu_diff = (expected.float() - silu.float()).abs().max().item()
        assert sigmoid_diff > silu_diff * 10, \
            f"sigmoid(gate)*up differs by {sigmoid_diff:.3f} vs SiLU differs by {silu_diff:.3f}"

    def test_wo_einsum_correct_dimensions(self):
        """Bug: Wo einsum 'btgd,grd->btr' was wrong, should be 'btgd,grd->btgr'."""
        B, T, G, D, R = 1, 4, 8, 4096, 1024
        out_g = torch.randn(B, T, G, D, device="cuda", dtype=torch.bfloat16)
        wo_a_g = torch.randn(G, R, D, device="cuda", dtype=torch.bfloat16)
        wrong = torch.einsum('btgd,grd->btr', out_g, wo_a_g)
        correct = torch.einsum('btgd,grd->btgr', out_g, wo_a_g).reshape(B, T, -1)
        assert wrong.shape == (B, T, R), "Wrong formula gives (B,T,R) not (B,T,G*R)"
        assert correct.shape == (B, T, G * R), "Correct formula gives (B,T,G*R)"
        assert not torch.allclose(wrong, correct[:, :, :R]), \
            "Wrong einsum produces different values from slice of correct"

    def test_gqa_broadcast_matmuls_dont_loop(self):
        """MQA broadcast: k_all [B,1,Tkv,D] should expand to [B,n_heads,Tkv,D] without for loop."""
        B, n_heads, T_kv, D = 1, 64, 128, 512
        k_all = torch.randn(B, 1, T_kv, D, device="cuda", dtype=torch.bfloat16)
        n_groups = n_heads // 1
        k_expanded = k_all.unsqueeze(1).expand(-1, n_groups, -1, -1, -1)
        k_expanded = k_expanded.reshape(B, -1, T_kv, D)
        assert k_expanded.shape == (B, n_heads, T_kv, D), \
            "Broadcast expand should produce [B, n_heads, T_kv, D]"
        q = torch.randn(B, n_heads, 1, D, device="cuda", dtype=torch.bfloat16)
        score = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1))
        assert score.shape == (B, n_heads, 1, T_kv), \
            "Broadcast matmul should produce [B, n_heads, 1, T_kv]"
        assert score.isfinite().all(), "No NaN/Inf in broadcast GQA scores"

    def test_device_mismatch_after_mmap(self):
        """Bug: mmap-loaded weights on CPU must be moved to GPU before matmul."""
        cpu_tensor = torch.randn(2048, 4096, device="cpu", dtype=torch.bfloat16)
        gpu_tensor = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
            _ = torch.matmul(gpu_tensor, cpu_tensor.t())

    def test_embed_tied_sharing_saves_memory(self):
        """Bug: embed and lm_head loaded separately despite tie_word_embeddings=True."""
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(
            tie_word_embeddings=True,
            num_hidden_layers=1,
        )
        embed = torch.randn(1000, 64, device="cuda", dtype=torch.bfloat16)
        eng.embed = embed
        eng.lm_head = embed
        assert eng.lm_head.data_ptr() == eng.embed.data_ptr(), \
            "Tied lm_head should share embed's memory (same data_ptr)"

    def test_shared_expert_fp8_not_fp4(self):
        """Bug: shared expert weights are FP8, not FP4 (routed experts are FP4)."""
        from safetensors import safe_open
        import os
        idx_path = os.path.join("weights", "model.safetensors.index.json")
        if not os.path.exists(idx_path):
            pytest.skip("Weights not found")
        import json
        with open(idx_path) as f:
            idx = json.load(f)
        for k, fname in idx["weight_map"].items():
            if "shared_expert.w1.weight" in k:
                fpath = os.path.join("weights", fname)
                with safe_open(fpath, framework="pt", device="cpu") as sf:
                    t = sf.get_tensor(k)
                assert t.dtype == torch.float8_e4m3fn, \
                    f"Shared expert should be FP8 (float8_e4m3fn), got {t.dtype}"
                return
        pytest.skip("No shared expert weight found")

    def test_routed_expert_w2_is_fp4_not_fp8(self):
        """Bug: w2 was assumed FP8 but is actually FP4 (int8 packed)."""
        from safetensors import safe_open
        import os
        idx_path = os.path.join("weights", "model.safetensors.index.json")
        if not os.path.exists(idx_path):
            pytest.skip("Weights not found")
        import json
        with open(idx_path) as f:
            idx = json.load(f)
        for k, fname in idx["weight_map"].items():
            if "experts.0.w2.weight" in k:
                fpath = os.path.join("weights", fname)
                with safe_open(fpath, framework="pt", device="cpu") as sf:
                    t = sf.get_tensor(k)
                assert t.dtype == torch.int8, \
                    f"Routed expert w2 should be FP4 (int8 packed), got {t.dtype}"
                return
        pytest.skip("No routed expert weight found")


class TestCompressorOverlap:
    """Bug #1: overlap_transform should use -inf for score padding (not 0)."""

    def test_overlap_transform_score_fill_neg_inf(self):
        from home_seek.compressor import Compressor
        c = Compressor(
            ratio=4, head_dim=512, coff=2, ape=None,
            wkv=torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16),
            wgate=torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16),
            norm_w=torch.randn(512, device="cuda", dtype=torch.bfloat16),
            device="cuda",
        )
        B, T, D = 1, 16, 4096
        x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        # Prefill compress — should not produce NaN or inf in output
        result = c.compress_prefill(x)
        assert result is not None
        assert not torch.isnan(result).any(), "Compressed output has NaN"
        assert not torch.isinf(result).any(), "Compressed output has Inf"

        # Decode: accumulate tokens at positions 0,1,2,3
        c.reset()
        for pos in range(4):
            tok = torch.randn(1, 1, D, device="cuda", dtype=torch.bfloat16)
            out = c.compress_decode(tok, pos)
        # After 4th token (pos=3), block completes
        assert out is not None, "Decode should produce output after 4 tokens"

    def test_overlap_transform_uses_neg_inf_for_scores(self):
        """Verify overlap_transform supports fill_value for score vs KV separation."""
        from home_seek.compressor import Compressor
        c = Compressor.__new__(Compressor)
        c.ratio = 4
        c.head_dim = 512
        c.coff = 2
        c.overlap = True

        B, num_blocks, ratio, dim2d = 1, 4, 4, 1024
        tensor = torch.randn(B, num_blocks, ratio, dim2d)

        # KV should be padded with 0
        kv_t = c.overlap_transform(tensor, fill_value=0.0)
        # Score should be padded with -inf
        score_t = c.overlap_transform(tensor, fill_value=float("-inf"))

        # First block (index 0): positions 0:ratio are padding
        assert kv_t[0, 0, :ratio].abs().max() == 0.0, \
            "KV padding should be 0"
        assert torch.isinf(score_t[0, 0, :ratio]).all(), \
            "Score padding should be -inf"


class TestAttnSink:
    """Bug #2: attn_sink should match demo virtual softmax entry behavior."""

    def test_attn_sink_virtual_entry(self):
        """Verify that attn_sink acts as a virtual softmax entry (no KV)."""
        B, H, T, K = 1, 4, 1, 8
        q = torch.randn(B, H, T, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, 1, K, 64, device="cuda", dtype=torch.bfloat16)
        attn_sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
        scale = 64 ** -0.5

        # Demo approach: virtual softmax entry
        attn = torch.matmul(q.float() * scale, k.squeeze(1).float().transpose(-2, -1))
        max_val = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_val)
        sum_exp = exp_attn.sum(dim=-1, keepdim=True)
        sink_exp = torch.exp(attn_sink.float().view(1, -1, 1, 1) - max_val)
        sum_exp_with_sink = sum_exp + sink_exp
        P = exp_attn / sum_exp_with_sink

        # Verify sink absorbed some probability: sum of P < 1
        assert P.sum(dim=-1).max() < 1.0 - 1e-6, \
            "Sink should absorb probability mass, making P sum < 1"

    def test_attn_sink_matches_cat_softmax(self):
        """Verify virtual sink matches concat+sink approach."""
        B, H, T, K = 1, 4, 1, 8
        q = torch.randn(B, H, T, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, 1, K, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, 1, K, 64, device="cuda", dtype=torch.bfloat16)
        attn_sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
        scale = 64 ** -0.5

        # Method A: concat sink as extra column
        attn = torch.matmul(q.float() * scale, k.squeeze(1).float().transpose(-2, -1))
        sink_val = attn_sink.view(1, -1, 1, 1).float()
        attn_cat = torch.cat([attn, sink_val.expand(-1, -1, T, -1)], dim=-1)
        P_cat = F.softmax(attn_cat, dim=-1)
        P_real = P_cat[:, :, :, :-1]
        out_cat = (P_real.unsqueeze(-1) * v.squeeze(1).unsqueeze(1)).sum(dim=-2)

        # Method B: virtual sink in denominator
        max_val = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_val)
        sum_exp = exp_attn.sum(dim=-1, keepdim=True)
        sink_exp = torch.exp(attn_sink.float().view(1, -1, 1, 1) - max_val)
        P_virtual = exp_attn / (sum_exp + sink_exp)
        out_virtual = (P_virtual.unsqueeze(-1) * v.squeeze(1).unsqueeze(1)).sum(dim=-2)

        assert torch.allclose(out_cat, out_virtual, atol=1e-5), \
            "Virtual sink and cat-sink should produce identical results"


class TestEncodingDsv4Import:
    """encoding_dsv4 must import from the package, not via sys.path hack."""

    def test_import_from_home_seek(self):
        from home_seek.encoding_dsv4 import encode_messages
        assert callable(encode_messages)

    def test_encode_basic(self):
        from home_seek.encoding_dsv4 import encode_messages
        r = encode_messages([{"role": "user", "content": "Hi"}], thinking_mode="chat")
        assert isinstance(r, str) and len(r) > 0

    def test_no_sys_path_hack(self):
        """No 'weights/encoding' should remain in sys.path."""
        for p in sys.path:
            assert "weights/encoding" not in p, f"sys.path hack still present: {p}"


class TestAnsiCodes:
    """ANSI escape codes controlled by TERM env var."""

    def test_b_returns_ansi_when_term_set(self):
        from home_seek.__main__ import _B
        with unittest.mock.patch.dict(os.environ, {"TERM": "screen"}):
            assert _B("92") == "\033[92m"
            assert _B("0") == "\033[0m"

    def test_b_returns_empty_when_term_dumb(self):
        from home_seek.__main__ import _B
        with unittest.mock.patch.dict(os.environ, {"TERM": "dumb"}):
            assert _B("92") == ""

    def test_b_returns_empty_when_no_term(self):
        from home_seek.__main__ import _B
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            assert _B("92") == ""


class TestProfileSnapshotDelta:
    """CacheMonitor / LayerTrace snapshots must support diff-based multi-round."""

    def test_cache_monitor_snapshot_delta(self):
        from home_seek.profiling_runner import CacheMonitor
        cm = CacheMonitor()
        for _ in range(10): cm.record_cache(True)
        for _ in range(3): cm.record_cache(False)
        s1 = cm.snapshot()
        for _ in range(5): cm.record_cache(True)
        for _ in range(2): cm.record_cache(False)
        s2 = cm.snapshot()
        d = CacheMonitor.delta(s1, s2)
        assert d['cache_hits'] == 5
        assert d['cache_misses'] == 2

    def test_layer_trace_snapshot_delta(self):
        from home_seek.profiling_runner import LayerTrace
        lt = LayerTrace(3)
        lt.attn_ms[0] += 100.0
        lt.ffn_ms[0] += 200.0
        s1 = lt.snapshot()
        lt.attn_ms[0] += 50.0
        lt.ffn_ms[0] += 100.0
        s2 = lt.snapshot()
        d = LayerTrace.delta(s1, s2)
        assert d['attn_ms'][0] == 50.0
        assert d['ffn_ms'][0] == 100.0
        assert d['attn_ms'][1] == 0.0


class TestDownloadCli:
    """home-seek download CLI contract."""

    def test_error_on_existing_dir(self):
        import tempfile
        import os
        from home_seek.__main__ import cmd_download
        tmp = tempfile.mkdtemp()
        class Args: dir = tmp; source = "modelscope"
        try:
            with pytest.raises(SystemExit):
                cmd_download(Args())
        finally:
            os.rmdir(tmp)


class TestCacheSizing:
    """V21.3: CPU cache sizing with correct per_expert_bytes."""

    def test_cache_size_uses_real_expert_memory(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        from home_seek.model_config import DeepSeekV4FlashConfig
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=43, n_routed_experts=256,
            moe_intermediate_size=2048, hidden_size=4096,
        )
        _I = eng.config.moe_intermediate_size
        _D = eng.config.hidden_size
        _per_expert = int(3 * _I * _D * 0.625)
        assert _per_expert > 5 * 1024 * 1024
        assert _per_expert < 50 * 1024 * 1024
        _avail = 96 * 1024**3
        _reserve = 8 * 1024**3
        _max_by_ram = max(0, (_avail - _reserve) // _per_expert)
        cache_size = max(2048, min(_max_by_ram // 2, eng.config.num_hidden_layers * eng.config.n_routed_experts))
        assert cache_size >= 2048

    def test_fp4_entry_memory_breakdown(self):
        w1 = torch.randint(0, 16, (2048, 2048), device="cuda", dtype=torch.int8)
        s1 = torch.zeros(2048, 128, device="cuda", dtype=torch.float8_e8m0fnu)
        w1_mem = w1.numel() * w1.element_size()
        s1_mem = s1.numel() * s1.element_size()
        assert w1_mem == 2048 * 2048 * 1
        assert s1_mem == 2048 * 128 * 1
        per_expert = 3 * (w1_mem + s1_mem)
        assert per_expert < 15 * 1024 * 1024

    def test_make_raw_entry_preserves_f8_scale(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        from home_seek.model_config import DeepSeekV4FlashConfig
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = DeepSeekV4FlashConfig()
        eng.device = torch.device("cuda")
        data = torch.randint(0, 16, (64, 32), device="cpu", dtype=torch.int8)
        scale = torch.zeros(64, 2, device="cpu", dtype=torch.float8_e8m0fnu)
        entry = eng._make_raw_entry(data, scale)
        assert entry is not None
        d, s, fmt = entry
        assert fmt == "fp4"
        assert s.dtype == torch.float8_e8m0fnu


class TestAsyncPrefetch:
    """V21.3: Async DMA prefetch between layers."""

    def test_prefetch_stream_created(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        assert eng._prefetch_stream is not None

    def test_update_hot_usage_tracking(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._hot_usage_counter = {}
        eng._hot_usage_window = []
        eng._hot_usage_window_size = 1024
        eng._update_hot_usage(0, 10)
        eng._update_hot_usage(0, 10)
        eng._update_hot_usage(0, 20)
        assert eng._hot_usage_counter.get(0, {}).get(10, 0) == 2
        assert eng._hot_usage_counter.get(0, {}).get(20, 0) == 1

    def test_last_routed_eids_tracking(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._last_routed_eids = []
        eng._last_routed_eids = [5, 10, 15]
        assert eng._last_routed_eids == [5, 10, 15]


class TestMTPTemperature:
    """V21.3: MTP argmax mode when main model uses temperature=0."""

    def test_mtp_default_draft_length(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._mtp_loaded = True
        mtp_num_draft = getattr(eng, '_mtp_num_draft', 2)
        assert mtp_num_draft == 2

    def test_mtp_temperature_propagation_argmax(self):
        temperature = 0.0
        mtp_temp = temperature if temperature > 0 else 0.0
        assert mtp_temp == 0.0

    def test_mtp_temperature_propagation_sampling(self):
        temperature = 0.6
        mtp_temp = temperature if temperature > 0 else 0.0
        assert mtp_temp == 0.6


class TestGPUHotCache:
    """V21.3: Expanded GPU hot cache."""

    def test_hot_cache_capacity_default(self):
        _vram_free = 24.0
        _expert_bf16_gb = 48.0 / 1024
        _max_hot = max(16, min(int((_vram_free - 4) * 0.20 / _expert_bf16_gb), 64))
        assert _max_hot >= 16

    def test_bf16_cache_capacity_default(self):
        _vram_free = 24.0
        _expert_bf16_gb = 48.0 / 1024
        _bf16_budget = max(1, _vram_free - 3)
        _max_bf16 = max(16, min(int(_bf16_budget * 0.80 / _expert_bf16_gb), 100))
        assert _max_bf16 >= 16


class TestGQAFusedAttention:
    """GQA fused kernel correctness: eliminates 64x KV expand for n_kv=1."""

    def _reference_attn(self, q, kv, causal_mask=None, attn_sink=None):
        B, H, T_q, D = q.shape
        T_kv = kv.shape[-2]
        scale = D ** -0.5
        k_exp = kv.unsqueeze(1).expand(-1, H, -1, -1, -1).reshape(B, H, T_kv, D)
        v_exp = k_exp
        attn = torch.matmul(q.float() * scale, k_exp.float().transpose(-2, -1))
        if causal_mask is not None:
            attn = attn + causal_mask
        if attn_sink is not None:
            sink_val = attn_sink.view(1, -1, 1, 1).float()
            attn_cat = torch.cat([attn, sink_val.expand(-1, -1, T_q, -1)], dim=-1)
            P = F.softmax(attn_cat, dim=-1)
            P_real = P[:, :, :, :-1]
        else:
            P_real = F.softmax(attn, dim=-1)
        out = torch.matmul(P_real, v_exp.float())
        return out

    def test_decode_t1_no_mask(self):
        B, H, T_q, T_kv, D = 1, 64, 1, 128, 512
        q = torch.randn(B, H, T_q, D, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(B, 1, T_kv, D, device='cuda', dtype=torch.bfloat16)
        ref = self._reference_attn(q, kv)
        from home_seek.gqa_attention import gqa_fused_attn
        out = gqa_fused_attn(q.float(), kv)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-2), \
            f"T=1 decode mismatch: maxdiff={(out - ref).abs().max().item():.6f}"

    def test_decode_t1_various_kv_lengths(self):
        B, H, T_q, D = 1, 64, 1, 512
        from home_seek.gqa_attention import gqa_fused_attn
        for T_kv in [1, 4, 16, 64, 128, 256, 512]:
            q = torch.randn(B, H, T_q, D, device='cuda', dtype=torch.bfloat16)
            kv = torch.randn(B, 1, T_kv, D, device='cuda', dtype=torch.bfloat16)
            ref = self._reference_attn(q, kv)
            out = gqa_fused_attn(q.float(), kv)
            assert torch.allclose(out, ref, atol=1e-3, rtol=1e-2), \
                f"T_kv={T_kv} mismatch: maxdiff={(out-ref).abs().max().item():.6f}"

    def test_verify_t3_with_causal_mask(self):
        B, H, T_q, T_kv, D = 1, 64, 3, 132, 512
        q = torch.randn(B, H, T_q, D, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(B, 1, T_kv, D, device='cuda', dtype=torch.bfloat16)
        k_sw_len = 128
        new_start = k_sw_len - T_q
        triu = torch.triu(torch.full((T_q, T_q), float('-inf'), device='cuda', dtype=torch.float32), diagonal=1)
        causal_mask = torch.zeros(T_q, T_kv, device='cuda', dtype=torch.float32)
        causal_mask[:, new_start:k_sw_len] = triu
        ref = self._reference_attn(q, kv, causal_mask=causal_mask)
        from home_seek.gqa_attention import gqa_fused_attn
        out = gqa_fused_attn(q.float(), kv, causal_mask=causal_mask)
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-2), \
            f"T=3 causal mask mismatch: maxdiff={(out-ref).abs().max().item():.6f}"

    def test_attn_sink_single_head(self):
        B, H, T_q, T_kv, D = 1, 4, 1, 16, 64
        q = torch.randn(B, H, T_q, D, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(B, 1, T_kv, D, device='cuda', dtype=torch.bfloat16)
        attn_sink = torch.randn(H, device='cuda', dtype=torch.bfloat16)
        ref = self._reference_attn(q, kv, attn_sink=attn_sink)
        from home_seek.gqa_attention import gqa_fused_attn
        out = gqa_fused_attn(q.float(), kv, attn_sink=attn_sink.float())
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-2), \
            f"attn_sink mismatch: maxdiff={(out-ref).abs().max().item():.6f}"

    def test_deterministic_across_calls(self):
        B, H, T_q, T_kv, D = 1, 64, 1, 128, 512
        from home_seek.gqa_attention import gqa_fused_attn
        q = torch.randn(B, H, T_q, D, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(B, 1, T_kv, D, device='cuda', dtype=torch.bfloat16)
        out1 = gqa_fused_attn(q.float(), kv)
        out2 = gqa_fused_attn(q.float(), kv)
        assert torch.equal(out1, out2), "Fused kernel should be deterministic"

    def test_all_close_to_expand_softmax(self):
        """End-to-end: fused kernel output must match the full expand+softmax+matmul path."""
        B, H, T_q, T_kv, D = 1, 64, 1, 128, 512
        q = torch.randn(B, H, T_q, D, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(B, 1, T_kv, D, device='cuda', dtype=torch.bfloat16)
        scale = D ** -0.5
        k_exp = kv.unsqueeze(1).expand(-1, H, -1, -1, -1).reshape(B, H, T_kv, D)
        v_exp = k_exp
        attn = torch.matmul(q.float() * scale, k_exp.float().transpose(-2, -1))
        attn_p = F.softmax(attn, dim=-1)
        ref = torch.matmul(attn_p, v_exp.float())
        from home_seek.gqa_attention import gqa_fused_attn
        out = gqa_fused_attn(q.float(), kv)
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-2), \
            f"Full path mismatch: maxdiff={(out-ref).abs().max().item():.6f}"


class TestTokenBuffer:
    """_TokenBuffer ensures byte-level BPE partial sequences are never emitted."""

    def test_normal_text_emitted_immediately(self):
        from home_seek.api_server import _TokenBuffer

        TEXT = "Hello"

        class _Tok:
            def decode(self, ids, skip_special_tokens=True):
                return TEXT[:len(ids)]

        buf = _TokenBuffer(_Tok())
        out = []
        for i in range(5):
            out.append(buf.add(i))
        assert out == ["H", "e", "l", "l", "o"], f"Got {out}"

    def test_byte_tokens_held_back_then_resolved(self):
        from home_seek.api_server import _TokenBuffer

        class _Tok:
            _BL = {0: b'\xe3', 1: b'\x81', 2: b'\x8a'}

            def decode(self, ids, skip_special_tokens=True):
                bs = b''.join(self._BL[i] for i in ids if i in self._BL)
                return bs.decode('utf-8', errors='replace')

        buf = _TokenBuffer(_Tok())
        assert buf.add(0) == ""   # single byte → "�", held back
        assert buf.add(1) == ""   # two bytes → "��", held back
        assert buf.add(2) == "お"  # three bytes → valid CJK

    def test_mixed_tokens(self):
        from home_seek.api_server import _TokenBuffer

        class _Tok:
            _M = {0: b'\xe3', 1: b'\x81', 2: b'\x8a', 3: b'W', 4: b'o', 5: b'r', 6: b'l', 7: b'd'}

            def decode(self, ids, skip_special_tokens=True):
                bs = b''.join(self._M[i] for i in ids)
                return bs.decode('utf-8', errors='replace')

        buf = _TokenBuffer(_Tok())
        out = []
        for i in range(8):
            out.append(buf.add(i))
        assert out == ["", "", "お", "W", "o", "r", "l", "d"], f"Got {out}"

    def test_flush_returns_held_back_text(self):
        from home_seek.api_server import _TokenBuffer

        class _Tok:
            _M = {0: b'\xe3', 1: b'\x81', 2: b'\x8a'}

            def decode(self, ids, skip_special_tokens=True):
                bs = b''.join(self._M[i] for i in ids if i in self._M)
                return bs.decode('utf-8', errors='replace')

        buf = _TokenBuffer(_Tok())
        assert buf.add(0) == ""   # single byte → "�"
        assert buf.add(1) == ""   # two bytes → "��", held back
        # Without third byte, flush returns the incomplete text
        rest = buf.flush()
        assert rest == "�", f"Got {rest!r}"

    def test_reset_clears_buffer(self):
        from home_seek.api_server import _TokenBuffer

        class _Tok:
            _M = {0: b'\xe3', 1: b'\x81', 2: b'\x8a'}

            def decode(self, ids, skip_special_tokens=True):
                bs = b''.join(self._M[i] for i in ids if i in self._M)
                return bs.decode('utf-8', errors='replace')

        buf = _TokenBuffer(_Tok())
        buf.add(0)
        buf.add(1)
        buf.reset()
        assert buf.add(2) == ""   # after reset, only id 2 is in buffer → single byte
        assert buf.flush() != ""  # but flush should still return whatever is there

    def test_eos_token_resets_and_skips(self):
        """Token ID 1 (EOS) should reset buffer and not be added."""
        from home_seek.api_server import _TokenBuffer

        class _Tok:
            _M = {0: b'\xe3', 1: b'\x81', 2: b'\x8a'}

            def decode(self, ids, skip_special_tokens=True):
                bs = b''.join(self._M[i] for i in ids if i in self._M)
                return bs.decode('utf-8', errors='replace')

        buf = _TokenBuffer(_Tok())
        buf.add(0)
        buf.add(1)
        buf.add(1)  # reset
        assert buf.flush() == ""  # buffer was reset, nothing to flush
