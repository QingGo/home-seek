import torch
import torch.nn.functional as F
import pytest
from home_seek.mhc import mhc_split_sinkhorn
from home_seek.inference_engine import HomeSeekInferenceEngine, rms_norm
from tile_reference import swiglu_forward, stable_topk, unpack_from_e2m1fn_x2


class TestExpandKV:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_expand_kv_direct(self):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'head_dim': 512})()

        kv_latent = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
        k, v = eng._expand_kv(kv_latent)
        assert k.shape == (2, 1, 8, 512)
        assert v.shape == (2, 1, 8, 512)
        assert torch.equal(k, v)

    def test_expand_kv_no_woa_crash(self):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'head_dim': 512})()
        kv_latent = torch.randn(1, 4, 512, device="cuda", dtype=torch.bfloat16)
        k, v = eng._expand_kv(kv_latent)
        assert k.shape[-2] == 4

    def test_w2_fp4_quantization(self):
        from home_seek.inference_engine import load_fp4_weight, load_fp8_weight
        w2_data = torch.randint(0, 15, (4, 1024), device="cuda", dtype=torch.int8)
        scale_u8 = torch.zeros(4, 64, device="cuda", dtype=torch.uint8)
        scale_u8[:, :] = 127
        scale_f32 = scale_u8.view(torch.uint8).to(torch.int32)
        scale_f32 = (scale_f32 << 23).view(torch.float32)
        deq = load_fp4_weight(w2_data, scale_f32)
        assert deq.shape == (4, 2048)
        assert deq.dtype == torch.bfloat16


class TestSwiGLU:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

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
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

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
        eng.config = type('obj', (object,), {
            'hc_mult': 4, 'hc_sinkhorn_iters': 5, 'hc_eps': 1e-6, 'rms_norm_eps': 1e-6,
            'num_attention_heads': 64, 'num_key_value_heads': 1,
            'head_dim': 512, 'o_groups': 8, 'o_lora_rank': 1024,
            'hidden_size': 4096, 'sliding_window': 128,
        })()
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
        eng.config = type('obj', (object,), {
            'hc_mult': 4, 'hc_sinkhorn_iters': 5, 'hc_eps': 1e-6, 'rms_norm_eps': 1e-6, 'hidden_size': 4096,
        })()
        hidden = torch.randn(1, 4, 4, 4096, device="cuda", dtype=torch.bfloat16)
        hc_base = torch.randn(12, device="cuda", dtype=torch.bfloat16)
        hc_fn_wrong = torch.randn(12, 4096, device="cuda", dtype=torch.bfloat16)
        hc_scale = torch.tensor([1.0, 0.5], device="cuda", dtype=torch.bfloat16)
        out, post, comb = eng._forward_mhc(hidden, hc_base, hc_fn_wrong, hc_scale, apply_pre=True)
        assert post is None and comb is None
        assert torch.equal(out, hidden.sum(dim=2))


class TestRouting:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_hash_routing(self):
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'num_experts_per_tok': 6, 'num_hash_layers': 3})()
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


class TestAttention:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_mla_attention_shapes(self):
        from home_seek.compressed_attention import MLAAttention
        from home_seek.model_config import DeepSeekV4FlashConfig
        config = DeepSeekV4FlashConfig()
        attn = MLAAttention(config, device="cuda")
        B, T = 1, 64
        hidden = torch.randn(B, T, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_a = torch.randn(config.q_lora_rank, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_b = torch.randn(config.num_attention_heads * config.head_dim, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        kv_a = torch.randn(config.q_lora_rank, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        kv_b = torch.randn(config.num_key_value_heads * config.head_dim * 2, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        o_p = torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim, device="cuda", dtype=torch.bfloat16)
        out, k, v = attn.forward_mla(hidden, q_a, q_b, kv_a, kv_b, o_p)
        assert out.shape == (B, T, config.hidden_size)

    def test_mla_attention_cache_incremental(self):
        from home_seek.compressed_attention import MLAAttention
        from home_seek.model_config import DeepSeekV4FlashConfig
        config = DeepSeekV4FlashConfig()
        attn = MLAAttention(config, device="cuda")
        B, T = 1, 4
        hidden = torch.randn(B, T, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_a = torch.randn(config.q_lora_rank, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_b = torch.randn(config.num_attention_heads * config.head_dim, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        kv_a = torch.randn(config.q_lora_rank, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        kv_b = torch.randn(config.num_key_value_heads * config.head_dim * 2, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        o_p = torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim, device="cuda", dtype=torch.bfloat16)
        out1, k1, v1 = attn.forward_mla(hidden, q_a, q_b, kv_a, kv_b, o_p)
        h2 = torch.randn(B, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out2, k2, v2 = attn.forward_mla(h2, q_a, q_b, kv_a, kv_b, o_p, past_k=k1, past_v=v1)
        assert out2.shape == (B, 1, config.hidden_size)
        assert k2.shape[-2] == T + 1


class TestKVCache:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_kv_cache_manager(self):
        from home_seek.kv_cache_manager import KVCacheManager
        from home_seek.model_config import DeepSeekV4FlashConfig
        config = DeepSeekV4FlashConfig()
        mgr = KVCacheManager(config, device="cuda")
        mgr.init_cache(batch_size=1)
        B, n_kv, D = 1, config.num_key_value_heads, config.head_dim
        for layer_idx in range(min(5, config.num_hidden_layers)):
            k = torch.randn(B, n_kv, 4, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, n_kv, 4, D, device="cuda", dtype=torch.bfloat16)
            mgr.update(layer_idx, k, v)
            swa_k, swa_v = mgr.get_swa_cache(layer_idx)
            assert swa_k.shape[-2] >= 4
        stats = mgr.get_memory_stats()
        assert "allocated_gb" in stats

    def test_compressed_kv_cache(self):
        from home_seek.inference_engine import CompressedKVCache
        c = CompressedKVCache(compress_ratio=4, dim=512, device="cuda")
        t1 = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
        t2 = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        c.append(t1)
        c.append(t2)
        assert c.get().shape == (5, 512)

    def test_get_compressed_attention_kv(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState, CompressedKVCache
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'head_dim': 512})()
        state = LayerState()
        state.compressed_kv = CompressedKVCache(4, 512, "cuda")
        state.compressed_kv.append(torch.randn(4, 512, device="cuda", dtype=torch.bfloat16))
        result = eng._get_compressed_attention_kv(state, {})
        assert result is not None and result.dim() == 4

    def test_compress_kv(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.device = torch.device("cuda")
        eng.config = type('obj', (object,), {
            'num_hidden_layers': 1, 'hidden_size': 4096, 'head_dim': 512,
        })()
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
        assert state.compressed_kv is not None
        c = state.compressed_kv.get()
        assert c.shape[0] == T // 4 and c.shape[1] == 512


class TestQuantization:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_fp4_roundtrip(self):
        from tile_reference import cast, unpack_from_e2m1fn_x2
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
        from tile_reference import cast, cast_back
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        q, sf = cast(x, fmt="e4m3", block_size=(32, 32))
        dq = cast_back((q, sf), fmt="fp32", block_size=(32, 32))
        dq = dq[:64, :128]
        cos = F.cosine_similarity(x.flatten().unsqueeze(0).float(), dq.flatten().unsqueeze(0)).item()
        assert cos >= 0.995

    def test_expert_weight_approximation(self):
        from tile_reference import cast, unpack_from_e2m1fn_x2
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


class TestExpertCache:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_expert_weight_cache(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=3)
        w1 = torch.randn(4, 8, device="cuda")
        w3 = torch.randn(4, 8, device="cuda")
        w2 = torch.randn(8, 4, device="cuda")
        cache.put("e0", w1, w3, w2)
        assert cache.get("e0") is not None
        cache.put("e1", w1, w3, w2)
        cache.put("e2", w1, w3, w2)
        cache.put("e3", w1, w3, w2)
        assert cache.get("e0") is None
