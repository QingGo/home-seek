import torch
import torch.nn.functional as F
import pytest


class TestFixes:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_expand_kv_direct(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'head_dim': 512, 'num_attention_heads': 64,
                                              'num_key_value_heads': 1, 'o_groups': 8,
                                              'o_lora_rank': 1024, 'hidden_size': 4096})()

        kv_latent = torch.randn(1, 16, 512, device="cuda", dtype=torch.bfloat16)
        k, v = eng._expand_kv(kv_latent)
        assert k.shape == (1, 1, 16, 512), f"K shape wrong: {k.shape}"
        assert v.shape == (1, 1, 16, 512), f"V shape wrong: {v.shape}"
        assert torch.equal(k, v), "K and V should be identical (kv_latent used directly)"

    def test_expand_kv_no_woa_crash(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'head_dim': 512, 'num_attention_heads': 64,
                                              'num_key_value_heads': 1, 'o_groups': 8,
                                              'o_lora_rank': 1024, 'hidden_size': 4096})()

        kv_latent = torch.randn(1, 4, 512, device="cuda", dtype=torch.bfloat16)
        k, v = eng._expand_kv(kv_latent)
        assert k.shape[-2] == 4
        assert v.shape[-2] == 4

    def test_swiglu_clamp(self):
        from tile_reference import swiglu_forward
        x = torch.tensor([[100.0, -100.0, 1.0, -1.0,
                           5.0, -5.0, 2.0, -2.0]], device="cuda", dtype=torch.bfloat16)
        out = swiglu_forward(x, swiglu_clamp_value=10.0)
        assert out.shape == (1, 4)
        assert out.dtype == torch.float32

    def test_mhc_split_sinkhorn(self):
        from home_seek.mhc import mhc_split_sinkhorn
        B, S = 1, 8
        hc_mult = 4
        mix_dim = hc_mult * (2 + hc_mult)
        mixes = torch.randn(B, S, mix_dim, device="cuda", dtype=torch.float32)
        hc_scale = torch.tensor([1.0, 0.5, 0.1], device="cuda", dtype=torch.float32)
        hc_base = torch.randn(mix_dim, device="cuda", dtype=torch.float32)

        pre, post, comb = mhc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=5)
        assert pre.shape == (B, S, hc_mult), f"pre shape: {pre.shape}"
        assert post.shape == (B, S, hc_mult), f"post shape: {post.shape}"
        assert comb.shape == (B, S, hc_mult, hc_mult), f"comb shape: {comb.shape}"
        assert (pre > 0).all(), "pre should be positive (sigmoid + eps)"

    def test_hash_routing(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'num_experts_per_tok': 6, 'num_hash_layers': 3})()

        input_ids = torch.tensor([[5, 10, 15]], device="cuda")
        tid2eid = torch.randint(0, 256, (100, 6), device="cuda")
        eids, weights = eng._compute_hash_experts(input_ids, 0, tid2eid)

        assert eids.shape == (1, 3, 6), f"eids shape: {eids.shape}"
        assert weights.shape == (1, 3, 6), f"weights shape: {weights.shape}"
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3, device="cuda"))

        expected_eid = tid2eid[input_ids]
        assert torch.equal(eids, expected_eid), "Hash routing should match tid2eid lookup"

    def test_routing_sqrtsoftplus(self):
        from home_seek.router import compute_expert_affinity
        hidden = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)

        indices, weights = compute_expert_affinity(hidden, gate, top_k=4)
        assert indices.shape == (2, 4)
        assert weights.shape == (2, 4)
        assert (weights > 0).all()

    def test_expert_weight_cache(self):
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=3)
        w1 = torch.randn(4, 8, device="cuda")
        w3 = torch.randn(4, 8, device="cuda")
        w2 = torch.randn(8, 4, device="cuda")

        cache.put("exp_0", w1, w3, w2)
        result = cache.get("exp_0")
        assert result is not None
        assert torch.equal(result[0], w1)

        assert cache.get("exp_none") is None

        cache.put("exp_1", w1, w3, w2)
        cache.put("exp_2", w1, w3, w2)
        cache.put("exp_3", w1, w3, w2)
        assert cache.get("exp_0") is None, "LRU should evict oldest"
        assert cache.get("exp_3") is not None

    def test_compressed_kv_cache(self):
        from home_seek.inference_engine import CompressedKVCache
        c = CompressedKVCache(compress_ratio=4, dim=512, device="cuda")
        t1 = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
        t2 = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        c.append(t1)
        c.append(t2)
        assert c.get().shape == (5, 512)
        c.clear()
        assert c.get().shape == (0, 512)

    def test_mhc_pre_post_integration(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.verbose = False
        eng.config = type('obj', (object,), {
            'hc_mult': 4, 'hc_sinkhorn_iters': 5, 'hc_eps': 1e-6,
            'num_attention_heads': 64, 'num_key_value_heads': 1,
            'head_dim': 512, 'o_groups': 8, 'o_lora_rank': 1024,
            'hidden_size': 4096, 'sliding_window': 128,
        })()

        B, T, D = 1, 4, 4096
        hidden = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        hc_mult = eng.config.hc_mult
        mix_dim = hc_mult * (2 + hc_mult)
        hc_base = torch.randn(mix_dim, device="cuda", dtype=torch.bfloat16)
        hc_fn = torch.randn(mix_dim, D * hc_mult, device="cuda", dtype=torch.bfloat16)
        hc_scale = torch.tensor([1.0, 0.5, 0.1], device="cuda", dtype=torch.bfloat16)

        h_out, post, comb = eng._forward_mhc(hidden, hc_base, hc_fn, hc_scale, apply_pre=True)
        assert h_out.shape == (B, T, D)

        ffn_out = torch.randn_like(hidden)
        mixed = eng._process_mhc_post(ffn_out, hidden, post.float(), comb.float())
        assert mixed.shape == (B, T, D)

    def test_routing_with_bias(self):
        from home_seek.router import compute_expert_affinity_with_bias
        hidden = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(8, device="cuda", dtype=torch.bfloat16)

        indices, weights = compute_expert_affinity_with_bias(hidden, gate, bias, top_k=4)
        assert indices.shape == (2, 4)
        assert weights.shape == (2, 4)

        indices2, weights2 = compute_expert_affinity_with_bias(hidden, gate, None, top_k=4)
        assert indices2.shape == (2, 4)

    def test_compress_kv(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.device = torch.device("cuda")
        eng.config = type('obj', (object,), {
            'num_hidden_layers': 1, 'hidden_size': 4096,
            'head_dim': 512,
        })()
        eng.config.get_compress_ratio = lambda idx: 4

        B, T, D = 1, 16, 4096
        hidden = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        lw = {
            "attn.compressor.wkv.weight": torch.randn(512, D, device="cuda", dtype=torch.bfloat16),
            "attn.compressor.wgate.weight": torch.randn(512, D, device="cuda", dtype=torch.bfloat16),
            "attn.compressor.norm.weight": torch.randn(512, device="cuda", dtype=torch.bfloat16),
        }
        state = LayerState()
        eng._compress_kv(hidden, lw, 0, state)
        assert state.compressed_kv is not None
        compressed = state.compressed_kv.get()
        assert compressed.shape[0] == T // 4
        assert compressed.shape[1] == 512

    def test_get_compressed_attention_kv(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState, CompressedKVCache
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'head_dim': 512})()

        state = LayerState()
        state.compressed_kv = CompressedKVCache(4, 512, "cuda")
        state.compressed_kv.append(torch.randn(4, 512, device="cuda", dtype=torch.bfloat16))

        result = eng._get_compressed_attention_kv(state, {})
        assert result is not None
        assert result.dim() == 4
        assert result.shape[-1] == 512

        state.compressed_kv.clear()
        result = eng._get_compressed_attention_kv(state, {})
        assert result is None

    def test_mhc_shape_mismatch_skip(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.verbose = False
        eng.config = type('obj', (object,), {
            'hc_mult': 4, 'hc_sinkhorn_iters': 5, 'hc_eps': 1e-6,
            'hidden_size': 4096,
        })()
        hidden = torch.randn(1, 4, 4096, device="cuda", dtype=torch.bfloat16)
        hc_base = torch.randn(12, device="cuda", dtype=torch.bfloat16)
        hc_fn_wrong = torch.randn(12, 4096, device="cuda", dtype=torch.bfloat16)
        hc_scale = torch.tensor([1.0, 0.5], device="cuda", dtype=torch.bfloat16)
        out, post, comb = eng._forward_mhc(hidden, hc_base, hc_fn_wrong, hc_scale, apply_pre=True)
        assert post is None and comb is None
        assert torch.equal(out, hidden)

    def test_forward_single_expert_swiglu(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = type('obj', (object,), {'swiglu_limit': 10.0, 'hidden_size': 4096})()

        h = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)

        out = eng._forward_single_expert(h, w1, w3, w2)
        assert out.shape == (1, 1, 4096)
