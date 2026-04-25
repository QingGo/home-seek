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

    def test_mhc_pre_big_fuse_api(self):
        import tile_kernels.modeling.mhc.ops as mhc_ops
        B, T, hc, D = 1, 4, 4, 4096
        residual = torch.randn(B, T, hc, D, device="cuda", dtype=torch.bfloat16)
        fn = torch.randn(hc * (2 + hc), hc * D, device="cuda", dtype=torch.float32)
        scale = torch.tensor([1.0, 0.5, 0.1], device="cuda", dtype=torch.float32)
        base = torch.randn(hc * (2 + hc), device="cuda", dtype=torch.float32)
        post_mix, comb_mix, layer_input = mhc_ops.mhc_pre_big_fuse(
            residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 5)
        assert post_mix.shape == (B, T, hc, 1)
        assert comb_mix.shape == (B, T, hc, hc)
        assert layer_input.shape == (B, T, D)

    def test_mhc_post_api(self):
        import tile_kernels.modeling.mhc.ops as mhc_ops
        B, T, hc, D = 1, 4, 4, 4096
        x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(B, T, hc, D, device="cuda", dtype=torch.bfloat16)
        post_mix = torch.randn(B, T, hc, 1, device="cuda", dtype=torch.float32)
        comb_mix = torch.randn(B, T, hc, hc, device="cuda", dtype=torch.float32)
        try:
            result = mhc_ops.mhc_post(x, residual, post_mix, comb_mix)
            assert result.shape == (B, T, D)
        except Exception as e:
            if "PDL" in str(e) or "tilelang" in str(e).lower():
                pytest.skip("TileKernels MHC post kernel compilation unavailable")
            raise

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


class TestCSAIndexer:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_compute_indexer_shapes(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState, CompressedKVCache
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
        state.compressed_kv = CompressedKVCache(4, config.kv_lora_rank, "cuda", idx_dim=idx_dim)
        state.compressed_kv.append(
            torch.randn(num_compressed, config.kv_lora_rank, device="cuda", dtype=torch.bfloat16),
            torch.randn(num_compressed, idx_dim, device="cuda", dtype=torch.bfloat16)
        )

        lw = {
            "attn.indexer.wq_b.weight": torch.randn(
                config.index_n_heads * config.index_head_dim, q_rank,
                device="cuda", dtype=torch.bfloat16),
        }

        result = eng._compute_indexer(q_latent, lw, state, 0)
        assert result is not None
        assert result.dim() == 4, f"Expected 4D, got {result.dim()}D"
        B_r, n_kv_r, seq_r, hd_r = result.shape
        assert n_kv_r == 1, f"Expected 1 KV head, got {n_kv_r}"
        assert hd_r == config.head_dim, f"Expected head_dim={config.head_dim}, got {hd_r}"
        assert seq_r == config.index_topk, f"Expected {config.index_topk} selected, got {seq_r}"

    def test_compute_indexer_no_weights_fallback(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine, LayerState, CompressedKVCache
        from home_seek.model_config import DeepSeekV4FlashConfig
        config = DeepSeekV4FlashConfig()
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.config = config
        eng._deq_cache = {}
        eng.verbose = False

        B, T, q_rank = 1, 4, config.q_lora_rank
        q_latent = torch.randn(B, T, q_rank, device="cuda", dtype=torch.bfloat16)

        state = LayerState(device="cuda")
        state.compressed_kv = CompressedKVCache(4, config.kv_lora_rank, "cuda")
        state.compressed_kv.append(torch.randn(2, config.kv_lora_rank, device="cuda", dtype=torch.bfloat16))

        result = eng._compute_indexer(q_latent, {}, state, 0)
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
        eng._prefetch_worker = None
        eng._prefetch_enabled = False
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
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

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
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

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
        cache = ExpertWeightCache(max_experts=10)
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


class TestPrefetchWorker:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_async_prefetch_worker_init(self):
        from home_seek.prefetch_worker import AsyncPrefetchWorker
        worker = AsyncPrefetchWorker("/tmp", {}, device="cuda")
        assert worker is not None
        assert worker._stream is not None
        worker.shutdown()

    def test_async_prefetch_clear(self):
        from home_seek.prefetch_worker import AsyncPrefetchWorker
        worker = AsyncPrefetchWorker("/tmp", {}, device="cuda")
        worker._prefetch_cache[(0, 1)] = "dummy"
        worker.clear()
        assert len(worker._prefetch_cache) == 0
        worker.shutdown()

    def test_async_prefetch_get_missing(self):
        from home_seek.prefetch_worker import AsyncPrefetchWorker
        worker = AsyncPrefetchWorker("/tmp", {}, device="cuda")
        result = worker.get(0, 1)
        assert result is None
        worker.shutdown()

    def test_async_prefetch_empty(self):
        from home_seek.prefetch_worker import AsyncPrefetchWorker
        worker = AsyncPrefetchWorker("/tmp", {}, device="cuda")
        worker.prefetch(0, [])
        assert len(worker._pending_keys) == 0
        assert len(worker._prefetch_cache) == 0
        worker.shutdown()


class TestWeightLoaderMmap:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_weight_loader_init_no_weights(self):
        from home_seek.inference_engine import WeightLoader
        import tempfile
        import os
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
