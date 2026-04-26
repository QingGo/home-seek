import torch
import pytest


class TestTileOps:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_cast_to_fp4_and_back(self):
        from tile_reference import cast, unpack_from_e2m1fn_x2
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        quantized, sf = cast(x, fmt="e2m1", block_size=(1, 32))

        deq = unpack_from_e2m1fn_x2(quantized)
        sf_expanded = sf.repeat_interleave(32, dim=1)
        deq = deq.to(torch.float32) * sf_expanded.to(torch.float32)
        deq = deq[:x.shape[0], :x.shape[1]]

        cos_sim = torch.nn.functional.cosine_similarity(
            x.flatten().unsqueeze(0).float(),
            deq.flatten().unsqueeze(0),
        ).item()
        assert cos_sim >= 0.99, f"Cosine similarity too low: {cos_sim}"

    def test_cast_to_fp8_and_back(self):
        from tile_reference import cast
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        quantized, sf = cast(x, fmt="e4m3", block_size=(32, 32))
        assert quantized.dtype == torch.float8_e4m3fn
        assert sf.shape[0] == 2
        assert sf.shape[1] == 4

    def test_stable_topk(self):
        from tile_reference import stable_topk
        scores = torch.tensor([[1.0, 5.0, 3.0, 7.0], [8.0, 2.0, 6.0, 4.0]], device="cuda")
        indices = stable_topk(scores, 2)
        assert indices[0, 0].item() == 3
        assert indices[0, 1].item() == 1
        assert indices[1, 0].item() == 0
        assert indices[1, 1].item() == 2

    def test_swiglu_forward(self):
        from tile_reference import swiglu_forward
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        out = swiglu_forward(x)
        assert out.shape == (4, 4)
        assert out.dtype == torch.float32

        x_left = x[:, :4].float()
        x_right = x[:, 4:].float()
        expected = (x_left / (1.0 + torch.exp(-x_left))) * x_right
        assert torch.allclose(out, expected, atol=1e-5)

    def test_swiglu_with_weights(self):
        from tile_reference import swiglu_forward
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        pos = torch.tensor([0, 1, 2, 3], device="cuda")
        weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]], device="cuda")
        out = swiglu_forward(x, pos_to_token_topk=pos, topk_weights=weights)
        assert out.shape == (4, 4)

    def test_reduce_fused(self):
        from tile_reference import reduce_fused
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        token_topk_to_pos = torch.tensor([
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], device="cuda")
        topk_weights = torch.tensor([
            [0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2],
        ], device="cuda")
        out = reduce_fused(x, topk_weights, token_topk_to_pos)
        assert out.shape == (4, 16)

    def test_expand_to_fused(self):
        from tile_reference import expand_to_fused
        x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        token_topk_to_pos = torch.tensor([
            [0, 4], [1, 5], [2, -1], [3, 7],
        ], device="cuda")
        pos_to_expert = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device="cuda")
        out = expand_to_fused(x, token_topk_to_pos, pos_to_expert)
        assert out.shape == (8, 16)
        assert (out[0] == x[0]).all()
        assert (out[4] == x[0]).all()
        assert (out[2] == x[2]).all()
        assert (out[6] == 0).all()  # pos 6 not mapped

    def test_rms_norm(self):
        from home_seek.inference_engine import rms_norm
        x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(8, device="cuda", dtype=torch.bfloat16)
        out = rms_norm(x, weight)
        assert out.shape == x.shape
        var = x.float().pow(2).mean(-1, keepdim=True)
        expected = x.float() * torch.rsqrt(var + 1e-6)
        assert torch.allclose(out.float(), expected, rtol=1e-2, atol=1e-2)

    def test_deterministic_routing(self):
        from home_seek.router import compute_expert_affinity
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        hidden = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)

        indices_1, weights_1 = compute_expert_affinity(hidden, gate, top_k=2)
        indices_2, weights_2 = compute_expert_affinity(hidden, gate, top_k=2)
        assert torch.equal(indices_1, indices_2)
        assert torch.equal(weights_1, weights_2)

    def test_mla_attention(self):
        from home_seek.compressed_attention import MLAAttention
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig()
        attn = MLAAttention(config, device="cuda")

        B, T = 1, 64
        hidden = torch.randn(B, T, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        q_a = torch.randn(config.q_lora_rank, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_b = torch.randn(config.num_attention_heads * config.head_dim, config.q_lora_rank,
                          device="cuda", dtype=torch.bfloat16)
        kv_a = torch.randn(config.q_lora_rank, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        kv_b = torch.randn(config.num_key_value_heads * config.head_dim * 2, config.q_lora_rank,
                          device="cuda", dtype=torch.bfloat16)
        o_p = torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim,
                          device="cuda", dtype=torch.bfloat16)

        out, k, v = attn.forward_mla(hidden, q_a, q_b, kv_a, kv_b, o_p)
        assert out.shape == (B, T, config.hidden_size)
        head_out_dim = config.num_attention_heads * config.head_dim
        assert k.shape == (B, config.num_key_value_heads, T, config.head_dim)
        assert v.shape == (B, config.num_key_value_heads, T, config.head_dim)

    def test_mla_attention_with_cache(self):
        from home_seek.compressed_attention import MLAAttention
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig()
        attn = MLAAttention(config, device="cuda")

        B, T = 1, 4
        hidden = torch.randn(B, T, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        q_a = torch.randn(config.q_lora_rank, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_b = torch.randn(config.num_attention_heads * config.head_dim, config.q_lora_rank,
                          device="cuda", dtype=torch.bfloat16)
        kv_a = torch.randn(config.q_lora_rank, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        kv_b = torch.randn(config.num_key_value_heads * config.head_dim * 2, config.q_lora_rank,
                          device="cuda", dtype=torch.bfloat16)
        o_p = torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim,
                          device="cuda", dtype=torch.bfloat16)

        out1, k1, v1 = attn.forward_mla(hidden, q_a, q_b, kv_a, kv_b, o_p)
        hidden2 = torch.randn(B, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out2, k2, v2 = attn.forward_mla(hidden2, q_a, q_b, kv_a, kv_b, o_p, past_k=k1, past_v=v1)
        assert out2.shape == (B, 1, config.hidden_size)
        assert k2.shape[-2] == T + 1

    def test_kv_cache_manager(self):
        from home_seek.kv_cache_manager import KVCacheManager
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig()
        manager = KVCacheManager(config, device="cuda")
        manager.init_cache(batch_size=1)

        B, n_kv, D = 1, config.num_key_value_heads, config.head_dim

        for layer_idx in range(min(5, config.num_hidden_layers)):
            k = torch.randn(B, n_kv, 4, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, n_kv, 4, D, device="cuda", dtype=torch.bfloat16)
            manager.update(layer_idx, k, v)

            swa_k, swa_v = manager.get_swa_cache(layer_idx)
            assert swa_k.shape[-2] >= 4

            compressed = manager.get_compressed_cache(layer_idx)
            if compressed[0] is not None:
                csa_k, csa_v, hca_k, hca_v = compressed
                assert csa_k.shape[-2] >= 0

        stats = manager.get_memory_stats()
        assert "allocated_gb" in stats
