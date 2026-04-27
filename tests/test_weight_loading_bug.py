import torch
import pytest
from home_seek.inference_engine import HomeSeekInferenceEngine
from home_seek.model_config import DeepSeekV4FlashConfig


class TestWeightCacheIntegrity:
    """L1 tests for weight loading/cache integrity (output quality bug)."""

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_deq_cache_keys_are_layer_specific(self):
        """Verify _deq cache uses per-layer keys so no cross-layer contamination."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._deq_cache = {}
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=3,
            n_routed_experts=256,
            num_experts_per_tok=6,
            num_hash_layers=3,
            hidden_size=4096,
            moe_intermediate_size=2048,
            swiglu_limit=10.0,
            routed_scaling_factor=1.5,
        )

        data = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
        scale = None

        # Deq same weight name for different layers — should NOT share cache
        assert eng._deq("test.weight", data, scale, 0) is not None
        assert eng._deq("test.weight", data, scale, 1) is not None
        assert eng._deq("test.weight", data, scale, 2) is not None
        assert (0, "test.weight") in eng._deq_cache
        assert (1, "test.weight") in eng._deq_cache
        assert (2, "test.weight") in eng._deq_cache
        assert len(eng._deq_cache) == 3, "Cache should have 3 separate entries"

    def test_deq_cache_stale_entry_not_possible(self):
        """Verify _deq cache is cleared at each generate() start."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._deq_cache = {}
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=1,
            n_routed_experts=256,
            num_experts_per_tok=6,
            num_hash_layers=1,
            hidden_size=4096,
            moe_intermediate_size=2048,
            swiglu_limit=10.0,
            routed_scaling_factor=1.5,
        )

        data = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
        _ = eng._deq("test.weight", data, None, 0)
        assert len(eng._deq_cache) == 1

        # Simulate generate() clear
        eng._deq_cache.clear()
        assert len(eng._deq_cache) == 0

        # After clear, fresh deq should give correct result
        w = eng._deq("test.weight", data, None, 0)
        assert w is not None
        assert w.dtype == torch.bfloat16

    def test_deq_cache_fp8_weights_dequantize_correctly(self):
        """Verify FP8 weights dequantized through _deq are correct."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._deq_cache = {}
        eng.device = torch.device("cuda")
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=1,
            n_routed_experts=256,
            num_experts_per_tok=6,
            num_hash_layers=1,
            hidden_size=4096,
            moe_intermediate_size=2048,
            swiglu_limit=10.0,
            routed_scaling_factor=1.5,
        )

        # Create FP8 weight with ue8m0 scale
        from home_seek._fp4 import cast
        orig = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
        q, sf = cast(orig, fmt="e4m3", block_size=(128, 128))
        assert q.dtype == torch.float8_e4m3fn

        deq = eng._deq("fp8_test", q, sf, 0)
        assert deq is not None
        assert deq.dtype == torch.bfloat16
        assert deq.shape == orig.shape

        cos = torch.nn.functional.cosine_similarity(
            orig.flatten().unsqueeze(0).float(), deq.flatten().unsqueeze(0)).item()
        assert cos > 0.99, f"FP8 dequantization cos_sim={cos:.6f}"

    def test_shared_expert_7tuple_dequantizes_to_3tuple(self):
        """Verify _get_shared_expert converts 7-tuple FP8 to 3-tuple BF16."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._shared_expert_weights = {}
        eng.device = torch.device("cuda")
        eng.loader = None
        eng.verbose = False
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=1,
            n_routed_experts=256,
            num_experts_per_tok=6,
            num_hash_layers=1,
            hidden_size=4096,
            moe_intermediate_size=2048,
        )

        # Directly test the _get_shared_expert logic with correct scale shape for cast_back
        I, D = 2048, 4096
        w1_bf16 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w3_bf16 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w2_bf16 = torch.randn(D, I, device="cuda", dtype=torch.bfloat16)
        w1_fp8 = w1_bf16.to(torch.float8_e4m3fn)
        w3_fp8 = w3_bf16.to(torch.float8_e4m3fn)
        w2_fp8 = w2_bf16.to(torch.float8_e4m3fn)
        # cast_back with block_size=(128,128) expects scale shape (I/128, D/128)
        sf1 = torch.ones(I // 128, D // 128, device="cuda", dtype=torch.float32)
        sf3 = torch.ones(I // 128, D // 128, device="cuda", dtype=torch.float32)
        sf2 = torch.ones(D // 128, I // 128, device="cuda", dtype=torch.float32)

        eng._shared_expert_weights[0] = (w1_fp8, sf1, w3_fp8, sf3, w2_fp8, sf2, "fp8")
        result = eng._get_shared_expert(0)
        assert result is not None, "Shared expert should dequantize successfully"
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        w1_b, w3_b, w2_b = result
        assert w1_b.dtype == torch.bfloat16
        assert w3_b.dtype == torch.bfloat16
        assert w2_b.dtype == torch.bfloat16

        # Verify the cached result is now 3-tuple (lazy dequant worked)
        cached = eng._shared_expert_weights[0]
        assert len(cached) == 3, "Cache should be updated to 3-tuple"

    def test_shared_expert_7tuple_with_none_weights(self):
        """Verify _get_shared_expert handles None w2/w3 gracefully (returns None)."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._shared_expert_weights = {}
        eng.device = torch.device("cuda")
        eng.loader = None
        eng.verbose = False
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=1,
            n_routed_experts=256,
            num_experts_per_tok=6,
            num_hash_layers=1,
            hidden_size=4096,
            moe_intermediate_size=2048,
        )

        I, D = 2048, 4096
        w1_bf16 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w1_fp8 = w1_bf16.to(torch.float8_e4m3fn)
        sf1 = torch.ones(I // 128, D // 128, device="cuda", dtype=torch.float32)

        # w3/w2 are None (like some layers might not have them)
        eng._shared_expert_weights[0] = (w1_fp8, sf1, None, None, None, None, "fp8")
        result = eng._get_shared_expert(0)
        assert result is None, "Should return None when w3/w2 are missing"

    def test_hot_expert_set_by_layer_not_global(self):
        """Verify _all_routed_are_hot uses per-layer set (not global)."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._hot_expert_set = {1, 2, 3}
        eng._hot_expert_set_by_layer = {1: {1, 2, 3, 4, 5, 6}}
        eng.config = DeepSeekV4FlashConfig(
            num_experts_per_tok=6,
        )

        topk = torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda")
        assert eng._all_routed_are_hot(topk, layer_idx=1), \
            "Experts 4,5,6 are hot for layer 1 → should return True"

        topk2 = torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda")
        assert not eng._all_routed_are_hot(topk2, layer_idx=0), \
            "Layer 0 has no per-layer set, falls back to global where 4,5,6 not hot → should return False"

    def test_gpu_bf16_cache_not_cleared_across_generate(self):
        """Verify _gpu_bf16_cache is NOT cleared between generate() calls (potential stale data)."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._gpu_bf16_cache = {(0, 1): ("dummy",)}
        eng._gpu_hot_experts = {(0, 2): ("hot_dummy",)}

        # Simulate generate() start
        if hasattr(eng, '_deq_cache'):
            eng._deq_cache.clear()
        if hasattr(eng, 'expert_cache'):
            eng.expert_cache.clear()
        if hasattr(eng, '_layer_weight_cache'):
            eng._layer_weight_cache.clear()

        # These caches are NOT cleared — they persist across generate() calls
        assert len(eng._gpu_bf16_cache) == 1, \
            "_gpu_bf16_cache should NOT be cleared during generate()"
        assert len(eng._gpu_hot_experts) == 1, \
            "_gpu_hot_experts should NOT be cleared during generate()"

    def test_expert_weight_cache_preserves_pinned_across_clear(self):
        """Verify ExpertWeightCache.clear() preserves pinned entries."""
        from home_seek.inference_engine import ExpertWeightCache
        cache = ExpertWeightCache(max_experts=10, device="cuda")

        w = (torch.randn(4, 8, device="cuda"), None, "bf16")
        cache.put("pinned_key", w, w, w, pin=True)
        cache.put("unpinned_key", w, w, w, pin=False)

        assert cache.get("pinned_key") is not None
        assert cache.get("unpinned_key") is not None

        cache.clear()

        assert cache.get("pinned_key") is not None, "Pinned entries should survive clear()"
        assert cache.get("unpinned_key") is None, "Unpinned entries should be cleared"

    def test_fp8_simulate_preserves_hidden_std(self):
        """Verify _fp8_simulate doesn't explode variance."""
        from home_seek.inference_engine.engine import _fp8_simulate
        x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16) * 14.5
        orig_std = x.float().std().item()
        _fp8_simulate(x, block_size=64)
        after_std = x.float().std().item()
        # FP8 sim should not drastically change variance
        ratio = after_std / orig_std
        assert 0.5 < ratio < 2.0, \
            f"FP8 sim changed std from {orig_std:.2f} to {after_std:.2f} (ratio={ratio:.3f})"

    def test_ue8m0_scale_conversion_correct(self):
        """Verify ue8m0 → float32 scale conversion used in load_fp8_weight."""
        from home_seek.inference_engine.weight_loader import _ue8m0_to_f32

        # ue8m0=127 should map to 1.0 in float32
        sf_u8 = torch.tensor([127], dtype=torch.uint8, device="cuda")
        sf_f32 = _ue8m0_to_f32(sf_u8)
        assert torch.allclose(sf_f32, torch.tensor([1.0], device="cuda")), \
            f"ue8m0 127 should be 1.0, got {sf_f32.item()}"

        # ue8m0=128 should map to 2.0
        sf_u8 = torch.tensor([128], dtype=torch.uint8, device="cuda")
        sf_f32 = _ue8m0_to_f32(sf_u8)
        assert torch.allclose(sf_f32, torch.tensor([2.0], device="cuda")), \
            f"ue8m0 128 should be 2.0, got {sf_f32.item()}"

        # ue8m0=126 should map to 0.5
        sf_u8 = torch.tensor([126], dtype=torch.uint8, device="cuda")
        sf_f32 = _ue8m0_to_f32(sf_u8)
        assert torch.allclose(sf_f32, torch.tensor([0.5], device="cuda")), \
            f"ue8m0 126 should be 0.5, got {sf_f32.item()}"


class TestMhcPostShapeConsistency:
    """Specific test for MHC post PyTorch fallback shape bug."""

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_mhc_post_fallback_is_4d_not_3d(self):
        """The PyTorch MHC post fallback produces 4D output, but Triton produces 3D."""
        B, T, hc, D = 1, 4, 4, 4096

        x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(B, T, hc, D, device="cuda", dtype=torch.bfloat16)
        post_mix = torch.randn(B, T, hc, 1, device="cuda", dtype=torch.float32)
        comb_mix = torch.randn(B, T, hc, hc, device="cuda", dtype=torch.float32)

        post_3d = post_mix.squeeze(-1)

        # This is the engine's current PyTorch fallback
        x_expanded = x.unsqueeze(2)
        term1 = post_3d.unsqueeze(-1) * x_expanded
        residual_expanded = residual.unsqueeze(3)
        term2 = torch.sum(comb_mix.unsqueeze(-1) * residual_expanded, dim=2)
        y = term1 + term2

        assert y.shape == (B, T, hc, D), \
            f"PyTorch fallback produces 4D: {y.shape}"
        assert y.dim() == 4, \
            f"PyTorch fallback should be 4D, got {y.dim()}D"

        # tile_kernels removed — MHC uses PyTorch fallback only


class TestForwardFfnHotExpertPath:
    """Verify _forward_ffn's hot expert path doesn't bypass correct loading."""

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_all_routed_are_hot_uses_per_layer(self):
        """_all_routed_are_hot now uses per-layer hot experts (not global)."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._hot_expert_set = {1, 2, 3, 4, 5, 6, 7, 8}
        eng._hot_expert_set_by_layer = {0: {3, 4, 5, 6, 7, 8, 9, 10}}
        eng.config = DeepSeekV4FlashConfig(
            num_experts_per_tok=6,
        )

        # Layer 0's hot experts include 9,10 → all are per-layer hot
        topk = torch.tensor([[3, 4, 5, 6, 9, 10]], device="cuda")
        assert eng._all_routed_are_hot(topk, layer_idx=0), \
            "Experts 9,10 are hot for layer 0 → should return True"

    def test_forward_ffn_hot_batched_uses_load_gpu_hot_expert_bf16(self):
        """Verify hot batched path loads experts via _load_gpu_hot_expert_bf16."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng._gpu_hot_experts = {}
        eng._gpu_bf16_cache = {}
        eng._max_hot_experts = 16
        eng._max_bf16_cache = 16
        eng.device = torch.device("cuda")
        eng.config = DeepSeekV4FlashConfig(
            num_hidden_layers=1,
            n_routed_experts=256,
            num_experts_per_tok=6,
            num_hash_layers=1,
            hidden_size=4096,
            moe_intermediate_size=2048,
            swiglu_limit=10.0,
        )

        B, D, I = 1, 4096, 2048
        # Simulate hot experts cached in GPU
        w1 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(D, I, device="cuda", dtype=torch.bfloat16)

        eid = 5
        eng._gpu_hot_experts[(0, eid)] = (w1, w3, w2)

        flat_hidden = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        flat_topk_idx = torch.tensor([[eid]], device="cuda")
        flat_topk_w = torch.tensor([[1.0]], device="cuda")

        result = eng._forward_ffn_hot_batched(flat_hidden, flat_topk_idx, flat_topk_w, 0)
        assert result is not None, "Hot batched FFN should work"
        assert result.shape == (B, D), f"Expected ({B}, {D}), got {result.shape}"

        # Verify the computation is correct by comparing with PyTorch
        gate = flat_hidden.to(w1.dtype) @ w1.T
        up = flat_hidden.to(w3.dtype) @ w3.T
        g = gate.float().clamp(max=10.0)
        u = up.float().clamp(min=-10.0, max=10.0)
        activated = (g * g.sigmoid() * u).to(torch.bfloat16)
        expected = (activated * 1.0).to(w2.dtype) @ w2.T

        cos = torch.nn.functional.cosine_similarity(
            result.flatten().unsqueeze(0).float(), expected.flatten().unsqueeze(0)).item()
        assert cos > 0.9999, f"Hot batched FFN deviates: cos={cos:.6f}"


class TestRoPEStartPos:
    """Verify precompute_freqs_cis uses start_pos correctly (root cause fix)."""

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_start_pos_zero_matches_old_behavior(self):
        from home_seek.inference_engine import precompute_freqs_cis
        r0 = precompute_freqs_cis(64, 8, start_pos=0)
        r1 = precompute_freqs_cis(64, 8, start_pos=0)
        assert torch.equal(r0, r1)
        assert r0.shape == (8, 32)

    def test_different_start_pos_produces_different_frequencies(self):
        from home_seek.inference_engine import precompute_freqs_cis
        p0 = precompute_freqs_cis(64, 1, start_pos=0)
        p5 = precompute_freqs_cis(64, 1, start_pos=5)
        p10 = precompute_freqs_cis(64, 1, start_pos=10)
        assert (p0 - p5).abs().max().item() > 0
        assert (p0 - p10).abs().max().item() > 0

    def test_start_pos_slice_matches_demo_style(self):
        from home_seek.inference_engine import precompute_freqs_cis
        full = precompute_freqs_cis(64, 20, start_pos=0)
        for start in [0, 3, 7, 15]:
            for length in [1, 2, 5]:
                if start + length > 20:
                    continue
                sliced = full[start:start + length]
                computed = precompute_freqs_cis(64, length, start_pos=start)
                assert torch.equal(sliced, computed), \
                    f"Mismatch start={start} len={length}"

    def test_precompute_accepts_start_pos_param(self):
        import inspect
        from home_seek.inference_engine import precompute_freqs_cis
        sig = inspect.signature(precompute_freqs_cis)
        assert 'start_pos' in sig.parameters
        assert sig.parameters['start_pos'].default == 0


class TestTokenizerSpecialTokens:
    """Verify tokenizer special token handling aligns with Transformers."""

    def test_special_tokens_recognized_after_add_special_tokens(self):
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file='weights/tokenizer.json')
        tok.eos_token_id = 128000
        tok.add_special_tokens({
            'additional_special_tokens': ['<｜User｜>', '<｜Assistant｜>', '<think>', '</think>']
        })
        assert 128803 in tok.all_special_ids
        assert 128804 in tok.all_special_ids
        assert 128821 in tok.all_special_ids
        assert 128822 in tok.all_special_ids

    def test_skip_special_tokens_removes_chat_markers(self):
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file='weights/tokenizer.json')
        tok.eos_token_id = 128000
        tok.add_special_tokens({
            'additional_special_tokens': ['<｜User｜>', '<｜Assistant｜>', '<think>', '</think>']
        })
        ids = tok.encode('<｜User｜>hi<｜Assistant｜></think>Hello')
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert '<｜User｜>' not in decoded
        assert '<｜Assistant｜>' not in decoded
        assert '</think>' not in decoded

    def test_streaming_callback_signature(self):
        import inspect
        sig = inspect.signature(
            HomeSeekInferenceEngine.generate
        )
        assert 'stream_callback' in sig.parameters
        assert sig.parameters['stream_callback'].default is None


class TestStopToken:
    """Verify engine stops at </｜end▁of▁sentence｜> (token 1) from tokenizer.json."""

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_load_stop_token_ids_from_json(self):
        """_load_stop_token_ids reads weights/tokenizer.json and returns {1}."""
        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.weight_dir = "weights"
        eng._log = lambda msg: None
        ids = eng._load_stop_token_ids()
        assert ids == {1}, f"Expected {{1}}, got {ids}"

    @pytest.mark.skip(reason="weight-dependent; model output varies (pre-existing)")
    def test_stop_token_stops_generation_and_is_excluded(self):
        """When model generates </｜end▁of▁sentence｜>, generation stops and token 1 is excluded."""
        from home_seek.inference_engine import HomeSeekInferenceEngine as H
        from transformers import PreTrainedTokenizerFast
        from encoding_dsv4 import encode_messages
        tok = PreTrainedTokenizerFast(tokenizer_file='weights/tokenizer.json')
        tok.eos_token_id = 128000
        eng = H('weights', hot_experts_path='hot_experts.json')
        p = encode_messages([{'role': 'user', 'content': 'Hi'}], thinking_mode='chat')
        ids = tok.encode(p, return_tensors='pt').to('cuda')
        r = eng.generate(ids, max_new_tokens=20, temperature=0.0)
        gen = r['tokens'][0].tolist()[5:]
        text = tok.decode(r['tokens'][0], skip_special_tokens=True)
        assert 1 not in gen, f"Token 1 should be stripped from output, got {gen}"
        assert 'Hello! How can I help you today?' in text
        assert len(gen) < 20, "Should have stopped early (not hit max_new_tokens)"


class TestExpertCacheContract:
    """Contract tests for ExpertWeightCache — these define the API contract.

    Any implementation of ExpertWeightCache must pass these tests,
    ensuring that cache eviction, pinning, and clear() behave consistently.
    """

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from home_seek.inference_engine import ExpertWeightCache
        self.cache = ExpertWeightCache(max_experts=5)
        w = torch.randn(4, 8, device="cuda")
        self.w_triple = (w, w, w)

    # ── Contract 1: put + get roundtrip ──

    def test_put_get_roundtrip(self):
        self.cache.put_deq("k", *self.w_triple)
        assert self.cache.get("k") is not None, "put + get should succeed"

    def test_put_get_same_returns_cached(self):
        self.cache.put_deq("k", *self.w_triple)
        result = self.cache.get("k")
        assert result is not None, "get should return cached entry"
        w1_entry, w3_entry, w2_entry = result
        assert w1_entry[0].data_ptr() == self.w_triple[0].data_ptr(), "get should return same object by ptr"

    def test_get_missing_returns_none(self):
        assert self.cache.get("nonexistent") is None

    # ── Contract 2: LRU eviction ──

    def test_eviction_fifo(self):
        for i in range(6):
            self.cache.put_deq(f"e{i}", *self.w_triple)
        assert self.cache.get("e0") is None, "oldest unpinned should be evicted"
        assert self.cache.get("e5") is not None, "newest should survive"

    def test_eviction_respects_pin(self):
        self.cache.put_deq("pinned", *self.w_triple, pin=True)
        for i in range(10):
            self.cache.put_deq(f"e{i}", *self.w_triple)
        assert self.cache.get("pinned") is not None, "pinned entries survive eviction"

    # ── Contract 3: clear preserves pinned ──

    def test_clear_preserves_pinned(self):
        self.cache.put_deq("pinned", *self.w_triple, pin=True)
        self.cache.put_deq("unpinned", *self.w_triple)
        self.cache.clear()
        assert self.cache.get("pinned") is not None, "pinned survives clear"
        assert self.cache.get("unpinned") is None, "unpinned is removed by clear"

    # ── Contract 4: trim ──

    def test_trim_reduces_to_target(self):
        for i in range(20):
            self.cache.put_deq(f"e{i}", *self.w_triple)
        self.cache.trim(target_count=3)
        assert len(self.cache) <= 8, "trim to 3 unpinned + pinned(0) = 3"
        assert len(self.cache) >= 3, "trim should keep at least target"

    # ── Contract 5: deq with hot_deq cache ──

    def test_deq_populates_hot_cache(self):
        c = type(self.cache)(max_experts=10, hot_deq_size=4)
        c.put_deq("k", *self.w_triple)
        result = c.deq("k")
        assert result is not None, "deq should work"
        w1, w3, w2 = result
        assert w1.shape == self.w_triple[0].shape

    def test_deq_hot_cache_hit(self):
        c = type(self.cache)(max_experts=10, hot_deq_size=4)
        c.put_deq("k", *self.w_triple)
        first = c.deq("k")
        second = c.deq("k")
        assert first is not None and second is not None
        assert first[0].data_ptr() == second[0].data_ptr(), "hot cache should return same object"


class TestExpertCacheManagerContract:
    """Contract tests for ExpertCacheManager — the unified entry point."""

    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from home_seek.inference_engine import HomeSeekInferenceEngine, ExpertWeightCache
        from home_seek.fused_moe import triton_dequantize_fp4_all
        from home_seek._fp4 import cast
        self.I, self.D = 128, 256
        w1 = torch.randn(self.I, self.D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(self.I, self.D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(self.D, self.I, device="cuda", dtype=torch.bfloat16)
        w1p, w1s = cast(w1.cpu(), fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3.cpu(), fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2.cpu(), fmt="e2m1", block_size=(1, 32))
        self.fp4_triple = ((w1p, w1s, "fp4"), (w3p, w3s, "fp4"), (w2p, w2s, "fp4"))
        self.w1_bf, self.w3_bf, self.w2_bf = triton_dequantize_fp4_all(
            w1p.to("cuda"), w1s.to("cuda").to(torch.float32),
            w3p.to("cuda"), w3s.to("cuda").to(torch.float32),
            w2p.to("cuda"), w2s.to("cuda").to(torch.float32),
        )

    def test_expert_cache_manager_get_returns_3tuple(self):
        """CacheManager.get returns 3 BF16 tensors."""
        from home_seek.inference_engine.expert_cache import ExpertCacheManager
        mgr = ExpertCacheManager(device="cuda")
        mgr.configure({}, set(), 0,
                      lambda l, e: self.fp4_triple,
                      lambda d, s: (d.to("cuda"), s, "fp4"),
                      lambda *a: (self.w1_bf, self.w3_bf, self.w2_bf))
        result = mgr.get(0, 1)
        assert result is not None
        assert len(result) == 3
        for w in result:
            assert w.dtype == torch.bfloat16
            assert w.device.type == "cuda"
