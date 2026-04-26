import os
import torch
import pytest


@pytest.mark.fast
class TestAllExpertFP4StoreIndex:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_build_index(self):
        from home_seek.gpu_expert_store import AllExpertFP4Store
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig("weights/config.json")
        store = AllExpertFP4Store("weights", config)

        assert store._file_map is not None
        assert len(store._file_map) > 0
        assert (0, 0, "w1.weight") in store._file_map

    def test_file_map_has_all_experts(self):
        from home_seek.gpu_expert_store import AllExpertFP4Store
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig("weights/config.json")
        store = AllExpertFP4Store("weights", config)

        for layer_idx in range(min(3, config.num_hidden_layers)):
            for eid in [0, 1, 127, 255]:
                for mat in ["w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale"]:
                    assert (layer_idx, eid, mat) in store._file_map, \
                        f"Missing {layer_idx=} {eid=} {mat=}"


@pytest.mark.fast
class TestAllExpertFP4StoreLoad:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_load_single_expert(self):
        from home_seek.gpu_expert_store import AllExpertFP4Store
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig("weights/config.json")
        store = AllExpertFP4Store("weights", config)

        packed = store.get_expert_packed(0, 0)
        assert packed is not None
        assert len(packed) == 6
        gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale = packed
        assert gate_packed.dtype == torch.int8
        assert gate_packed.shape == (config.moe_intermediate_size,
                                     config.hidden_size // 2)
        assert down_packed.shape == (config.hidden_size,
                                     config.moe_intermediate_size // 2)

    def test_load_and_cache(self):
        from home_seek.gpu_expert_store import AllExpertFP4Store
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig("weights/config.json")
        store = AllExpertFP4Store("weights", config, max_experts=16)

        packed = store.get_expert_packed(0, 0)
        assert packed is not None
        assert store.num_cached == 1

        packed_again = store.get_expert_packed(0, 0)
        assert packed_again is not None
        assert store.num_cached == 1

    def test_lru_eviction(self):
        from home_seek.gpu_expert_store import AllExpertFP4Store
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig("weights/config.json")
        store = AllExpertFP4Store("weights", config, max_experts=4)
        for i in range(8):
            store.get_expert_packed(0, i)
        assert store.num_cached <= 4

    def test_cache_on_gpu(self):
        from home_seek.gpu_expert_store import AllExpertFP4Store
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig("weights/config.json")
        store = AllExpertFP4Store("weights", config)

        cpu_data = torch.randint(0, 127, (4, 32), dtype=torch.int8, device="cpu")
        cpu_scale = torch.ones(4, 2, dtype=torch.uint8, device="cpu")
        store.cache_on_gpu(0, 0, cpu_data, cpu_scale, cpu_data, cpu_scale, cpu_data, cpu_scale)

        cached = store.get_cache_key(0, 0)
        assert cached is not None
        d0, _, _, _, _, _ = cached
        assert str(d0.device) == "cuda:0"

    def test_clear(self):
        from home_seek.gpu_expert_store import AllExpertFP4Store
        from home_seek.model_config import DeepSeekV4FlashConfig

        config = DeepSeekV4FlashConfig("weights/config.json")
        store = AllExpertFP4Store("weights", config)

        store.get_expert_packed(0, 0)
        assert store.num_cached > 0
        store.clear()
        assert store.num_cached == 0


@pytest.mark.fast
class TestMakeRawEntryGpuFP4:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_fp4_already_on_gpu(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine

        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.device = torch.device("cuda")

        data_on_gpu = torch.randint(0, 127, (4, 32), dtype=torch.int8, device="cuda")
        scale_on_gpu = torch.ones(4, 2, device="cuda", dtype=torch.float32)

        entry = eng._make_raw_entry(data_on_gpu, scale_on_gpu)
        assert entry is not None
        d, s, fmt = entry
        assert fmt == "fp4_gpu"
        assert str(d.device) == "cuda:0"
        assert str(s.device) == "cuda:0"

    def test_fp4_on_gpu_without_scale(self):
        from home_seek.inference_engine import HomeSeekInferenceEngine

        eng = HomeSeekInferenceEngine.__new__(HomeSeekInferenceEngine)
        eng.device = torch.device("cuda")

        data_on_gpu = torch.randint(0, 127, (4, 32), dtype=torch.int8, device="cuda")
        entry = eng._make_raw_entry(data_on_gpu, None)
        assert entry is not None
        d, s, fmt = entry
        assert fmt == "fp4_gpu"
        assert s is None


@pytest.mark.fast
class TestDequantizeGpuFP4:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

    def test_deq_fp4_gpu_no_device_transfer(self):
        from home_seek.inference_engine import ExpertWeightCache

        cache = ExpertWeightCache(max_experts=16, device="cuda")
        data = torch.randint(0, 127, (4, 32), dtype=torch.int8, device="cuda")
        scale = torch.ones(4, 2, device="cuda", dtype=torch.float32)
        entry = (data, scale, "fp4_gpu")

        result = cache._dequantize_entry(entry, cache.device)
        assert result is not None
        assert result.dtype == torch.bfloat16
        assert str(result.device) == "cuda:0"
        assert result.shape == (4, 64)

    def test_deq_fp4_gpu_with_fp8_scale(self):
        from home_seek.inference_engine import ExpertWeightCache

        cache = ExpertWeightCache(max_experts=16, device="cuda")
        data = torch.randint(0, 127, (4, 32), dtype=torch.int8, device="cuda")
        scale_ue8 = torch.zeros(4, 2, device="cuda", dtype=torch.uint8)
        scale_ue8[:] = 127
        entry = (data, scale_ue8, "fp4_gpu")

        result = cache._dequantize_entry(entry, cache.device)
        assert result is not None
        assert result.dtype == torch.bfloat16
        assert str(result.device) == "cuda:0"
