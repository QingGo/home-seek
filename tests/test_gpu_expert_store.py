import torch
import pytest





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
