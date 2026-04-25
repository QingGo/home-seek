import os
import torch
import pytest


class TestInferenceE2E:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        weight_dir = os.environ.get("HOME_SEEK_WEIGHT_DIR", "weights")
        config_path = os.path.join(weight_dir, "config.json")
        if not os.path.exists(config_path):
            pytest.skip(f"Weights not found at {weight_dir}")

        from home_seek.inference_engine import HomeSeekInferenceEngine
        self.engine = HomeSeekInferenceEngine(
            weight_dir=weight_dir,
            verbose=False,
        )

    def _get_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(
                os.environ.get("HOME_SEEK_WEIGHT_DIR", "weights"),
                trust_remote_code=True,
            )
        except Exception as e:
            pytest.skip(f"Tokenizer not available: {e}")

    def test_deterministic_generation(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tokenizer = self._get_tokenizer()
        input_ids = tokenizer.encode("Hello, world", return_tensors="pt").to(self.engine.device)

        results = []
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result = self.engine.generate(input_ids, max_new_tokens=1)
            results.append(result["tokens"].cpu())

        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), (
                f"Run 0 and Run {i} differ! Not deterministic."
            )

        peak = torch.cuda.max_memory_allocated()
        print(f"\nDeterministic test passed. Peak memory: {peak / (1024**3):.2f} GB")
        assert peak < 22.5 * (1024**3), f"Memory exceeded 22.5 GB: {peak / (1024**3):.2f} GB"

    def test_generation_succeeds(self):
        tokenizer = self._get_tokenizer()
        input_ids = tokenizer.encode("Testing inference", return_tensors="pt").to(self.engine.device)

        torch.cuda.reset_peak_memory_stats()
        result = self.engine.generate(input_ids, max_new_tokens=10)
        assert result["num_generated_tokens"] == 10
        assert result["tokens"].shape[1] == input_ids.shape[1] + 10

        peak = torch.cuda.max_memory_allocated()
        print(f"Generation test: {result['num_generated_tokens']} tokens, "
              f"peak: {peak / (1024**3):.2f} GB")
