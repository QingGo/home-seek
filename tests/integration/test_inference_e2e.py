import os
import sys
import torch
import pytest


_current_dir = os.path.dirname(os.path.abspath(__file__))
_encoding_dir = os.path.join(_current_dir, '../../weights/encoding')
sys.path.insert(0, os.path.abspath(_encoding_dir))
from encoding_dsv4 import encode_messages


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
        import os
        weight_dir = os.environ.get("HOME_SEEK_WEIGHT_DIR", "weights")
        tokenizer_path = os.path.join(weight_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            pytest.skip(f"Tokenizer file not found at {tokenizer_path}")
        try:
            from transformers import PreTrainedTokenizerFast
            return PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        except Exception as e:
            pytest.skip(f"Tokenizer not available: {e}")

    def _encode_prompt(self, tokenizer, text):
        prompt_text = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
        return tokenizer.encode(prompt_text, return_tensors="pt").to(self.engine.device)

    def test_deterministic_generation(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tokenizer = self._get_tokenizer()
        input_ids = self._encode_prompt(tokenizer, "Hello, world")

        results = []
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result = self.engine.generate(input_ids, max_new_tokens=1, temperature=0.0)
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
        input_ids = self._encode_prompt(tokenizer, "Testing inference")

        torch.cuda.reset_peak_memory_stats()
        result = self.engine.generate(input_ids, max_new_tokens=10)
        assert result["num_generated_tokens"] == 10
        assert result["tokens"].shape[1] == input_ids.shape[1] + 10

        peak = torch.cuda.max_memory_allocated()
        print(f"Generation test: {result['num_generated_tokens']} tokens, "
              f"peak: {peak / (1024**3):.2f} GB")
