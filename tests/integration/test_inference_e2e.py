import os
import sys
import hashlib
import torch
import pytest


_current_dir = os.path.dirname(os.path.abspath(__file__))
_encoding_dir = os.path.join(_current_dir, '../../weights/encoding')
sys.path.insert(0, os.path.abspath(_encoding_dir))
from encoding_dsv4 import encode_messages


_engine = None
_tokenizer = None


def _get_engine():
    global _engine
    if _engine is None:
        weight_dir = os.environ.get("HOME_SEEK_WEIGHT_DIR", "weights")
        from home_seek.inference_engine import HomeSeekInferenceEngine
        _engine = HomeSeekInferenceEngine(weight_dir=weight_dir, verbose=False)
    return _engine


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        weight_dir = os.environ.get("HOME_SEEK_WEIGHT_DIR", "weights")
        tokenizer_path = os.path.join(weight_dir, "tokenizer.json")
        from transformers import PreTrainedTokenizerFast
        _tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    return _tokenizer


def _encode_prompt(tokenizer, engine, text):
    prompt_text = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
    return tokenizer.encode(prompt_text, return_tensors="pt").to(engine.device)


class TestInferenceE2E:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        weight_dir = os.environ.get("HOME_SEEK_WEIGHT_DIR", "weights")
        config_path = os.path.join(weight_dir, "config.json")
        if not os.path.exists(config_path):
            pytest.skip(f"Weights not found at {weight_dir}")
        tokenizer_path = os.path.join(weight_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            pytest.skip(f"Tokenizer not found at {tokenizer_path}")

    def test_deterministic_generation(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        engine = _get_engine()
        tokenizer = _get_tokenizer()
        input_ids = _encode_prompt(tokenizer, engine, "Hello, world")

        results = []
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            result = engine.generate(input_ids, max_new_tokens=1, temperature=0.0)
            results.append(result["tokens"].cpu())

        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), (
                f"Run 0 and Run {i} differ! Not deterministic."
            )

        peak = torch.cuda.max_memory_allocated()
        print(f"\nDeterministic test passed. Peak memory: {peak / (1024**3):.2f} GB")
        assert peak < 22.5 * (1024**3), f"Memory exceeded 22.5 GB: {peak / (1024**3):.2f} GB"

    def test_generation_succeeds(self):
        engine = _get_engine()
        tokenizer = _get_tokenizer()
        input_ids = _encode_prompt(tokenizer, engine, "Testing inference")

        torch.cuda.reset_peak_memory_stats()
        result = engine.generate(input_ids, max_new_tokens=10)
        assert result["num_generated_tokens"] == 10
        assert result["tokens"].shape[1] == input_ids.shape[1] + 10

        peak = torch.cuda.max_memory_allocated()
        print(f"Generation test: {result['num_generated_tokens']} tokens, "
              f"peak: {peak / (1024**3):.2f} GB")

    def test_regression_hash(self):
        """Verify output matches known-good reference hash.
        Update EXPECTED_HASH by running with --update-hash flag."""
        engine = _get_engine()
        tokenizer = _get_tokenizer()
        input_ids = _encode_prompt(tokenizer, engine, "Hello")

        result = engine.generate(input_ids, max_new_tokens=5, temperature=0.0)
        text = tokenizer.decode(result["tokens"][0], skip_special_tokens=True)
        actual = hashlib.sha256(text.encode()).hexdigest()

        expected = os.environ.get("HOME_SEEK_EXPECTED_HASH")
        if expected:
            assert actual == expected, (
                f"Output hash mismatch!\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}"
            )
        else:
            print(f"\n  [regression] Recorded hash: HOME_SEEK_EXPECTED_HASH={actual}")
            print(f"  [regression] Set env or hardcode this as EXPECTED_HASH in the test")
