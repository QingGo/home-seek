import os
import hashlib
import pytest
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ──────────────────────────────────────────────────────────────────────
# 1. Session 级 Triton kernel 预热 —— 消除每个测试文件的首次 JIT 编译延迟
# ──────────────────────────────────────────────────────────────────────

_triton_prewarmed = False


def _triton_warmup():
    """Trigger one-time JIT compilation of all Triton kernels used in tests."""
    global _triton_prewarmed
    if _triton_prewarmed or not torch.cuda.is_available():
        return
    try:
        import triton
        from home_seek.fused_moe import fused_expert_ffn_triton, _tune_blocks
        from tile_reference import cast

        BM, BN, BK = _tune_blocks(torch.device("cuda"))
        D = max(BK, 128)
        I = max(BN * 2, 64)

        # Create FP4-packed data and warm the fused kernel once
        w1 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w3 = torch.randn(I, D, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(D, I, device="cuda", dtype=torch.bfloat16)
        w1p, w1s = cast(w1, fmt="e2m1", block_size=(1, 32))
        w3p, w3s = cast(w3, fmt="e2m1", block_size=(1, 32))
        w2p, w2s = cast(w2, fmt="e2m1", block_size=(1, 32))

        hidden = torch.randn(1, D, device="cuda", dtype=torch.bfloat16)
        _ = fused_expert_ffn_triton(
            hidden, w1p, w3p, w2p, swiglu_limit=10.0,
            w1_scale=w1s, w3_scale=w3s, w2_scale=w2s,
        )

        hidden_b = torch.randn(4, D, device="cuda", dtype=torch.bfloat16)
        _ = fused_expert_ffn_triton(
            hidden_b, w1p, w3p, w2p, swiglu_limit=10.0,
            w1_scale=w1s, w3_scale=w3s, w2_scale=w2s,
        )

        torch.cuda.synchronize()
        _triton_prewarmed = True
        print("[conftest] Triton kernels pre-warmed OK")
    except Exception as e:
        print(f"[conftest] Triton pre-warm skipped: {type(e).__name__}")


@pytest.fixture(scope="session", autouse=True)
def _triton_session_warmup():
    _triton_warmup()
    yield


# ──────────────────────────────────────────────────────────────────────
# 2. 确定性随机种子 —— 每个测试可复现，失败时打印种子值
# ──────────────────────────────────────────────────────────────────────

def _make_seed(test_name: str) -> int:
    """Derive a deterministic per-test seed from the test name."""
    hash_bytes = hashlib.md5(test_name.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "big") % (2**16)


def pytest_runtest_setup(item):
    """Set deterministic random seeds before each test."""
    test_name = item.nodeid
    seed = _make_seed(test_name)
    # Store the seed on the item so we can retrieve it on failure
    item._deterministic_seed = seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pytest_runtest_makereport(item, call):
    if call.when == "call" and call.excinfo is not None:
        seed = getattr(item, "_deterministic_seed", None)
        if seed is not None:
            setattr(item, "_failed_seed", seed)


# ──────────────────────────────────────────────────────────────────────
# 3. 权重文件缺失 skip 守卫
# ──────────────────────────────────────────────────────────────────────

_weights_available: bool | None = None


def weights_available() -> bool:
    """Check if model weights are present (cached)."""
    global _weights_available
    if _weights_available is None:
        _weights_available = os.path.exists(
            os.path.join("weights", "model.safetensors.index.json")
        )
    return _weights_available


@pytest.fixture
def require_weights():
    """Skip test if model weights are not available."""
    if not weights_available():
        pytest.skip("Model weights not found at weights/")
    return True


# ──────────────────────────────────────────────────────────────────────
# 4. GPU 显存自动清理 —— 防止前一个测试泄漏导致后续 OOM
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _gpu_memory_cleanup():
    """Clean GPU memory before and after each test."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
