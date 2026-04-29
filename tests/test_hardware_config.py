"""Tests for HardwareConfig — Phase 1 of hardware config refactor."""
from __future__ import annotations

from home_seek.hardware_config import HardwareConfig
from home_seek.hw_profile import HWProfile
from home_seek.model_config import DeepSeekV4FlashConfig


def _test_config() -> DeepSeekV4FlashConfig:
    return DeepSeekV4FlashConfig(
        num_hidden_layers=43,
        n_routed_experts=256,
    )


def test_4090_defaults():
    hw = HWProfile(vram_free_gb=22, sm_count=128,
                   gpu_name="NVIDIA GeForce RTX 4090",
                   vram_total_gb=24, mem_bw_gb_s=1008)
    cfg = HardwareConfig.auto(hw, _test_config())
    assert cfg.gpu_hot_max == 64
    assert cfg.gpu_bf16_max == 100
    assert cfg.cublas_max_tokens == 8
    assert cfg.devices == ("cuda:0",)
    assert cfg.mtp_enabled is False
    assert cfg.prefetch_enabled is True
    assert cfg.sm_count == 128


def test_a100_defaults():
    hw = HWProfile(vram_free_gb=38, sm_count=108,
                   gpu_name="NVIDIA A100-PCIE-40GB",
                   vram_total_gb=40, mem_bw_gb_s=1555)
    cfg = HardwareConfig.auto(hw, _test_config())
    # actual computed from 38GB free: (38-4)*0.20/(48/1024)=145
    assert cfg.gpu_hot_max == 145
    assert cfg.gpu_bf16_max == 300
    assert cfg.mtp_enabled is True
    assert cfg.mtp_num_draft == 3
    assert cfg.cublas_max_tokens == 4
    assert cfg.triton_blocks == (16, 32, 64)


def test_h20_defaults():
    hw = HWProfile(vram_free_gb=94, sm_count=78,
                   gpu_name="NVIDIA H20 NVLink 96GB",
                   vram_total_gb=96, mem_bw_gb_s=4000)
    cfg = HardwareConfig.auto(hw, _test_config())
    # hot = (94-4)*0.20/(48/1024) = 384  capped at 1024 → 384
    assert cfg.gpu_hot_max == 384
    assert cfg.gpu_bf16_max == 512
    # cpu_cache_max = max(2048, min(1024, 11008)) = 2048
    assert cfg.cpu_cache_max >= 2048
    assert cfg.triton_blocks == (16, 32, 32)
    assert cfg.cublas_max_tokens == 16
    assert cfg.prefetch_enabled is False
    assert cfg.mtp_enabled is True
    assert cfg.mtp_num_draft == 4


def test_2080_ti_dual():
    hw = HWProfile(vram_free_gb=10, sm_count=68,
                   gpu_name="NVIDIA GeForce RTX 2080 Ti",
                   vram_total_gb=11, mem_bw_gb_s=616)
    cfg = HardwareConfig.auto(hw, _test_config())
    # 10GB free → hot: (10-4)*0.20/(48/1024) = 25, cap 48
    assert cfg.gpu_hot_max == 25
    # bf16: (10-3)*0.80/(48/1024) = 119, cap 80
    assert cfg.gpu_bf16_max == 80
    assert cfg.devices == ("cuda:0", "cuda:1")
    assert len(cfg.device_map) == 43
    assert cfg.device_map[:22] == (0,) * 22
    assert cfg.device_map[22:] == (1,) * 21
    assert cfg.triton_blocks == (16, 16, 32)
    assert cfg.mtp_enabled is False


def test_auto_device_map_2gpus():
    mapping = HardwareConfig._auto_device_map(43, ("cuda:0", "cuda:1"))
    assert mapping[:22] == (0,) * 22
    assert mapping[22:] == (1,) * 21


def test_auto_device_map_3gpus():
    mapping = HardwareConfig._auto_device_map(43, ("cuda:0", "cuda:1", "cuda:2"))
    assert len(mapping) == 43
    # 43/3 = 14.33 → 15 per device: 15+15+13 or 15+14+14
    assert mapping.count(0) >= mapping.count(1) >= mapping.count(2)
    assert mapping.count(0) + mapping.count(1) + mapping.count(2) == 43


def test_auto_device_map_exact():
    mapping = HardwareConfig._auto_device_map(4, ("cuda:0", "cuda:1"))
    assert mapping == (0, 0, 1, 1)


def test_validation_clamps_overcommit():
    hw = HWProfile(vram_free_gb=12, sm_count=68,
                   gpu_name="RTX 2080 Ti")
    cfg = HardwareConfig.auto(hw, _test_config(),
                               gpu_hot_max=9999)
    per_exp = 48.0 / 1024
    max_possible = int(12 * 0.4 / per_exp)
    assert cfg.gpu_hot_max <= max_possible


def test_user_override_wins():
    hw = HWProfile(vram_free_gb=22, sm_count=128,
                   gpu_name="RTX 4090")
    cfg = HardwareConfig.auto(hw, _test_config(),
                               mtp_enabled=True, mtp_num_draft=4)
    assert cfg.mtp_enabled is True
    assert cfg.mtp_num_draft == 4
    assert cfg.gpu_hot_max == 64  # 4090 baseline preserved


def test_unknown_gpu_fallback():
    hw = HWProfile(vram_free_gb=4, sm_count=32,
                   gpu_name="Unknown GPU")
    cfg = HardwareConfig.auto(hw, _test_config())
    # vram clamped to 8 min, cap 32 → (8-4)*0.20/(48/1024)=17, min(17,32)=17, max(16,17)=17
    assert cfg.gpu_hot_max == 17
    assert cfg.gpu_bf16_max == 64
    assert cfg.triton_blocks == (16, 16, 32)


def test_low_vram_clamp():
    hw = HWProfile(vram_free_gb=8, sm_count=68,
                   gpu_name="Old GPU")
    cfg = HardwareConfig.auto(hw, _test_config())
    per_exp = 48.0 / 1024
    max_hot = int(8 * 0.4 / per_exp)
    assert cfg.gpu_hot_max <= max_hot
    assert cfg.gpu_hot_max >= 16


def test_device_map_length_enforced():
    hw = HWProfile(vram_free_gb=22, sm_count=128, gpu_name="RTX 4090")
    cfg = HardwareConfig.auto(
        hw, _test_config(),
        devices=("cuda:0", "cuda:1"),
        device_map=(0, 1) * 21 + (0,),
    )
    assert len(cfg.device_map) == 43


def test_frozen():
    hw = HWProfile(vram_free_gb=22, sm_count=128, gpu_name="RTX 4090")
    cfg = HardwareConfig.auto(hw, _test_config())
    try:
        cfg.gpu_hot_max = 99
        assert False, "should be frozen"
    except Exception:
        pass


def test_to_json(tmp_path):
    hw = HWProfile(vram_free_gb=22, sm_count=128,
                   gpu_name="RTX 4090")
    cfg = HardwareConfig.auto(hw, _test_config())
    p = tmp_path / "hw_config.json"
    cfg.to_json(str(p))
    assert p.exists()
    import json
    data = json.loads(p.read_text())
    assert data["gpu_hot_max"] == 64


def test_pick_triton_blocks():
    assert HardwareConfig._pick_triton_blocks(128, "auto") == (16, 32, 64)
    assert HardwareConfig._pick_triton_blocks(78, "auto") == (16, 32, 32)
    assert HardwareConfig._pick_triton_blocks(68, "auto") == (16, 16, 32)
    assert HardwareConfig._pick_triton_blocks(68, (32, 64, 128)) == (32, 64, 128)


# ── 多 GPU 自动探测 ─────────────────────────────────────


def test_auto_multi_gpu_8x4090():
    """8×4090: n_gpu=8, strategy 不设 devices → 自动生成 8 卡 device_map."""
    hw = HWProfile(vram_free_gb=22, sm_count=128,
                   gpu_name="NVIDIA GeForce RTX 4090",
                   vram_total_gb=24, mem_bw_gb_s=1008,
                   n_gpu=8)
    cfg = HardwareConfig.auto(hw, _test_config())
    # 策略值保持不变（per-GPU）
    assert cfg.gpu_hot_max == 64
    assert cfg.gpu_bf16_max == 100
    # 自动生成 8 卡 device list
    assert cfg.devices == tuple(f"cuda:{i}" for i in range(8))
    assert len(cfg.device_map) == 43
    # 43/8=5.375 → ceil=6, 前 7 卡各 6 层, 末卡 1 层
    counts = [cfg.device_map.count(d) for d in range(8)]
    assert counts == [6, 6, 6, 6, 6, 6, 6, 1], f"unexpected dist: {counts}"
    assert sum(counts) == 43


def test_auto_multi_gpu_4x_unknown():
    """4×未知 GPU: n_gpu=4, fallback 策略 + 自动多卡."""
    hw = HWProfile(vram_free_gb=22, sm_count=82,
                   gpu_name="Unknown GPU Model",
                   vram_total_gb=24,
                   n_gpu=4)
    cfg = HardwareConfig.auto(hw, _test_config())
    # fallback: gpu_hot_cap=32, (22-4)*0.20/(48/1024)=76 → min(76,32)=32
    assert cfg.gpu_hot_max == 32
    assert cfg.devices == tuple(f"cuda:{i}" for i in range(4))
    assert len(cfg.device_map) == 43
    # (43+4-1)//4 = 11 → 前 3 卡各 11 层, 末卡 10
    counts = [cfg.device_map.count(d) for d in range(4)]
    assert counts == [11, 11, 11, 10], f"unexpected dist: {counts}"
    assert sum(counts) == 43
    # fallback 显式设置 triton_preset=(16,16,32), 不走 SM 自动
    assert cfg.triton_blocks == (16, 16, 32)


def test_auto_multi_gpu_does_not_trigger_on_n_gpu_1():
    """单卡 n_gpu=1 不触发自动多卡."""
    hw = HWProfile(vram_free_gb=22, sm_count=128,
                   gpu_name="RTX 4090",
                   n_gpu=1)
    cfg = HardwareConfig.auto(hw, _test_config())
    assert cfg.devices == ("cuda:0",)
    assert cfg.device_map == tuple([0] * 43)


def test_auto_multi_gpu_explicit_strategy_still_works():
    """2080 策略已有 devices=2, 即使 n_gpu=2 不走自动路径."""
    hw = HWProfile(vram_free_gb=10, sm_count=68,
                   gpu_name="NVIDIA GeForce RTX 2080 Ti",
                   vram_total_gb=11,
                   n_gpu=2)
    cfg = HardwareConfig.auto(hw, _test_config())
    # 策略已有 devices → 不走自动路径
    assert cfg.devices == ("cuda:0", "cuda:1")
    assert cfg.device_map[:22] == (0,) * 22
    assert cfg.device_map[22:] == (1,) * 21


def test_auto_multi_gpu_n_gpu_high_but_single_available():
    """n_gpu > 1 但只有一块 GPU（测试假数据），行为无异常."""
    hw = HWProfile(vram_free_gb=4, sm_count=32,
                   gpu_name="Unknown GPU",
                   n_gpu=1)
    cfg = HardwareConfig.auto(hw, _test_config())
    assert cfg.devices == ("cuda:0",)
    assert len(cfg.device_map) == 43


def test_auto_multi_gpu_device_map_distributes_evenly():
    """2 卡 4 层: device_map 均匀分配."""
    hw = HWProfile(vram_free_gb=22, sm_count=128,
                   gpu_name="RTX 4090", n_gpu=2)
    cfg = HardwareConfig.auto(
        hw, DeepSeekV4FlashConfig(num_hidden_layers=4, n_routed_experts=4))
    assert cfg.devices == ("cuda:0", "cuda:1")
    assert cfg.device_map == (0, 0, 1, 1)


# ── 新 GPU 策略 ───────────────────────────────────────


def test_4090d_defaults():
    """RTX 4090 D: 与 4090 基线相同, 但通过 '4090 d' 匹配."""
    hw = HWProfile(vram_free_gb=22, sm_count=128,
                   gpu_name="NVIDIA GeForce RTX 4090 D",
                   vram_total_gb=24, n_gpu=1)
    cfg = HardwareConfig.auto(hw, _test_config())
    assert cfg.gpu_hot_max == 64
    assert cfg.gpu_bf16_max == 100
    assert cfg.devices == ("cuda:0",)


def test_4090d_does_not_trigger_on_plain_4090():
    """RTX 4090 (无 D) 不匹配 '4090 d', 走 '4090' 策略."""
    hw = HWProfile(vram_free_gb=22, sm_count=128,
                   gpu_name="NVIDIA GeForce RTX 4090",
                   vram_total_gb=24)
    cfg = HardwareConfig.auto(hw, _test_config())
    assert cfg.gpu_hot_max == 64


def test_5090_defaults():
    """RTX 5090 (32GB, 170+SM, high BW): 大缓存 + MTP."""
    hw = HWProfile(vram_free_gb=30, sm_count=170,
                   gpu_name="NVIDIA GeForce RTX 5090",
                   vram_total_gb=32, mem_bw_gb_s=1800)
    cfg = HardwareConfig.auto(hw, _test_config())
    # (30-4)*0.20/(48/1024)=110, cap 128
    assert cfg.gpu_hot_max == 110
    # (30-3)*0.80/(48/1024)=461, cap 256
    assert cfg.gpu_bf16_max == 256
    assert cfg.cublas_max_tokens == 4
    assert cfg.prefetch_enabled is False
    assert cfg.mtp_enabled is True
    assert cfg.mtp_num_draft == 3
    assert cfg.triton_blocks == (16, 32, 64)


def test_3090_defaults():
    """RTX 3090 (24GB, 82SM): 同 4090 缓存但 SM 82 → (16,32,32)."""
    hw = HWProfile(vram_free_gb=22, sm_count=82,
                   gpu_name="NVIDIA GeForce RTX 3090",
                   vram_total_gb=24, mem_bw_gb_s=936)
    cfg = HardwareConfig.auto(hw, _test_config())
    assert cfg.gpu_hot_max == 64
    assert cfg.gpu_bf16_max == 100
    assert cfg.triton_blocks == (16, 32, 32)  # SM=82 auto
    assert cfg.mtp_enabled is False


def test_3080_ti_defaults():
    """RTX 3080 Ti (12GB, 80SM): 匹配 '3080 ti' 策略 (先于 '3080')."""
    hw = HWProfile(vram_free_gb=10, sm_count=80,
                   gpu_name="NVIDIA GeForce RTX 3080 Ti",
                   vram_total_gb=12, mem_bw_gb_s=912)
    cfg = HardwareConfig.auto(hw, _test_config())
    # 10GB free → (10-4)*0.20/(48/1024)=25, cap 48
    assert cfg.gpu_hot_max == 25
    assert cfg.gpu_bf16_max == 80  # cap
    assert cfg.triton_blocks == (16, 32, 32)  # SM=80 auto


def test_3080_defaults():
    """RTX 3080 (10GB, 68SM): 保守缓存 + (16,16,32)."""
    hw = HWProfile(vram_free_gb=8, sm_count=68,
                   gpu_name="NVIDIA GeForce RTX 3080",
                   vram_total_gb=10, mem_bw_gb_s=760)
    cfg = HardwareConfig.auto(hw, _test_config())
    # vram clamped to 8 → (8-4)*0.20/(48/1024)=17, cap 32
    assert cfg.gpu_hot_max == 17
    # (8-3)*0.80/(48/1024)=85, cap 64
    assert cfg.gpu_bf16_max == 64
    assert cfg.triton_blocks == (16, 16, 32)  # SM=68 + preset
    assert cfg.cublas_max_tokens == 4


def test_3080_ti_matches_before_3080():
    """gpu_name='RTX 3080 Ti' 应匹配 3080 Ti, 不是 3080."""
    hw_ti = HWProfile(vram_free_gb=10, sm_count=80,
                       gpu_name="RTX 3080 Ti",
                       vram_total_gb=12)
    cfg_ti = HardwareConfig.auto(hw_ti, _test_config())
    hw_plain = HWProfile(vram_free_gb=8, sm_count=68,
                          gpu_name="RTX 3080",
                          vram_total_gb=10)
    cfg_plain = HardwareConfig.auto(hw_plain, _test_config())
    assert cfg_ti.gpu_hot_max > cfg_plain.gpu_hot_max  # Ti 更激进
    assert cfg_ti.triton_blocks == (16, 32, 32)
    assert cfg_plain.triton_blocks == (16, 16, 32)
