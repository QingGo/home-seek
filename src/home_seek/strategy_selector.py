"""
硬件自适应并行策略选择器。

根据 HWProfile 中的拓扑信息 (interconnect_tier, NUMA map, P2P 带宽)，
自动选择最优并行后端 (PP/EP)、device_map、GPU 缓存大小和功能开关。

被 hardware_config.py:auto() 在 GPU 型号匹配后调用，覆盖并行相关参数。
"""

from __future__ import annotations

import logging

from home_seek.hw_profile import HWProfile
from home_seek.model_config import DeepSeekV4FlashConfig

_logger = logging.getLogger(__name__)


def select_parallel_strategy(
    hw: HWProfile, cfg: DeepSeekV4FlashConfig
) -> dict:
    """根据硬件拓扑选择最优并行策略配置。

    Returns:
        dict  — 要覆盖到 HardwareConfig 的参数 (parallel_backend, devices,
                device_map, gpu_hot_max, gpu_bf16_max 等)。
    """
    tier = getattr(hw, "interconnect_tier", "single")
    n = max(1, hw.n_gpu if hasattr(hw, "n_gpu") else 1)
    n_layers = cfg.num_hidden_layers if hasattr(cfg, "num_hidden_layers") else 43

    params: dict = {
        "devices": tuple(f"cuda:{i}" for i in range(n)),
        "device_map": tuple([0] * n_layers),
        "parallel_backend": "pp",
    }

    if n <= 1:
        _configure_single(params, hw, cfg)
    elif tier in ("unknown", "single"):
        # n>1 但 tier=unknown/single: 保守走 PP
        _configure_pcie_numa(params, hw, cfg)
    elif tier == "nvlink":
        _configure_nvlink(params, hw, cfg)
    elif tier == "pcie_p2p":
        _configure_pcie_p2p(params, hw, cfg)
    elif tier == "numa_remote":
        _configure_numa_remote(params, hw, cfg)
    else:
        _configure_pcie_numa(params, hw, cfg)

    return params


# ── 各策略实现 ───────────────────────────────────────


def _configure_single(params: dict, hw: HWProfile, cfg: DeepSeekV4FlashConfig) -> None:
    """单卡：禁用并行。缓存大小由 GPU 型号基线策略决定。"""
    params["devices"] = ("cuda:0",)
    params["device_map"] = tuple([0] * cfg.num_hidden_layers)
    params["parallel_backend"] = "pp"


def _configure_nvlink(params: dict, hw: HWProfile, cfg: DeepSeekV4FlashConfig) -> None:
    """NVLink：最高带宽互联 → EP（专家可通过 NVLink 零拷贝共享）。"""
    vram = _vram_per_gpu(hw)
    n = _n_gpu(hw)
    per_exp = 48.0 / 1024

    params["parallel_backend"] = "ep"
    params["device_map"] = tuple([0] * cfg.num_hidden_layers)
    params["gpu_hot_max"] = max(32, min(int(vram * 0.30 / per_exp), 512))
    params["gpu_bf16_max"] = max(64, min(int(vram * 0.50 / per_exp), 512))
    params["prefetch_enabled"] = False
    params["mtp_enabled"] = True
    _logger.info(f"  [Strategy] NVLink ×{n}: EP backend, large GPU cache")


def _configure_pcie_p2p(params: dict, hw: HWProfile, cfg: DeepSeekV4FlashConfig) -> None:
    """同 NUMA + P2P 可达 → EP（专家可通过 P2P 在卡间搬运）。"""
    vram = _vram_per_gpu(hw)
    n = _n_gpu(hw)
    per_exp = 48.0 / 1024

    params["parallel_backend"] = "ep"
    params["device_map"] = tuple([0] * cfg.num_hidden_layers)
    params["gpu_hot_max"] = max(32, int((vram - 3) * 0.20 / per_exp))
    params["gpu_bf16_max"] = max(64, int((vram - 3) * 0.50 / per_exp))
    params["prefetch_enabled"] = True
    _logger.info(f"  [Strategy] PCIe+P2P ×{n}: EP backend")


def _configure_numa_remote(params: dict, hw: HWProfile, cfg: DeepSeekV4FlashConfig) -> None:
    """跨 NUMA 无 P2P：EP + NUMA 绑定，利用独立内存控制器并行加载。"""
    vram = _vram_per_gpu(hw)
    n = _n_gpu(hw)
    per_exp = 48.0 / 1024

    params["parallel_backend"] = "ep"
    params["device_map"] = tuple([0] * cfg.num_hidden_layers)
    # P2: EP 下每个 GPU 需缓存所有层的专家。
    # VRAM 预算: total - 3GB(non-expert) - 2GB(KV) - 1GB(misc) = available
    # BF16 缓存上限 50% 可用 VRAM, 留余量避免 OOM
    _safe_vram = max(8, vram - 6)
    params["gpu_hot_max"] = max(32, min(int(_safe_vram * 0.20 / per_exp), 128))
    params["gpu_bf16_max"] = max(64, min(int(_safe_vram * 0.50 / per_exp), 250))
    params["prefetch_enabled"] = True
    params["ep_numa_aware"] = True
    _logger.info(f"  [Strategy] NUMA Remote ×{n}: EP backend + NUMA bind")


def _configure_pcie_numa(params: dict, hw: HWProfile, cfg: DeepSeekV4FlashConfig) -> None:
    """同 NUMA 无 P2P（保守）：PP 最可靠，层均分。

    PP 模式下缓存大小由 GPU 型号基线策略决定，策略选择器不覆盖。
    """
    n = _n_gpu(hw)
    params["parallel_backend"] = "pp"
    params["device_map"] = _auto_device_map(cfg.num_hidden_layers, params["devices"])
    params["prefetch_enabled"] = True
    _logger.info(f"  [Strategy] PCIe NUMA ×{n}: PP backend (conservative)")


# ── 工具函数 ─────────────────────────────────────────


def _vram_per_gpu(hw: HWProfile) -> float:
    if hasattr(hw, "per_device_vram_gb") and hw.per_device_vram_gb:
        return hw.per_device_vram_gb[0]
    return getattr(hw, "vram_total_gb", 24.0)


def _n_gpu(hw: HWProfile) -> int:
    return max(1, hw.n_gpu if hasattr(hw, "n_gpu") else 1)


def _auto_device_map(n_layers: int, devices: tuple[str, ...]) -> tuple[int, ...]:
    """均分层到各 GPU。"""
    n = len(devices)
    per_device = (n_layers + n - 1) // n
    mapping: list[int] = []
    for dev_idx in range(n):
        start = dev_idx * per_device
        end = min(start + per_device, n_layers)
        mapping.extend([dev_idx] * (end - start))
    return tuple(mapping)
