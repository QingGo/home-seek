from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import ClassVar

from home_seek.hw_profile import HWProfile
from home_seek.model_config import DeepSeekV4FlashConfig

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareConfig:
    """Engine 只读的硬件配置 —— 所有硬件相关参数集中在此。

    设计约束:
    - frozen=True: 防止运行时意外修改
    - 仅含原生类型 + tuple (可 JSON 序列化)
    - 不含 torch.Tensor 或 CUDA 引用
    """

    # ── GPU 缓存 ────────────────────────────────────
    gpu_hot_max: int = 64
    """_gpu_hot_experts LRU 上限。每项 ~48 MB (BF16 dequant 后)。"""

    gpu_bf16_max: int = 100
    """_gpu_bf16_cache LRU 上限。"""

    # ── CPU 缓存 ────────────────────────────────────
    cpu_cache_max: int = 8192
    """ExpertWeightCache (CPU FP4) 上限。"""

    # ── KV offload ──────────────────────────────────
    kv_offload_threshold_gb: float = 18.0
    """GPU 已分配显存超过此值 (GiB) 时触发 CPU offload。"""

    # ── Triton kernel ────────────────────────────────
    triton_blocks: tuple[int, int, int] = (16, 32, 64)
    """(BM, BN, BK) — 传给所有 Triton kernel 的 block 大小。"""

    cublas_max_tokens: int = 8
    """M ≤ 此值时走 cuBLAS 而非 Triton (小 M 时 Triton SM 空转)。"""

    # ── 多 GPU ──────────────────────────────────────
    devices: tuple[str, ...] = ("cuda:0",)
    """可用 GPU 列表, 按优先级排序。"""

    device_map: tuple[int, ...] = ()
    """每层对应的 devices 索引。空元组 = 全部用 devices[0] (单卡)。"""

    # ── 功能开关 ────────────────────────────────────
    prefetch_enabled: bool = True
    """异步预取下一层专家。高 VRAM 时可关闭。"""

    mtp_enabled: bool = False
    """启用 MTP 投机解码。高带宽 GPU 推荐启用。"""

    mtp_num_draft: int = 2
    """MTP 每步生成的 draft token 数。"""

    # ── 身份信息 (仅供日志/调试) ────────────────────
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    mem_bw_gb_s: float = 0.0
    sm_count: int = 0

    # ────────────────────────────────────────────────
    # Factory methods
    # ────────────────────────────────────────────────

    _STRATEGIES: ClassVar[dict[str, dict]] = {
        # ── 当前基线 ──────────────────────────────────
        "4090": {
            "gpu_hot_cap": 64,
            "gpu_bf16_cap": 100,
            "kv_offload_gb": 18,
            "cublas_max_tokens": 8,
            "prefetch": True,
            "mtp": False,
        },
        # ── 配置 1: A100-40GB ────────────────────────
        "a100": {
            "gpu_hot_cap": 160,
            "gpu_bf16_cap": 300,
            "kv_offload_gb": 30,
            "cublas_max_tokens": 4,
            "prefetch": True,
            "mtp": True,
            "mtp_draft": 3,
        },
        # ── 配置 2: H20-96GB ─────────────────────────
        "h20": {
            "gpu_hot_cap": 1024,
            "gpu_bf16_cap": 512,
            "cpu_cache_max": 1024,
            "kv_offload_gb": 75,
            "triton_preset": (16, 32, 32),
            "cublas_max_tokens": 16,
            "prefetch": False,
            "mtp": True,
            "mtp_draft": 4,
        },
        # ── 配置 3: RTX PRO 6000-96GB ───────────────
        "rtx pro 6000": {
            "gpu_hot_cap": 1024,
            "gpu_bf16_cap": 512,
            "kv_offload_gb": 75,
            "cublas_max_tokens": 8,
            "prefetch": False,
            "mtp": False,
        },
        # ── 配置 4: 双卡 2080 Ti ─────────────────────
        "2080": {
            "gpu_hot_cap": 48,
            "gpu_bf16_cap": 80,
            "kv_offload_gb": 7,
            "cublas_max_tokens": 4,
            "triton_preset": (16, 16, 32),
            "prefetch": True,
            "mtp": False,
            "devices": ("cuda:0", "cuda:1"),
        },
        # ── 未识别 GPU 的保守 fallback ──────────────
        "fallback": {
            "gpu_hot_cap": 32,
            "gpu_bf16_cap": 64,
            "kv_offload_gb": 8,
            "cublas_max_tokens": 4,
            "triton_preset": (16, 16, 32),
            "prefetch": True,
            "mtp": False,
        },
    }

    @classmethod
    def auto(cls, hw: HWProfile, model_cfg: DeepSeekV4FlashConfig,
             **overrides) -> HardwareConfig:
        """从硬件探测结果自动推导配置, 支持用户覆盖任意字段。"""
        base = cls._match_strategy(hw.gpu_name, hw, model_cfg)
        merged = {**base, **overrides}
        validated = cls._validate(merged, hw, model_cfg)
        config = cls(**validated)

        n_gpu = max(1, hw.n_gpu if hasattr(hw, 'n_gpu') else 1)
        n_layers = model_cfg.num_hidden_layers
        _logger.info(
            f"HardwareConfig: {hw.gpu_name}"
            f"  ×{n_gpu}  "
            f"VRAM={config.vram_total_gb:.0f}GB  "
            f"SM={config.sm_count}  "
            f"hot={config.gpu_hot_max}  "
            f"bf16={config.gpu_bf16_max}  "
            f"cpu={config.cpu_cache_max}  "
            f"offload={config.kv_offload_threshold_gb:.0f}GB  "
            f"blocks={config.triton_blocks}  "
            f"cublas≤{config.cublas_max_tokens}  "
            f"prefetch={config.prefetch_enabled}  "
            f"mtp={config.mtp_enabled}(draft={config.mtp_num_draft})  "
            f"devices={len(config.devices)}  "
            f"device_map={len(config.device_map)}layers"
        )
        if len(config.devices) > 1:
            _logger.info(
                f"  Multi-GPU device map ({n_layers} layers): "
                + ", ".join(
                    f"GPU{d}:{config.device_map.count(d)}layers"
                    for d in range(len(config.devices))
                )
            )
        if overrides:
            _logger.info(f"  User overrides: {overrides}")
        return config

    @classmethod
    def _match_strategy(cls, gpu_name: str, hw: HWProfile,
                        cfg: DeepSeekV4FlashConfig) -> dict:
        """按 GPU 型号匹配最近似的预设策略。"""
        name_lower = gpu_name.lower()
        for pattern, strategy in cls._STRATEGIES.items():
            if pattern in name_lower:
                if pattern != "fallback":
                    _logger.info(f"  Matched strategy: {pattern}")
                return cls._compute(strategy, hw, cfg)
        _logger.info("  No strategy match, using fallback")
        return cls._compute(cls._STRATEGIES.get("fallback", {}), hw, cfg)

    @classmethod
    def _compute(cls, strategy: dict, hw: HWProfile,
                 cfg: DeepSeekV4FlashConfig) -> dict:
        """将预设策略的语义参数展开为具体数值。"""
        vram = max(hw.vram_free_gb, 8.0)
        per_exp_bf16 = 48.0 / 1024  # GiB

        hot_cap = strategy.get("gpu_hot_cap", 64)
        hot = max(16, min(int((vram - 4) * 0.20 / per_exp_bf16), hot_cap))

        bf16_cap = strategy.get("gpu_bf16_cap", 100)
        bf16 = max(16, min(int((vram - 3) * 0.80 / per_exp_bf16), bf16_cap))

        cpu_max = max(2048,
                      min(strategy.get("cpu_cache_max", 8192),
                          cfg.num_hidden_layers * cfg.n_routed_experts))

        # ── 多 GPU 自动探测 ──────────────────────────
        n_gpu = max(1, hw.n_gpu if hasattr(hw, 'n_gpu') else 1)
        n_layers = cfg.num_hidden_layers
        devices_explicit = strategy.get("devices")

        if devices_explicit is not None:
            devices = devices_explicit
        elif n_gpu > 1:
            _logger.info(
                f"  Auto multi-GPU: {n_gpu} × {hw.gpu_name}")
            devices = tuple(f"cuda:{i}" for i in range(n_gpu))
        else:
            devices = ("cuda:0",)

        if len(devices) > 1 and not strategy.get("device_map"):
            device_map = cls._auto_device_map(n_layers, devices)
        else:
            device_map = strategy.get("device_map", ())

        return {
            "gpu_hot_max": hot,
            "gpu_bf16_max": bf16,
            "cpu_cache_max": cpu_max,
            "kv_offload_threshold_gb": strategy.get("kv_offload_gb", vram - 6),
            "triton_blocks": cls._pick_triton_blocks(
                hw.sm_count, strategy.get("triton_preset", "auto")),
            "cublas_max_tokens": strategy.get("cublas_max_tokens", 8),
            "devices": devices,
            "device_map": device_map,
            "prefetch_enabled": strategy.get("prefetch", True),
            "mtp_enabled": strategy.get("mtp", False),
            "mtp_num_draft": strategy.get("mtp_draft", 2),
            "gpu_name": hw.gpu_name,
            "vram_total_gb": hw.vram_total_gb,
            "mem_bw_gb_s": hw.mem_bw_gb_s,
            "sm_count": hw.sm_count,
        }

    @staticmethod
    def _auto_device_map(n_layers: int, devices: tuple[str, ...]) -> tuple[int, ...]:
        """自动均分层到各 GPU。"""
        per_device = (n_layers + len(devices) - 1) // len(devices)
        mapping = []
        for dev_idx in range(len(devices)):
            start = dev_idx * per_device
            end = min(start + per_device, n_layers)
            mapping.extend([dev_idx] * (end - start))
        return tuple(mapping)

    @staticmethod
    def _pick_triton_blocks(sm_count: int, preset: str | tuple) -> tuple[int, int, int]:
        if preset != "auto":
            return preset
        if sm_count >= 100:
            return (16, 32, 64)
        elif sm_count >= 70:
            return (16, 32, 32)
        else:
            return (16, 16, 32)

    @classmethod
    def _validate(cls, params: dict, hw: HWProfile,
                  cfg: DeepSeekV4FlashConfig) -> dict:
        """验证字段合法性, 修正不可行值。"""
        params = dict(params)
        vram = max(hw.vram_free_gb, 8.0)
        per_exp_bf16 = 48.0 / 1024

        hot = params.get("gpu_hot_max", 64)
        hot_gb = hot * per_exp_bf16
        if hot_gb > vram * 0.4:
            hot = int(vram * 0.4 / per_exp_bf16)
        params["gpu_hot_max"] = hot

        bf16 = params.get("gpu_bf16_max", 100)
        total_gb = (hot * per_exp_bf16) + (bf16 * per_exp_bf16)
        if total_gb > vram * 0.85:
            bf16 = int((vram * 0.85 - hot * per_exp_bf16) / per_exp_bf16)
        params["gpu_bf16_max"] = max(16, bf16)

        offload = params.get("kv_offload_threshold_gb", vram - 6)
        if offload > vram - 2:
            offload = vram - 2
        params["kv_offload_threshold_gb"] = offload

        dm = params.get("device_map", ())
        devs = params.get("devices", ("cuda:0",))
        n_layers = cfg.num_hidden_layers
        if dm and len(dm) != n_layers:
            dm = cls._auto_device_map(n_layers, devs)
        elif not dm:
            dm = tuple([0] * n_layers)
        params["device_map"] = dm

        return params

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
