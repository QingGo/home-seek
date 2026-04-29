"""并行后端抽象 —— PP (Pipeline Parallel), 预留 EP/TP 扩展点。"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any

import torch

from home_seek.hardware_config import HardwareConfig


@dataclass
class PerDeviceState:
    """每个 GPU 的独立状态容器。"""
    device: str
    gpu_hot_experts: dict[tuple[int, int], tuple[torch.Tensor, ...]] = field(default_factory=dict)
    gpu_bf16_cache: OrderedDict = field(default_factory=OrderedDict)
    shared_expert_weights: dict[int, Any] = field(default_factory=dict)
    embed: torch.Tensor | None = None
    lm_head: torch.Tensor | None = None
    norm_weight: torch.Tensor | None = None


class ParallelBackend:
    """并行策略抽象基类。

    子类:
    - PPBackend: 流水线并行 (当前实现)
    - EPBackend: 专家并行 (预留)
    - TPBackend: 张量并行 (预留)
    """

    strategy: str
    devices: tuple[str, ...]
    device_map: tuple[int, ...]
    n_layers: int

    def __init__(self, hw_config: HardwareConfig, n_layers: int):
        self.strategy = getattr(hw_config, 'parallel_backend', 'pp')
        self.devices = hw_config.devices
        self.device_map = hw_config.device_map
        self.n_layers = n_layers
        self._per_device: dict[str, PerDeviceState] = {}

    # ── 设备查询 ────────────────────────────────────────

    def layer_device(self, layer_idx: int) -> str:
        """返回第 layer_idx 层所在的 device 字符串。"""
        return self.devices[self.device_map[layer_idx]]

    def first_device(self) -> str:
        return self.devices[0]

    def last_device(self) -> str:
        return self.devices[self.device_map[self.n_layers - 1]]

    def device_index(self, layer_idx: int) -> int:
        return self.device_map[layer_idx]

    def n_gpu(self) -> int:
        return len(self.devices)

    def is_multigpu(self) -> bool:
        return self.n_gpu() > 1

    # ── Per-GPU 状态 ────────────────────────────────────

    def get_device_state(self, device: str) -> PerDeviceState:
        if device not in self._per_device:
            self._per_device[device] = PerDeviceState(device=device)
        return self._per_device[device]

    def all_device_states(self) -> list[PerDeviceState]:
        return [self.get_device_state(d) for d in self.devices]

    # ── Per-GPU 专家缓存快捷访问 ─────────────────────
    def gpu_hot_experts(self, layer_idx: int) -> dict:
        return self.get_device_state(self.layer_device(layer_idx)).gpu_hot_experts

    def gpu_bf16_cache(self, layer_idx: int) -> OrderedDict:
        return self.get_device_state(self.layer_device(layer_idx)).gpu_bf16_cache

    def shared_expert_cache(self, layer_idx: int) -> dict:
        return self.get_device_state(self.layer_device(layer_idx)).shared_expert_weights

    # ── 通信原语 (预留 EP/TP) ──────────────────────────

    def hidden_start_device(self) -> str:
        """embedding 和设备初始 hidden state 所在的 device。"""
        return self.first_device()

    def hidden_end_device(self) -> str:
        """lm_head / final norm 需要复制的 device。"""
        return self.last_device()

    def transfer_hidden(self, h: torch.Tensor, target_device: str) -> torch.Tensor:
        """将 hidden state 搬运到目标 device (PP)."""
        current = str(h.device)
        if current != target_device:
            h = h.to(target_device, non_blocking=True)
            torch.cuda.synchronize(h.device)
        return h

    # ── 专家路由 (PP vs EP 不同) ──────────────────────

    def resolve_expert_device(self, layer_idx: int, eid: int) -> str:
        """专家权重应该在哪个 device 上。

        PP: 专家在所属层的 device 上.
        EP: 专家按 eid 分配, 与层无关 (预留).
        TP: 专家权重被切分到所有 device (预留).
        """
        return self.layer_device(layer_idx)

    # ── Factory ──────────────────────────────────────────

    @classmethod
    def from_config(cls, hw_config: HardwareConfig, n_layers: int) -> ParallelBackend:
        strategy = getattr(hw_config, 'parallel_backend', 'pp')
        if strategy == 'pp':
            return PPBackend(hw_config, n_layers)
        if strategy == 'ep':
            raise NotImplementedError("EP backend not yet implemented")
        if strategy == 'tp':
            raise NotImplementedError("TP backend not yet implemented")
        return PPBackend(hw_config, n_layers)


class PPBackend(ParallelBackend):
    """Pipeline Parallel: 层均分到各 GPU, hidden state 串行传递。

    通信模式: 单向 point-to-point, 每步传输 ~2 MB (4×4096×bf16).
    PCIe 友好 (小数据量, 流式), NVLink 也兼容.
    """

    strategy: str = "pp"


# ── 预留 EP/TP 桩 ──────────────────────────────────────

class EPBackend(ParallelBackend):
    """Expert Parallel: 专家池均分到各 GPU, all-to-all 路由. (预留)"""
    strategy: str = "ep"

    def resolve_expert_device(self, layer_idx: int, eid: int) -> str:
        # 按 eid 哈希分配: eid % n_gpu
        return self.devices[eid % self.n_gpu()]


class TPBackend(ParallelBackend):
    """Tensor Parallel: 每层权重切分, all-reduce 聚合. (预留)"""
    strategy: str = "tp"
