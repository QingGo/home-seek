"""并行后端抽象 —— PP (Pipeline Parallel), EP (Expert Parallel)."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any, Callable

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
    layer_weights: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    """EP: 该 GPU 上所有层的 attention/MHC 权重 (EP 下 GPU1 不需要)"""


class ParallelBackend:
    """并行策略抽象基类。

    子类:
    - PPBackend: 流水线并行 (当前实现)
    - EPBackend: 专家并行
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
        # EP: per-GPU streams for overlapped execution
        self._ep_streams: dict[str, torch.cuda.Stream] = {}
        self._ep_events: dict[str, torch.cuda.Event] = {}

    # ── 设备查询 ────────────────────────────────────────

    def layer_device(self, layer_idx: int) -> str:
        """返回第 layer_idx 层所在的 device 字符串。
        PP: 层均分; EP: 所有层在 first_device (GPU0)."""
        if self.strategy == 'pp':
            return self.devices[self.device_map[layer_idx]]
        return self.devices[0]

    def first_device(self) -> str:
        return self.devices[0]

    def last_device(self) -> str:
        if self.strategy == 'ep':
            return self.devices[0]
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

    # ── 通信原语 ──────────────────────────────────────────

    def hidden_start_device(self) -> str:
        """embedding 和设备初始 hidden state 所在的 device。"""
        return self.first_device()

    def hidden_end_device(self) -> str:
        """lm_head / final norm 需要复制的 device。"""
        return self.last_device()

    def transfer_hidden(self, h: torch.Tensor, target_device: str) -> torch.Tensor:
        """将 hidden state 搬运到目标 device."""
        current = str(h.device)
        if current != target_device:
            h = h.to(target_device, non_blocking=True)
            torch.cuda.synchronize(h.device)
        return h

    # ── 专家路由 ──────────────────────────────────────

    def resolve_expert_device(self, layer_idx: int, eid: int) -> str:
        """专家权重应该在哪个 device 上。
        PP: 专家在所属层的 device; EP: 按 eid 哈希分配."""
        if self.strategy == 'ep':
            return self.devices[eid % self.n_gpu()]
        return self.layer_device(layer_idx)

    # ── EP 跨设备通信 ─────────────────────────────────

    def ep_send_hidden(self, h: torch.Tensor, target_dev: str) -> tuple[torch.Tensor, torch.cuda.Event]:
        """异步传递 hidden state 到目标 GPU (用于 EP 每层前传)."""
        s0 = torch.cuda.current_stream(self.devices[0])
        h_t = h.to(target_dev, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(s0)
        return h_t, ev

    def ep_launch_gpu1(self, fn: Callable, h_t: torch.Tensor, wait_ev: torch.cuda.Event) -> torch.cuda.Event:
        """在 GPU1 上启动 expert 计算, 等 h_t 到达后执行."""
        if self.strategy != 'ep' or len(self.devices) < 2:
            done_ev = torch.cuda.Event()
            done_ev.record()
            return done_ev
        dev1 = self.devices[1]
        s1 = self._get_ep_stream(dev1)
        _prev = torch.cuda.current_device()
        try:
            with torch.cuda.device(dev1):
                with torch.cuda.stream(s1):
                    s1.wait_event(wait_ev)
                    fn(h_t)
            done_ev = torch.cuda.Event()
            done_ev.record(s1)
        finally:
            torch.cuda.set_device(_prev)
        return done_ev

    def ep_recv_result(self, result_tensor: torch.Tensor, wait_ev: torch.cuda.Event) -> torch.Tensor:
        """等 GPU1 完成, 把结果传回 GPU0."""
        if result_tensor is None or result_tensor.numel() == 0:
            s0 = torch.cuda.current_stream(self.devices[0])
            s0.wait_event(wait_ev)
            return torch.empty(0, device=self.devices[0])
        s0 = torch.cuda.current_stream(self.devices[0])
        s0.wait_event(wait_ev)
        return result_tensor.to(self.devices[0], non_blocking=False)

    def _get_ep_stream(self, device: str) -> torch.cuda.Stream:
        if device not in self._ep_streams:
            self._ep_streams[device] = torch.cuda.Stream(device)
        return self._ep_streams[device]

    # ── Factory ──────────────────────────────────────────

    @classmethod
    def from_config(cls, hw_config: HardwareConfig, n_layers: int) -> ParallelBackend:
        strategy = getattr(hw_config, 'parallel_backend', 'pp')
        if strategy == 'pp':
            return PPBackend(hw_config, n_layers)
        if strategy == 'ep':
            return EPBackend(hw_config, n_layers)
        raise ValueError(f"Unknown parallel backend: {strategy}")


class PPBackend(ParallelBackend):
    """Pipeline Parallel: 层均分到各 GPU, hidden state 串行传递。"""
    strategy: str = "pp"


class EPBackend(ParallelBackend):
    """Expert Parallel: 专家池均分到各 GPU.

    GPU0: MHC + Attention + 路由 + expert(eid%2==0) + shared expert
    GPU1: expert(eid%2==1)
    每层 GPU0→GPU1 传 h_pre, GPU1→GPU0 传部分 FFN 结果.
    两张卡通过 Stream/Event 实现计算重叠.
    """
    strategy: str = "ep"

    def resolve_expert_device(self, layer_idx: int, eid: int) -> str:
        return self.devices[eid % self.n_gpu()]


class TPBackend(ParallelBackend):
    """Tensor Parallel: 每层权重切分, all-reduce 聚合. (预留)"""
    strategy: str = "tp"
