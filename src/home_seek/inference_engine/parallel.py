"""并行后端抽象 —— PP (Pipeline Parallel), EP (Expert Parallel)."""

from __future__ import annotations

import queue
import threading
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


@dataclass
class _EPWorkItem:
    """单个 EP worker 工作项。"""
    hidden: torch.Tensor | None = None
    topk_idx: torch.Tensor | None = None
    topk_weights: torch.Tensor | None = None
    layer_idx: int = -1
    copy_event: torch.cuda.Event | None = None
    done: threading.Event = field(default_factory=threading.Event)
    result: torch.Tensor | None = None
    exception: Exception | None = None
    gpu1_ev_start: torch.cuda.Event | None = None
    gpu1_ev_end: torch.cuda.Event | None = None
    gpu1_elapsed_ms: float = 0.0


class _EPWorker(threading.Thread):
    """GPU1 专用工作线程: 接收 EP 工作项, 在 GPU1 上执行 FusedMoE.

    P2: 支持 NUMA 绑定 (numa_node≥0), 在 run() 开始时绑定到指定
    NUMA node, 确保后续 CPU 内存分配 (mmap page fault, torch tensor)
    落在此 NUMA node 的本地内存上, 减少 GPU1 PCIe 跨 NUMA 读取延迟。
    """

    def __init__(self, engine, device: str, numa_node: int = -1):
        super().__init__(daemon=True)
        self.engine = engine
        self.device = device
        self.device_index = int(device.split(":")[1])
        self._numa_node = numa_node
        self._work_queue: queue.Queue[_EPWorkItem] = queue.Queue()
        self._shutdown_event = threading.Event()

    def submit(self, item: _EPWorkItem) -> None:
        self._work_queue.put(item)

    def stop(self) -> None:
        self._shutdown_event.set()

    def run(self) -> None:
        # P2: NUMA 绑定 (必须在 torch.cuda.set_device 之前)
        if self._numa_node >= 0:
            try:
                from home_seek.topology_prober import bind_thread_to_numa
                bound = bind_thread_to_numa(self._numa_node)
                if bound:
                    import logging
                    logging.getLogger(__name__).info(
                        f"  EP worker bound to NUMA node {self._numa_node}")
            except Exception:
                pass
        torch.cuda.set_device(self.device_index)
        while not self._shutdown_event.is_set():
            try:
                item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # 等 async copy 完成 (GPU1 stream wait GPU0 event)
                if item.copy_event is not None:
                    torch.cuda.current_stream(self.device).wait_event(item.copy_event)

                # GPU1 timing events
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                item.gpu1_ev_start = ev_start
                item.gpu1_ev_end = ev_end

                D = self.engine.config.hidden_size
                h_flat = item.hidden.reshape(-1, D)

                def _load_expert(layer, eid):
                    return self.engine._load_bf16_deq(layer, eid)

                ev_start.record()
                r = self.engine._fused_moe.forward(
                    h_flat, item.topk_idx, item.topk_weights,
                    _load_expert,
                    item.layer_idx,
                )
                ev_end.record()
                item.result = r
            except Exception:
                # fallback: 逐 expert 循环
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                item.gpu1_ev_start = ev_start
                item.gpu1_ev_end = ev_end
                ev_start.record()
                try:
                    D = self.engine.config.hidden_size
                    total_tokens = item.hidden.shape[0]
                    h_flat = item.hidden.reshape(-1, D)
                    r = torch.zeros(total_tokens, D, device=self.device, dtype=item.hidden.dtype)
                    for k in range(item.topk_idx.shape[1]):
                        eids_k = item.topk_idx[:, k]
                        w_k = item.topk_weights[:, k]
                        for tok_i in range(total_tokens):
                            eid = int(eids_k[tok_i].item())
                            if eid < 0:
                                continue
                            bf16 = self.engine._load_bf16_deq(item.layer_idx, eid)
                            if bf16 is None:
                                continue
                            w1_d, w3_d, w2_d = bf16
                            h_tok = h_flat[tok_i:tok_i + 1].to(w1_d.dtype)
                            gate_out = torch.matmul(h_tok, w1_d.t())
                            up_out = torch.matmul(h_tok, w3_d.t())
                            g = gate_out.float().clamp(max=self.engine.config.swiglu_limit)
                            u = up_out.float().clamp(min=-self.engine.config.swiglu_limit,
                                                     max=self.engine.config.swiglu_limit)
                            activated = (g * g.sigmoid() * u).to(w1_d.dtype)
                            out = torch.matmul(activated.to(w2_d.dtype), w2_d.t())
                            r[tok_i] += out[0] * w_k[tok_i]
                    ev_end.record()
                    item.result = r
                except Exception as e2:
                    item.exception = e2
            finally:
                # Calculate GPU1 elapsed time from CUDA events
                ev_s = getattr(item, 'gpu1_ev_start', None)
                ev_e = getattr(item, 'gpu1_ev_end', None)
                if ev_s is not None and ev_e is not None:
                    try:
                        ev_e.synchronize()
                        item.gpu1_elapsed_ms = ev_s.elapsed_time(ev_e)
                    except Exception:
                        pass
                item.done.set()


class EPBackend(ParallelBackend):
    """Expert Parallel: 专家池均分到各 GPU.

    GPU0: MHC + Attention + 路由 + expert(eid%2==0) + shared expert
    GPU1: expert(eid%2==1)

    V21.5: 每个 GPU 一个 Python 线程, 实现真正计算并行.
    """

    strategy: str = "ep"
    _ep_worker: _EPWorker | None = None

    def resolve_expert_device(self, layer_idx: int, eid: int) -> str:
        return self.devices[eid % self.n_gpu()]

    def ep_start_worker(self, engine) -> None:
        """启动 GPU1 工作线程 (惰性, 首次 EP 前传时调用).

        P2: 当 engine 的 ep_numa_aware=True 时, 从 hw_profile.numa_map
        获取 GPU1 对应的 NUMA node, 传给 _EPWorker 进行线程绑定。
        """
        if self._ep_worker is not None:
            return
        if self.n_gpu() < 2:
            return
        numa_node = -1
        hw_config = getattr(engine, 'hw_config', None)
        hw_profile = getattr(engine, 'hw_profile', None)
        if (hw_config is not None and hw_profile is not None
                and getattr(hw_config, 'ep_numa_aware', False)):
            numa_map = getattr(hw_profile, 'numa_map', {})
            gpu1_idx = int(self.devices[1].split(":")[1])
            numa_node = numa_map.get(gpu1_idx, -1)
        worker = _EPWorker(engine, self.devices[1], numa_node=numa_node)
        worker.start()
        self._ep_worker = worker

    def ep_submit_work(self, hidden: torch.Tensor,
                       topk_idx: torch.Tensor,
                       topk_weights: torch.Tensor,
                       layer_idx: int,
                       copy_event: torch.cuda.Event | None = None) -> _EPWorkItem:
        """提交 GPU1 expert 计算任务, 立即返回 work item."""
        item = _EPWorkItem(
            hidden=hidden,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            layer_idx=layer_idx,
            copy_event=copy_event,
        )
        if self._ep_worker is not None:
            self._ep_worker.submit(item)
        return item

    def ep_wait_result(self, item: _EPWorkItem) -> torch.Tensor:
        """等 GPU1 完成, 返回结果 tensor."""
        if item is None:
            return torch.empty(0, device=self.devices[0])
        if self._ep_worker is not None:
            item.done.wait()
            if item.exception is not None:
                raise item.exception
            if item.result is not None:
                return item.result.to(self.devices[0], non_blocking=False)
        return torch.empty(0, device=self.devices[0])


class TPBackend(ParallelBackend):
    """Tensor Parallel: 每层权重切分, all-reduce 聚合. (预留)"""
    strategy: str = "tp"
