"""
GPU 拓扑探测 —— P2P matrix, NUMA map, 互联带宽, interconnect tier 分类。

被 hw_profile.py:probe_hardware 在多 GPU 环境下调用。
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

import torch

_logger = logging.getLogger(__name__)


def probe_topology() -> dict[str, Any]:
    """全面探测 GPU 互联拓扑。返回 dict 供 HWProfile 填充。"""
    n = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if n <= 1:
        return {"interconnect_tier": "single"}

    result: dict[str, Any] = {}

    # 1. P2P matrix
    p2p: dict[str, bool] = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                key = f"{i}→{j}"
                p2p[key] = torch.cuda.can_device_access_peer(i, j)
    result["p2p_matrix"] = p2p

    # 2. NUMA map
    numa = _parse_nvidia_smi_topo()
    result["numa_map"] = numa

    # 3. CPU→GPU bandwidth per device
    cpu_bw = _probe_cpu_to_gpu_bw(n)
    result["cpu_to_gpu_bw_gb_s"] = cpu_bw

    # 4. P2P bandwidth (only if P2P available)
    p2p_bw: dict[str, float] = {}
    for i in range(n):
        for j in range(n):
            if i != j and p2p.get(f"{i}→{j}", False):
                p2p_bw[f"{i}→{j}"] = _probe_gpu_to_gpu_bw(i, j)
    result["p2p_bw_gb_s"] = p2p_bw

    # 5. Classify interconnect tier
    tier = _classify_interconnect(n, p2p, numa, p2p_bw)
    result["interconnect_tier"] = tier

    _logger.info(
        f"Topology: {n} GPUs, P2P={sum(p2p.values())}/{n*(n-1)} links, "
        f"NUMA nodes={len(set(numa.values())) if numa else 1}, "
        f"tier={tier}"
    )
    if cpu_bw:
        _logger.info(f"  CPU→GPU BW (GB/s): {[f'{b:.1f}' for b in cpu_bw]}")
    if p2p_bw:
        _logger.info(f"  P2P BW (GB/s): {dict((k,f'{v:.1f}') for k,v in p2p_bw.items())}")

    return result


# ── 互联等级定义 ─────────────────────────────────────

# nvlink:      NVLink/NVSwitch (P2P + >20 GB/s)
# pcie_p2p:    同 NUMA + P2P 可达 (PCIe 同一根)
# pcie_numa:   同 NUMA 但无 P2P (PCIe switch/PLX)
# numa_remote: 跨 NUMA 节点, 无 P2P (经 QPI/UPI)
# single:      单卡
TIER_NVLINK = "nvlink"
TIER_PCIE_P2P = "pcie_p2p"
TIER_PCIE_NUMA = "pcie_numa"
TIER_NUMA_REMOTE = "numa_remote"
TIER_SINGLE = "single"


def _classify_interconnect(
    n: int,
    p2p: dict[str, bool],
    numa_map: dict[int, int],
    p2p_bw: dict[str, float],
) -> str:
    """根据拓扑特征分类互联等级。"""
    if n <= 1:
        return TIER_SINGLE

    has_any_p2p = any(p2p.values())
    max_p2p_bw = max(p2p_bw.values()) if p2p_bw else 0.0
    unique_numa = len(set(numa_map.values())) if numa_map else 1
    all_same_numa = unique_numa <= 1

    if has_any_p2p and all_same_numa:
        # NVLink: P2P 带宽显著高于 PCIe 4.0 x16 (~32 GB/s)
        if max_p2p_bw > 20:
            return TIER_NVLINK
        return TIER_PCIE_P2P

    if all_same_numa:
        return TIER_PCIE_NUMA

    return TIER_NUMA_REMOTE


# ── NUMA 探测 ──────────────────────────────────────


def _parse_nvidia_smi_topo() -> dict[int, int]:
    """从 nvidia-smi topo -m 解析 NUMA map。

    Returns:
        dict[gpu_idx, numa_node]  例如 {0: 0, 1: 1}
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}

    numa: dict[int, int] = {}
    for line in out.splitlines():
        if not line.startswith("GPU"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            gpu_idx = int(parts[0][3:])  # "GPU0" → 0
            # parts[-1] is "GPU NUMA ID"(N/A), parts[-2] is "NUMA Affinity"
            numa_str = parts[-2]
            node = int(numa_str) if numa_str.lstrip("-").isdigit() else -1
            numa[gpu_idx] = node
        except (ValueError, IndexError):
            pass
    return numa


# ── 带宽探测 ──────────────────────────────────────#


def _probe_cpu_to_gpu_bw(n: int, size_mb: int = 64) -> list[float]:
    """探测每个 GPU 的 CPU→HtoD 带宽。"""
    bws: list[float] = []
    size = size_mb * 1024 * 1024 // 4  # float32 elements
    cpu_data = torch.randn(size, 1, pin_memory=True)

    for dev in range(n):
        try:
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _ = cpu_data.to(f"cuda:{dev}", non_blocking=False)
            torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            bw = (size_mb / 1024) / dt if dt > 0 else 0.0
            bws.append(bw)
        except Exception:
            bws.append(0.0)
    return bws


def _probe_gpu_to_gpu_bw(src: int, dst: int, size_mb: int = 64) -> float:
    """探测 GPU src → GPU dst 的 P2P 带宽。（需要 P2P 已 enabled）"""
    size = size_mb * 1024 * 1024 // 4
    try:
        src_t = torch.randn(size, 1, device=f"cuda:{src}")
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        t0 = time.perf_counter()
        _ = src_t.to(f"cuda:{dst}", non_blocking=False)
        torch.cuda.synchronize(dst)
        dt = time.perf_counter() - t0
        return (size_mb / 1024) / dt if dt > 0 else 0.0
    except Exception:
        return 0.0


# ── NUMA 线程绑定 ────────────────────────────────────


def _parse_cpulist(cpulist: str) -> list[int]:
    """解析 /sys/.../cpulist 格式, 如 "0-3,8-11" → [0,1,2,3,8,9,10,11]"""
    cpus: list[int] = []
    for part in cpulist.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def bind_thread_to_numa(numa_node: int) -> bool:
    """将当前线程绑定到指定 NUMA node 的 CPU 核心。

    通过读取 /sys/devices/system/node/node{numa_node}/cpulist
    获取该 node 的 CPU 列表，然后调用 os.sched_setaffinity 绑定。

    Returns:
        True 绑定成功, False 失败（NUMA node 不存在或权限不足）。
    """
    import os

    path = f"/sys/devices/system/node/node{numa_node}/cpulist"
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            cpulist = _parse_cpulist(f.read().strip())
        if not cpulist:
            return False
        os.sched_setaffinity(0, cpulist)
        return True
    except Exception:
        return False
