import os
import json
import time
import torch
from dataclasses import dataclass, field, asdict


@dataclass
class HWProfile:
    disk_bw_gb_s: float = 0.0
    pcie_bw_gb_s: float = 0.0
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    sm_count: int = 0
    mem_bw_gb_s: float = 0.0
    decode_matmul_us: float = 0.0
    gpu_name: str = ""
    n_gpu: int = 1
    probe_timestamp: float = 0.0

    # ── Phase 1: 拓扑感知字段 (topology_prober 填充) ─────
    interconnect_tier: str = "single"
    """互联等级: single|pcie_numa|numa_remote|pcie_p2p|nvlink"""

    numa_map: dict[int, int] = field(default_factory=dict)
    """gpu_idx → NUMA node id。例如 {0: 0, 1: 1}"""

    p2p_matrix: dict[str, bool] = field(default_factory=dict)
    """"i→j" → 是否支持 P2P 访问。例如 {"0→1": False, "1→0": False}"""

    cpu_to_gpu_bw_gb_s: list[float] = field(default_factory=list)
    """per-GPU CPU→GPU 实测带宽"""

    p2p_bw_gb_s: dict[str, float] = field(default_factory=dict)
    """"i→j" → P2P 实测带宽 GB/s"""

    per_device_vram_gb: list[float] = field(default_factory=list)
    """每个 GPU 的 VRAM (GiB)"""

    per_device_sm: list[int] = field(default_factory=list)
    """每个 GPU 的 SM 数"""


_PROFILE_PATH = os.environ.get("HW_PROFILE_PATH", "hw_profile.json")


def _probe_disk_bw(path: str = "/dev/md0", size_mb: int = 1024) -> float:
    try:
        data = os.urandom(size_mb * 1024 * 1024)
        tmp = path if os.path.exists(path) else "/tmp"
        fpath = os.path.join(tmp, ".hseek_bw_test")
        with open(fpath, "wb") as f:
            f.write(data)
        t0 = time.perf_counter()
        with open(fpath, "rb") as f:
            while f.read(16 * 1024 * 1024):
                pass
        t1 = time.perf_counter()
        os.remove(fpath)
        return size_mb / (t1 - t0) / 1024
    except Exception:
        return 1.5


def _probe_pcie_bw() -> float:
    try:
        cpu = torch.randn(256, 4096, dtype=torch.bfloat16, device="cpu").pin_memory()
        t0 = time.perf_counter()
        cpu.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        size_gb = 256 * 4096 * 2 / (1024**3)
        return size_gb / (t1 - t0)
    except Exception:
        return 6.6


def _probe_gpu() -> tuple[int, float, str, float]:
    try:
        props = torch.cuda.get_device_properties("cuda")
        sm = props.multi_processor_count
        vram_total = props.total_memory / (1024**3)
        name = props.name
        mem_bw = props.memory_bandwidth / 1e9 if hasattr(props, "memory_bandwidth") else 404.0
        free = vram_total - (torch.cuda.memory_allocated() / (1024**3)) - 0.5
        return sm, vram_total, name, mem_bw, free
    except Exception:
        return 0, 0.0, "", 0.0, 0.0


def _probe_all_gpus() -> tuple[list[float], list[int], list[str]]:
    """探测所有 GPU 的 VRAM/SM/名称，替代只探 device 0。"""
    vrams, sms, names = [], [], []
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for i in range(n):
        try:
            props = torch.cuda.get_device_properties(i)
            vrams.append(props.total_memory / (1024**3))
            sms.append(props.multi_processor_count)
            names.append(props.name)
        except Exception:
            vrams.append(0.0)
            sms.append(0)
            names.append("")
    return vrams, sms, names


def _probe_matmul_us() -> float:
    try:
        a = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            a @ b.T
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) / 100 * 1e6
    except Exception:
        return 410.0


def probe_hardware(force: bool = False, weight_dir: str = "weights") -> HWProfile:
    if not force and os.path.exists(_PROFILE_PATH):
        with open(_PROFILE_PATH) as f:
            raw = json.load(f)
        return HWProfile(**raw)
    sm, vram_total, gpu_name, mem_bw, vram_free = _probe_gpu()
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1

    vram_list, sm_list, name_list = _probe_all_gpus()

    profile = HWProfile(
        disk_bw_gb_s=_probe_disk_bw(weight_dir.rstrip("/weights").rstrip("/") or "/dev/md0"),
        pcie_bw_gb_s=_probe_pcie_bw(),
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        sm_count=sm,
        mem_bw_gb_s=mem_bw,
        decode_matmul_us=_probe_matmul_us(),
        gpu_name=gpu_name,
        n_gpu=n_gpu,
        probe_timestamp=time.time(),
        # 拓扑字段：延迟惰性填充（避免 probe_hardware 耦合 torch.cuda.device 切换）
        interconnect_tier="single" if n_gpu <= 1 else "unknown",
        per_device_vram_gb=vram_list,
        per_device_sm=sm_list,
    )

    # 多 GPU: 执行拓扑探测
    if n_gpu > 1:
        try:
            from home_seek.topology_prober import probe_topology
            topo = probe_topology()
            profile.interconnect_tier = topo.get("interconnect_tier", "pcie_numa")
            profile.numa_map = topo.get("numa_map", {})
            profile.p2p_matrix = topo.get("p2p_matrix", {})
            profile.cpu_to_gpu_bw_gb_s = topo.get("cpu_to_gpu_bw_gb_s", [])
            profile.p2p_bw_gb_s = topo.get("p2p_bw_gb_s", {})
        except Exception as exc:
            print(f"[hw_profile] Topology probe failed: {exc}, using defaults")

    if n_gpu > 1:
        print(f"[hw_profile] Detected {n_gpu} × {gpu_name}  tier={profile.interconnect_tier}")
    with open(_PROFILE_PATH, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return profile


def load_profile(path: str | None = None) -> HWProfile | None:
    p = path or _PROFILE_PATH
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return HWProfile(**json.load(f))
