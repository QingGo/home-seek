import os
import json
import time
import torch
from dataclasses import dataclass, asdict


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
    )
    if n_gpu > 1:
        print(f"[hw_profile] Detected {n_gpu} × {gpu_name}")
    with open(_PROFILE_PATH, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return profile


def load_profile(path: str | None = None) -> HWProfile | None:
    p = path or _PROFILE_PATH
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return HWProfile(**json.load(f))
