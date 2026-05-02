"""
End-to-end profiling script for HomeSeekInferenceEngine.

Usage:
    python -m home_seek.profiling_runner --prompt "介绍一下你自己" --max-tokens 50

Metrics collected:
    - Per-layer timing (attention, FFN, MHC, total)
    - GPU memory allocation trace
    - Expert cache hit/miss rates
    - GPU FP4 store activity
    - CPU utilization (%)
    - GPU compute + memory utilization (%)
    - Prefill vs decode breakdown
    - Token throughput (tokens/s, ms/token)
    - Generated text
"""

import os
import json
import time
import argparse
import threading
import logging
from collections import defaultdict

import torch

from home_seek.inference_engine import HomeSeekInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


class UtilMonitor:
    def __init__(self):
        self._stop = threading.Event()
        self._samples = []
        self._thread = None
        self._has_pynvml = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            self._has_pynvml = True
        except Exception:
            self._handle = None
            self._pynvml = None

    def _sample_loop(self):
        import psutil
        proc = psutil.Process()
        while not self._stop.is_set():
            cpu = proc.cpu_percent(interval=0)
            gpu_util = -1.0
            gpu_mem = -1.0
            gpu_temp = -1.0
            if self._has_pynvml:
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    temp = self._pynvml.nvmlDeviceGetTemperature(
                        self._handle, self._pynvml.NVML_TEMPERATURE_GPU)
                    gpu_util = util.gpu
                    gpu_mem = mem.used / mem.total * 100
                    gpu_temp = temp
                except Exception:
                    pass
            self._samples.append({
                "time": time.time(),
                "cpu_percent": cpu,
                "gpu_util_percent": gpu_util,
                "gpu_mem_percent": gpu_mem,
                "gpu_temp_c": gpu_temp,
            })
            self._stop.wait(0.5)

    def start(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            self._thread = None

    def summary(self):
        if not self._samples:
            return "UtilMonitor: no samples collected"

        cpu_vals = [s["cpu_percent"] for s in self._samples]
        gpu_vals = [s["gpu_util_percent"] for s in self._samples if s["gpu_util_percent"] >= 0]
        gpu_mem_vals = [s["gpu_mem_percent"] for s in self._samples if s["gpu_mem_percent"] >= 0]
        gpu_temp_vals = [s["gpu_temp_c"] for s in self._samples if s["gpu_temp_c"] >= 0]

        def stats(vals):
            if not vals:
                return (0, 0, 0)
            return (min(vals), sum(vals) / len(vals), max(vals))

        c_min, c_avg, c_max = stats(cpu_vals)
        g_min, g_avg, g_max = stats(gpu_vals)
        gm_min, gm_avg, gm_max = stats(gpu_mem_vals)
        gt_min, gt_avg, gt_max = stats(gpu_temp_vals)

        lines = [
            "Resource Utilization (sampled during inference):",
            f"  CPU:      avg={c_avg:.1f}%  min={c_min:.1f}%  max={c_max:.1f}%",
        ]
        if gpu_vals:
            idle_pct = sum(1 for v in gpu_vals if v < 30) / len(gpu_vals) * 100
            lines.append(
                f"  GPU Compute: avg={g_avg:.1f}%  min={g_min:.1f}%  "
                f"max={g_max:.1f}%  idle_time={idle_pct:.0f}%"
            )
            lines.append(f"  GPU Memory:  avg={gm_avg:.1f}%  min={gm_min:.1f}%  max={gm_max:.1f}%")
            lines.append(f"  GPU Temp:    avg={gt_avg:.0f}°C  max={gt_max:.0f}°C")
            if g_avg < 50:
                lines.append(f"  >> GPU计算利用率偏低({g_avg:.0f}%)，说明瓶颈在数据传输或CPU调度而非计算")
        return "\n".join(lines)

    def gpu_idle_ratio(self):
        gpu_vals = [s["gpu_util_percent"] for s in self._samples if s["gpu_util_percent"] >= 0]
        if not gpu_vals:
            return -1
        return sum(1 for v in gpu_vals if v < 30) / len(gpu_vals)


class PerLayerTimer:
    def __init__(self):
        self.events = defaultdict(list)

    def record(self, stage: str, layer: int, duration_ms: float):
        self.events[stage].append(duration_ms)
        if layer >= 0:
            key = f"{stage}_layer{layer}"
        else:
            key = stage
        self.events[key].append(duration_ms)

    def snapshot(self) -> dict:
        """Return copy of current counts per stage."""
        return {k: len(v) for k, v in self.events.items()}

    @staticmethod
    def timer_snapshot(timer: 'PerLayerTimer') -> dict:
        return timer.snapshot()

    @staticmethod
    def delta_summary(before: dict, after: dict) -> dict:
        """Return dict of stage -> count, total_ms for the delta."""
        result = {}
        all_keys = set(before) | set(after)
        for key in all_keys:
            # only per-layer stages
            if key in ('attention', 'ffn', 'mhc_attn', 'mhc_ffn', 'mhc_post',
                       'embed', 'hc_head', 'norm', 'lm_head', 'total_layer', 'decode_step'):
                count_before = before.get(key, 0)
                count_after = after.get(key, 0)
                if count_after > count_before:
                    result[key] = {'count': count_after - count_before}
        return result

    def summary(self):
        lines = []
        lines.append(f"{'Stage':<25} {'Mean(ms)':<10} {'Total(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10} {'Count':<8}")
        lines.append("-" * 75)
        per_layer_stages = ["attention", "ffn", "mhc_attn", "mhc_ffn", "mhc_post"]
        for stage in per_layer_stages:
            vals = self.events.get(stage, [])
            if not vals:
                continue
            mn = sum(vals) / len(vals)
            tt = sum(vals)
            mi = min(vals)
            mx = max(vals)
            lines.append(f"{stage:<25} {mn:<10.2f} {tt:<10.2f} {mi:<10.2f} {mx:<10.2f} {len(vals):<8}")
        global_stages = ["embed", "hc_head", "norm", "lm_head", "total_layer", "decode_step"]
        for stage in global_stages:
            vals = self.events.get(stage, [])
            if not vals:
                continue
            mn = sum(vals) / len(vals)
            tt = sum(vals)
            mi = min(vals)
            mx = max(vals)
            lines.append(f"{stage:<25} {mn:<10.2f} {tt:<10.2f} {mi:<10.2f} {mx:<10.2f} {len(vals):<8}")
        return "\n".join(lines)


class CacheMonitor:
    def __init__(self):
        self.hot_hits = 0
        self.hot_misses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.gpu_store_hits = 0
        self.gpu_store_misses = 0
        self.gpu_per_device_hits: dict[int, int] = {}
        self.gpu_per_device_misses: dict[int, int] = {}
        self.file_loads = 0
        self.deq_times = []
        self.file_load_times = []
        self.ep_gpu0_ms = 0.0
        self.ep_gpu1_ms = 0.0
        self.ep_copy_ms = 0.0

    def snapshot(self) -> dict:
        """Return serializable snapshot for differential profiling."""
        return {
            'hot_hits': self.hot_hits,
            'hot_misses': self.hot_misses,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'gpu_store_hits': self.gpu_store_hits,
            'gpu_store_misses': self.gpu_store_misses,
            'gpu_per_device_hits': dict(self.gpu_per_device_hits),
            'gpu_per_device_misses': dict(self.gpu_per_device_misses),
            'file_loads': self.file_loads,
            'deq_n': len(self.deq_times),
            'deq_total': sum(self.deq_times),
            'file_n': len(self.file_load_times),
            'file_total': sum(self.file_load_times),
            'ep_gpu0_ms': self.ep_gpu0_ms,
            'ep_gpu1_ms': self.ep_gpu1_ms,
            'ep_copy_ms': self.ep_copy_ms,
        }

    @staticmethod
    def delta(before: dict, after: dict) -> dict:
        """Compute difference between two snapshots."""
        def _dict_diff(b, a):
            keys = set(b) | set(a)
            return {k: a.get(k, 0) - b.get(k, 0) for k in keys if a.get(k, 0) - b.get(k, 0) != 0}
        return {
            'hot_hits': after['hot_hits'] - before['hot_hits'],
            'hot_misses': after['hot_misses'] - before['hot_misses'],
            'cache_hits': after['cache_hits'] - before['cache_hits'],
            'cache_misses': after['cache_misses'] - before['cache_misses'],
            'gpu_store_hits': after['gpu_store_hits'] - before['gpu_store_hits'],
            'gpu_store_misses': after['gpu_store_misses'] - before['gpu_store_misses'],
            'gpu_per_device_hits': _dict_diff(
                before.get('gpu_per_device_hits', {}),
                after.get('gpu_per_device_hits', {}),
            ),
            'gpu_per_device_misses': _dict_diff(
                before.get('gpu_per_device_misses', {}),
                after.get('gpu_per_device_misses', {}),
            ),
            'file_loads': after['file_n'] - before['file_n'],
            'deq_n': after['deq_n'] - before['deq_n'],
            'deq_total': after['deq_total'] - before['deq_total'],
            'file_n': after['file_n'] - before['file_n'],
            'file_total': after['file_total'] - before['file_total'],
            'ep_gpu0_ms': after.get('ep_gpu0_ms', 0) - before.get('ep_gpu0_ms', 0),
            'ep_gpu1_ms': after.get('ep_gpu1_ms', 0) - before.get('ep_gpu1_ms', 0),
            'ep_copy_ms': after.get('ep_copy_ms', 0) - before.get('ep_copy_ms', 0),
        }

    def record_hot(self, hit: bool):
        if hit:
            self.hot_hits += 1
        else:
            self.hot_misses += 1

    def record_cache(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_gpu_store(self, hit: bool, device_idx: int = -1):
        if hit:
            self.gpu_store_hits += 1
            if device_idx >= 0:
                self.gpu_per_device_hits[device_idx] = self.gpu_per_device_hits.get(device_idx, 0) + 1
        else:
            self.gpu_store_misses += 1
            if device_idx >= 0:
                self.gpu_per_device_misses[device_idx] = self.gpu_per_device_misses.get(device_idx, 0) + 1

    def record_ep_timing(self, gpu0_ms: float, gpu1_ms: float, copy_ms: float):
        self.ep_gpu0_ms += gpu0_ms
        self.ep_gpu1_ms += gpu1_ms
        self.ep_copy_ms += copy_ms

    def record_file_load(self):
        self.file_loads += 1

    def record_deq_time(self, ms: float):
        self.deq_times.append(ms)

    def record_file_load_time(self, ms: float):
        self.file_load_times.append(ms)

    def summary(self, snap: dict | None = None):
        s = snap or self.snapshot()
        total_hot = s['hot_hits'] + s['hot_misses']
        hot_rate = s['hot_hits'] / total_hot * 100 if total_hot > 0 else 0
        total_cache = s['cache_hits'] + s['cache_misses']
        cache_rate = s['cache_hits'] / total_cache * 100 if total_cache > 0 else 0
        total_gpu = s['gpu_store_hits'] + s['gpu_store_misses']
        gpu_rate = s['gpu_store_hits'] / total_gpu * 100 if total_gpu > 0 else 0
        fl_n, fl_avg = s['file_n'], (s['file_total'] / s['file_n'] if s['file_n'] > 0 else 0)

        per_gpu_hits = s.get('gpu_per_device_hits', {})
        per_gpu_misses = s.get('gpu_per_device_misses', {})
        gpu_hit_lines = []
        for dev_idx in sorted(set(per_gpu_hits) | set(per_gpu_misses)):
            h = per_gpu_hits.get(dev_idx, 0)
            m = per_gpu_misses.get(dev_idx, 0)
            rate = h / (h + m) * 100 if h + m > 0 else 0
            gpu_hit_lines.append(f"GPU{dev_idx}: {h}h/{m}m ({rate:.0f}%)")

        def stats(vals):
            if not vals:
                return (0, 0, 0, 0)
            return (len(vals), min(vals), sum(vals) / len(vals), max(vals))

        deq_n, deq_min, deq_avg, deq_max = stats(self.deq_times)
        fl_n, fl_min, fl_avg, fl_max = stats(self.file_load_times)

        lines = [
            "Expert Cache Performance:",
            f"  Hot cache hits: {s['hot_hits']}, misses: {s['hot_misses']}  (hit rate: {hot_rate:.1f}%)",
            f"  Raw cache hits: {s['cache_hits']}, misses: {s['cache_misses']}  (hit rate: {cache_rate:.1f}%)",
            f"  GPU store hits: {s['gpu_store_hits']}, misses: {s['gpu_store_misses']}  (hit rate: {gpu_rate:.1f}%)",
        ]
        if gpu_hit_lines:
            lines.append(f"    Per-GPU: {', '.join(gpu_hit_lines)}")
        lines.extend([
            f"  File loads: {s['file_n']}  (avg {fl_avg:.1f}ms, total {s['file_total']/1000:.1f}s)",
            f"  Dequantize times (ms): n={deq_n}  avg={deq_avg:.2f}  min={deq_min:.2f}  max={deq_max:.2f}",
            f"  File load times (ms):  n={fl_n}  avg={fl_avg:.2f}  min={fl_min:.2f}  max={fl_max:.2f}",
        ])
        ep_gpu0 = s.get('ep_gpu0_ms', 0)
        ep_gpu1 = s.get('ep_gpu1_ms', 0)
        ep_copy = s.get('ep_copy_ms', 0)
        if ep_gpu0 > 0 or ep_gpu1 > 0:
            lines.append(
                f"  EP timing (cumulative): GPU0={ep_gpu0:.0f}ms  "
                f"GPU1={ep_gpu1:.0f}ms  PCIe_copy={ep_copy:.0f}ms"
            )
        if fl_n > 0:
            total_file_time = sum(self.file_load_times) / 1000
            lines.append(f"  Total file I/O time: {total_file_time:.2f}s")
        return "\n".join(lines)


class LayerTrace:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.attn_ms = [0.0] * num_layers
        self.ffn_ms = [0.0] * num_layers
        self.mhc_attn_ms = [0.0] * num_layers
        self.mhc_ffn_ms = [0.0] * num_layers
        self.mhc_post_ms = [0.0] * num_layers
        self.expert_ids = [[] for _ in range(num_layers)]
        self.layer_total_ms = [0.0] * num_layers

    def snapshot(self) -> dict:
        return {
            'attn_ms': self.attn_ms[:],
            'ffn_ms': self.ffn_ms[:],
            'mhc_attn_ms': self.mhc_attn_ms[:],
            'mhc_ffn_ms': self.mhc_ffn_ms[:],
            'mhc_post_ms': self.mhc_post_ms[:],
        }

    @staticmethod
    def delta(before: dict, after: dict) -> dict:
        num = len(before['attn_ms'])
        return {
            'attn_ms': [after['attn_ms'][i] - before['attn_ms'][i] for i in range(num)],
            'ffn_ms': [after['ffn_ms'][i] - before['ffn_ms'][i] for i in range(num)],
            'mhc_attn_ms': [after['mhc_attn_ms'][i] - before['mhc_attn_ms'][i] for i in range(num)],
            'mhc_ffn_ms': [after['mhc_ffn_ms'][i] - before['mhc_ffn_ms'][i] for i in range(num)],
            'mhc_post_ms': [after['mhc_post_ms'][i] - before['mhc_post_ms'][i] for i in range(num)],
        }

    def summary(self):
        lines = ["Per-Layer Breakdown:"]
        header = (
            f"{'Layer':<7} {'Attn(ms)':<10} {'FFN(ms)':<10} "
            f"{'MHC_A(ms)':<10} {'MHC_F(ms)':<10} {'Post(ms)':<10} "
            f"{'Total(ms)':<10} {'Experts'}"
        )
        lines.append(header)
        lines.append("-" * (len(header) + 20))
        for i in range(self.num_layers):
            experts_str = ",".join(str(e) for e in self.expert_ids[i][:6]) if self.expert_ids[i] else "-"
            if len(self.expert_ids[i]) > 6:
                experts_str += "..."
            t = self.attn_ms[i] + self.ffn_ms[i] + self.mhc_attn_ms[i] + self.mhc_ffn_ms[i] + self.mhc_post_ms[i]
            lines.append(
                f"{i:<7} {self.attn_ms[i]:<10.2f} {self.ffn_ms[i]:<10.2f} "
                f"{self.mhc_attn_ms[i]:<10.2f} {self.mhc_ffn_ms[i]:<10.2f} "
                f"{self.mhc_post_ms[i]:<10.2f} {t:<10.2f} {experts_str}"
            )
        total_attn = sum(self.attn_ms)
        total_ffn = sum(self.ffn_ms)
        total_mhc_a = sum(self.mhc_attn_ms)
        total_mhc_f = sum(self.mhc_ffn_ms)
        total_post = sum(self.mhc_post_ms)
        total_all = total_attn + total_ffn + total_mhc_a + total_mhc_f + total_post
        lines.append("-" * (len(header) + 20))
        lines.append(
            f"{'TOTAL':<7} {total_attn:<10.2f} {total_ffn:<10.2f} "
            f"{total_mhc_a:<10.2f} {total_mhc_f:<10.2f} "
            f"{total_post:<10.2f} {total_all:<10.2f}"
        )
        pct_ffn = total_ffn / total_all * 100 if total_all > 0 else 0
        pct_attn = total_attn / total_all * 100 if total_all > 0 else 0
        lines.append("")
        lines.append(f"FFN占比: {pct_ffn:.1f}%, Attn占比: {pct_attn:.1f}%")
        return "\n".join(lines)


def patch_engine(engine, timer, cache_mon, layer_trace, mem_trace, layer_trace_decode=None):
    original_forward_attn = engine._forward_attn
    original_forward_ffn = engine._forward_ffn
    original_process_mhc = engine._process_mhc_layer
    original_mhc_post = engine._process_mhc_post
    original_load_expert = engine._load_expert_weights
    original_expert_deq = engine._load_expert_deq
    original_expert_cache_get = engine.expert_cache.get
    original_hc_head = engine._hc_head
    original_load_fp4_raw = engine._load_expert_fp4_raw

    def traced_forward_attn(hidden_states, lw, layer_idx):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_forward_attn(hidden_states, lw, layer_idx)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        timer.record("attention", layer_idx, dt)
        layer_trace.attn_ms[layer_idx] += dt
        if layer_trace_decode is not None and getattr(engine, '_phase', None) == 'decode':
            layer_trace_decode.attn_ms[layer_idx] += dt
        mem_trace.append(("attn_end", layer_idx, torch.cuda.memory_allocated()))
        return result

    def traced_forward_ffn(hidden_states, lw, layer_idx, input_ids=None):
        torch.cuda.synchronize()
        is_ep = (
            getattr(engine, '_is_multigpu', False)
            and getattr(getattr(engine, '_backend', None), 'strategy', None) == 'ep'
        )
        t0 = time.perf_counter()
        result, used = original_forward_ffn(hidden_states, lw, layer_idx, input_ids)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        timer.record("ffn", layer_idx, dt)
        layer_trace.ffn_ms[layer_idx] += dt
        if layer_trace_decode is not None and getattr(engine, '_phase', None) == 'decode':
            layer_trace_decode.ffn_ms[layer_idx] += dt
        layer_trace.expert_ids[layer_idx] = sorted(used) if used else []
        if layer_trace_decode is not None and getattr(engine, '_phase', None) == 'decode':
            layer_trace_decode.expert_ids[layer_idx] = sorted(used) if used else []
        mem_trace.append(("ffn_end", layer_idx, torch.cuda.memory_allocated()))
        # EP timing breakdown
        if is_ep and hasattr(engine, '_ep_timing'):
            timing = engine._ep_timing
            gpu0 = timing.get('gpu0_ms', 0)
            gpu1 = timing.get('gpu1_ms', 0)
            copy_ = timing.get('copy_ms', 0)
            cache_mon.record_ep_timing(gpu0, gpu1, copy_)
        return result, used

    def traced_process_mhc(hidden_4d, lw, prefix):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_process_mhc(hidden_4d, lw, prefix)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        lines = ("mhc_attn" if "attn" in prefix else "mhc_ffn",
                 -1, dt)
        timer.record(lines[0], -1, dt)
        return result

    def traced_mhc_post(hidden, residual, post, comb):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_mhc_post(hidden, residual, post, comb)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        timer.record("mhc_post", -1, dt)
        return result

    def traced_hc_head(hidden_4d):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_hc_head(hidden_4d)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        timer.record("hc_head", -1, dt)
        return result

    def traced_load_expert(layer_idx, eid):
        cache_key = f"{layer_idx}_{eid}"
        was_cached = cache_key in engine.expert_cache.cache
        t0 = time.perf_counter()
        result = original_load_expert(layer_idx, eid)
        dt = (time.perf_counter() - t0) * 1000
        if not was_cached:
            cache_mon.record_file_load()
            cache_mon.record_file_load_time(dt)
        return result

    def traced_expert_deq(layer_idx, eid):
        key = f"{layer_idx}_{eid}"
        hit = key in engine.expert_cache._hot_deq
        cache_mon.record_hot(hit)
        t0 = time.perf_counter()
        result = original_expert_deq(layer_idx, eid)
        dt = (time.perf_counter() - t0) * 1000
        cache_mon.record_deq_time(dt)
        return result

    def traced_cache_get(key):
        hit = key in engine.expert_cache.cache
        cache_mon.record_cache(hit)
        return original_expert_cache_get(key)

    def traced_load_fp4_raw(layer_idx, eid):
        # EP: track per-GPU GPU cache hit/miss
        is_ep = (
            getattr(engine, '_is_multigpu', False)
            and getattr(getattr(engine, '_backend', None), 'strategy', None) == 'ep'
        )
        if is_ep:
            hot_key = (layer_idx, eid)
            hit_dev = -1
            for di, d in enumerate(engine._backend.devices):
                st = engine._backend.get_device_state(d)
                if hot_key in st.gpu_hot_experts or hot_key in st.gpu_bf16_cache:
                    hit_dev = di
                    break
            cache_mon.record_gpu_store(hit_dev >= 0, hit_dev)
        return original_load_fp4_raw(layer_idx, eid)

    engine._forward_attn = traced_forward_attn
    engine._forward_ffn = traced_forward_ffn
    engine._process_mhc_layer = traced_process_mhc
    engine._process_mhc_post = traced_mhc_post
    engine._hc_head = traced_hc_head
    engine._load_expert_weights = traced_load_expert
    engine._load_expert_deq = traced_expert_deq
    engine.expert_cache.get = traced_cache_get
    engine._load_expert_fp4_raw = traced_load_fp4_raw
    return engine


def _print_round_result(label, result, cache_mon_snap, layer_snap, timer_snap, num_layers, tokenizer,
                         decode_layer_delta=None):
    """Print single round's performance summary."""
    out = tokenizer.decode(result["tokens"][0], skip_special_tokens=True)
    print(f"\n  >>> {label} <<<")
    print(f"  Output: {out[:80]}")
    print()

    decode_tps = result.get('decode_tokens_per_second', 0)
    prefill_t = result.get('prefill_time_s', 0)
    decode_t = result.get('decode_time_s', 0)
    total_t = result['total_time_s']
    prompt_tok = result['num_prompt_tokens']
    decode_tok = max(1, result['num_generated_tokens'])

    print(f"  {'Prefill':>12}  {'Decode':>12}  {'Total':>12}")
    print(f"  {'-'*40}")
    print(f"  {'time':>12}  {prefill_t:>10.2f}s  {decode_t:>10.2f}s  {total_t:>10.2f}s")
    n_tps = result.get('prefill_tokens_per_second', 0)
    print(f"  {'t/s':>12}  {n_tps:>10.1f}  {decode_tps:>10.2f}  {result['new_tokens_per_second']:>10.2f}")
    print(f"  {'tokens':>12}  {prompt_tok:>10}  {decode_tok:>10}  {prompt_tok+decode_tok:>10}")
    print(f"  {'latency':>12}  {'':>10}  {decode_t / decode_tok * 1000:>9.1f}ms/tok")
    print(f"  {'Peak mem':>12}  {result['peak_memory_gb']:>9.1f}GB")

    # Cache + file I/O
    fl_n, fl_total = cache_mon_snap['file_n'], cache_mon_snap['file_total']
    total_cache = cache_mon_snap['cache_hits'] + cache_mon_snap['cache_misses']
    hit_rate = cache_mon_snap['cache_hits'] / total_cache * 100 if total_cache > 0 else 0
    total_gpu_store = cache_mon_snap['gpu_store_hits'] + cache_mon_snap['gpu_store_misses']
    gpu_store_rate = cache_mon_snap['gpu_store_hits'] / total_gpu_store * 100 if total_gpu_store > 0 else 0
    print(f"\n  Cache: {cache_mon_snap['cache_hits']}h/{cache_mon_snap['cache_misses']}m ({hit_rate:.0f}%)")
    print(
        f"  GPU store: {cache_mon_snap['gpu_store_hits']}h/"
        f"{cache_mon_snap['gpu_store_misses']}m ({gpu_store_rate:.0f}%)"
    )
    per_gpu_hits = cache_mon_snap.get('gpu_per_device_hits', {})
    per_gpu_misses = cache_mon_snap.get('gpu_per_device_misses', {})
    gpu_parts = []
    for dev_idx in sorted(set(per_gpu_hits) | set(per_gpu_misses)):
        h = per_gpu_hits.get(dev_idx, 0)
        m = per_gpu_misses.get(dev_idx, 0)
        r = h / (h + m) * 100 if h + m > 0 else 0
        gpu_parts.append(f"GPU{dev_idx}={h}h/{m}m({r:.0f}%)")
    if gpu_parts:
        print(f"    Per-GPU: {'  '.join(gpu_parts)}")
    if fl_n > 0:
        print(f"  File:  {fl_n} loads, {fl_total/1000:.1f}s total, {fl_total/fl_n:.1f}ms avg")

    # Layer total (all phases)
    total_attn = sum(layer_snap['attn_ms'])
    total_ffn = sum(layer_snap['ffn_ms'])
    total_misc = sum(layer_snap['mhc_attn_ms']) + sum(layer_snap['mhc_ffn_ms']) + sum(layer_snap['mhc_post_ms'])
    total_layer = total_attn + total_ffn + total_misc
    print(
        f"\n  Layer:  Attn={total_attn:.0f}ms  FFN={total_ffn:.0f}ms  "
        f"Misc={total_misc:.0f}ms  Total={total_layer:.0f}ms"
    )
    print(f"  Per-tok layer: {total_layer/decode_tok:.0f}ms/tok  FFN: {total_ffn/decode_tok:.0f}ms/tok")

    # Decode-only layer breakdown
    if decode_layer_delta is not None:
        dec_attn = sum(decode_layer_delta['attn_ms'])
        dec_ffn = sum(decode_layer_delta['ffn_ms'])
        dec_total = dec_attn + dec_ffn
        print(f"  [Decode-only] Attn={dec_attn:.0f}ms  FFN={dec_ffn:.0f}ms  Total={dec_total:.0f}ms")
        if decode_tok > 0:
            print(f"  [Decode-only] Per-tok: {dec_total/decode_tok:.0f}ms/tok  FFN: {dec_ffn/decode_tok:.0f}ms/tok")

    # EP timing breakdown
    ep_gpu0 = cache_mon_snap.get('ep_gpu0_ms', 0)
    ep_gpu1 = cache_mon_snap.get('ep_gpu1_ms', 0)
    ep_copy = cache_mon_snap.get('ep_copy_ms', 0)
    if ep_gpu0 > 0 or ep_gpu1 > 0:
        print(f"  EP timing: GPU0={ep_gpu0:.0f}ms  GPU1={ep_gpu1:.0f}ms  PCIe_copy={ep_copy:.0f}ms")


def run_profile(args):
    weight_dir = args.weight_dir
    max_new_tokens = args.max_tokens
    temperature = args.temperature
    use_mtp = args.use_mtp
    num_rounds = getattr(args, 'rounds', 1)
    prompts = getattr(args, 'prompts', args.prompt)

    print("=" * 70)
    print("HomeSeek Profiling Runner")
    print(f"  Model: {weight_dir}")
    print(f"  Rounds: {num_rounds}")
    if isinstance(prompts, list):
        print(f"  Prompts: {len(prompts)}")
        for p in prompts:
            print(f"    - \"{p[:50]}\"")
    else:
        print(f"  Prompt: \"{prompts}\"")
    print(f"  Max tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  MTP: {use_mtp}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  Device: {props.name}")
        print(f"  GPU Memory: {props.total_memory / (1024**3):.1f} GB")
    print("=" * 70)

    print("\n[1/5] Initializing engine...")
    t_init = time.time()
    engine = HomeSeekInferenceEngine(
        weight_dir, device="cuda", verbose=args.verbose,
        hot_experts_path=os.path.join(weight_dir, "..", "hot_experts.json")
        if not os.path.exists(args.hot_experts) else args.hot_experts,
        preload_all=args.preload_all,
        use_triton=not args.no_triton,
        use_gqa_fusion=args.use_gqa_fusion,
    )
    if use_mtp:
        engine._mtp_loaded = True
    if args.mtp_eager:
        engine._mtp_eager = True
    init_time = time.time() - t_init
    print(f"  Init done in {init_time:.2f}s")

    print("\n[2/5] Preparing prompts...")
    from transformers import PreTrainedTokenizerFast
    tokenizer_path = os.path.join(weight_dir, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    from home_seek.encoding_dsv4 import encode_messages

    if isinstance(prompts, str):
        prompts_list = [prompts] * num_rounds
    else:
        prompts_list = prompts[:num_rounds]

    encoded_inputs = []
    for p in prompts_list:
        text = encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
        ids = tokenizer.encode(text, return_tensors="pt").to(engine.device)
        encoded_inputs.append(ids)
        print(f"  Prompt \"{p[:40]}\" -> {ids.shape[1]} tokens")

    timer = PerLayerTimer()
    cache_mon = CacheMonitor()
    layer_trace = LayerTrace(engine.config.num_hidden_layers)
    layer_trace_decode = LayerTrace(engine.config.num_hidden_layers)
    mem_trace = []
    mem_trace.append(("init", -1, torch.cuda.memory_allocated()))

    patch_engine(engine, timer, cache_mon, layer_trace, mem_trace, layer_trace_decode)
    util_mon = UtilMonitor()

    print("\n[3/5] Running inference ...")
    print("-" * 70)

    round_results = []
    for round_idx in range(num_rounds):
        input_ids = encoded_inputs[round_idx % len(encoded_inputs)]

        if round_idx > 0:
            engine._deq_cache.clear()

        # Snapshot monitors before
        cache_before = cache_mon.snapshot()
        timer_before = timer.snapshot()
        layer_before = layer_trace.snapshot()
        decode_before = layer_trace_decode.snapshot()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        use_profiler = args.profiler != "none" and round_idx >= args.profiler_warmup
        util_mon.start()
        with torch.no_grad():
            if use_profiler:
                trace_name = f"profile_round{round_idx}_{args.profiler}"
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    result = engine.generate(input_ids, max_new_tokens=max_new_tokens,
                                              temperature=temperature)
                if args.profiler == "chrome":
                    prof.export_chrome_trace(f"{trace_name}.json")
                    cpu_total = sum(e.cpu_time_total for e in prof.events())
                    cuda_total = sum(e.cuda_time_total for e in prof.events())
                    print(f"  [Profiler] CPU total: {cpu_total/1e6:.1f}s  CUDA total: {cuda_total/1e6:.1f}s")
                    print(f"  [Profiler] Trace saved: {trace_name}.json")
            else:
                result = engine.generate(input_ids, max_new_tokens=max_new_tokens,
                                          temperature=temperature)
        util_mon.stop()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Snapshot after
        cache_after = cache_mon.snapshot()
        timer_after = timer.snapshot()
        layer_after = layer_trace.snapshot()
        decode_after = layer_trace_decode.snapshot()

        # Compute deltas
        cache_delta = CacheMonitor.delta(cache_before, cache_after)
        layer_delta = LayerTrace.delta(layer_before, layer_after)
        decode_delta = LayerTrace.delta(decode_before, decode_after)
        result['_elapsed'] = elapsed
        result['_peak_mem'] = torch.cuda.max_memory_allocated() / (1024**3)

        label = f"Round {round_idx + 1}"
        if isinstance(prompts_list[round_idx], str):
            pname = prompts_list[round_idx][:30]
            label += f" \"{pname}\""

        _print_round_result(label, result, cache_delta, layer_delta,
                           {'before': timer_before, 'after': timer_after},
                           engine.config.num_hidden_layers, tokenizer,
                           decode_layer_delta=decode_delta)
        print()

        round_results.append({
            'label': label,
            'result': result,
            'cache': cache_delta,
            'layer': layer_delta,
            'decode_layer': decode_delta,
        })

    # Multi-round comparison table
    if num_rounds > 1:
        print("=" * 70)
        print("Multi-Round Comparison")
        print("=" * 70)
        header = (
            f"{'Round':<15} {'t/s':>8} {'Decode':>8} {'File(s)':>9} "
            f"{'Loads':>7} {'Hit%':>7} {'GPU_hit':>7} {'Attn':>7} "
            f"{'FFN':>7} {'EP_G0':>7} {'EP_G1':>7} {'Mem':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in round_results:
            res = r['result']
            tps = res.get('decode_tokens_per_second', 0)
            dec = res.get('decode_time_s', 0)
            c = r['cache']
            hit = c['cache_hits'] / max(c['cache_hits'] + c['cache_misses'], 1) * 100
            gpu_hit = c['gpu_store_hits'] / max(c['gpu_store_hits'] + c['gpu_store_misses'], 1) * 100
            l = r['layer']
            attn = sum(l['attn_ms'])
            ffn = sum(l['ffn_ms'])
            ep_g0 = c.get('ep_gpu0_ms', 0)
            ep_g1 = c.get('ep_gpu1_ms', 0)
            mem = res.get('peak_memory_gb', 0)
            print(f"{r['label']:<15} {tps:>8.2f} {dec:>8.2f}s {c['file_total']/1000:>8.2f}s "
                  f"{c['file_n']:>7} {hit:>6.0f}% {gpu_hit:>6.0f}% {attn:>7.0f} {ffn:>7.0f} "
                  f"{ep_g0:>7.0f} {ep_g1:>7.0f} {mem:>6.1f}")

    # Save results
    if args.output:
        save_data = {
            'performance': {
                'total_time_s': round_results[-1]['result'].get('total_time_s', 0),
                'tokens_per_second': round_results[-1]['result'].get('new_tokens_per_second', 0),
                'ms_per_token': (
                    round_results[-1]['result'].get('total_time_s', 0)
                    / max(1, round_results[-1]['result'].get('num_generated_tokens', 1))
                    * 1000
                ),
                'peak_memory_gb': round_results[-1]['result'].get('peak_memory_gb', 0),
                'num_generated': round_results[-1]['result'].get('num_generated_tokens', 0),
            },
            'rounds': [{
                'label': r['label'],
                'decode_tps': r['result'].get('decode_tokens_per_second', 0),
                'decode_time_s': r['result'].get('decode_time_s', 0),
                'cache': r['cache'],
                'decode_layer': r.get('decode_layer'),
            } for r in round_results],
        }
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if num_rounds == 1:
        last = round_results[0]
        result = last['result']
        print(f"\n{'-' * 70}")
        print(util_mon.summary())

        print(f"\n{'-' * 70}")
        print(layer_trace.summary())

        print(f"\n{'-' * 70}")
        print(timer.summary())

        print(f"\n{'-' * 70}")
        print('\n'.join(cache_mon.summary()))

        print(f"\n{'-' * 70}")
        print("Bottleneck Analysis:")
        total_attn = sum(layer_trace.attn_ms) / 1000
        total_ffn = sum(layer_trace.ffn_ms) / 1000
        total_misc = sum(layer_trace.mhc_attn_ms) / 1000 + sum(layer_trace.mhc_ffn_ms) / 1000
        total_layer = total_attn + total_ffn + total_misc
        gen_tok = max(1, result['num_generated_tokens'])

        print(f"  Layer loop total:   {total_layer:.2f}s  ({total_layer/result['total_time_s']*100:.0f}% of total)")
        print(f"  Attention:          {total_attn:.2f}s  ({total_attn/total_layer*100:.1f}% of layer)")
        print(f"  FFN:                {total_ffn:.2f}s  ({total_ffn/total_layer*100:.1f}% of layer)")
        print(f"  MHC (pre+post):     {total_misc:.2f}s  ({total_misc/total_layer*100:.1f}% of layer)")

        if total_layer > 0:
            per_token_layer = total_layer / result['num_generated_tokens'] * 1000
            per_token_ffn = total_ffn / result['num_generated_tokens'] * 1000
            print(f"  Per-token layer loop: {per_token_layer:.0f} ms/token")
            print(f"  Per-token FFN:        {per_token_ffn:.0f} ms/token")

        hot_rate = cache_mon.hot_hits / max(cache_mon.hot_hits + cache_mon.hot_misses, 1) * 100
        print(f"  Hot cache hit rate: {hot_rate:.1f}%")
        if hot_rate < 85:
            print(f"  >> Hot cache命中率偏低({hot_rate:.1f}%)")

        idle_ratio = util_mon.gpu_idle_ratio()
        if idle_ratio >= 0:
            print(f"  GPU idle ratio (util<30%): {idle_ratio*100:.0f}%")
            if idle_ratio > 0.5:
                print("  >> GPU大量时间空闲，瓶颈在CPU/PCIe/File I/O")

        total_file_io_ms = last['cache']['file_total']
        io_pct = total_file_io_ms / 1000 / result['total_time_s'] * 100
        print(f"  Total file I/O time: {total_file_io_ms/1000:.1f}s  ({io_pct:.0f}% of total)")
        if total_file_io_ms / result['total_time_s'] > 0.2:
            io_pct = total_file_io_ms / 1000 / result['total_time_s'] * 100
            print(f"  >> File I/O占比过高({io_pct:.0f}%)，专家权重从文件读取是主要瓶颈")

        if args.output:
            output_data = {
                "config": {
                    "prompt": str(prompts_list[0]),
                    "prompt_tokens": result['num_prompt_tokens'],
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "mtp": use_mtp,
                },
                "performance": {
                    "total_time_s": result["total_time_s"],
                    "tokens_per_second": result["new_tokens_per_second"],
                    "ms_per_token": result["total_time_s"] / gen_tok * 1000,
                    "peak_memory_gb": result["peak_memory_gb"],
                    "num_generated": result["num_generated_tokens"],
                },
                "layers": {
                    "attn_ms": layer_trace.attn_ms,
                    "ffn_ms": layer_trace.ffn_ms,
                },
                "decode_layers": {
                    "attn_ms": layer_trace_decode.attn_ms,
                    "ffn_ms": layer_trace_decode.ffn_ms,
                },
                "cache": last['cache'],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")

    return round_results


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end profiling for HomeSeekInferenceEngine")
    parser.add_argument("--weight-dir", default="weights")
    parser.add_argument("--prompt", default="Hello",
                        help="Single prompt, or first prompt when --prompts used")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="List of prompts for multi-round. Overrides --prompt")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds (default 1). Prompts cycle if fewer prompts than rounds")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--use-gqa-fusion", action="store_true",
                        help="Use GQA fused attention kernel (experimental, no throughput gain in current profile)")
    parser.add_argument("--mtp-eager", action="store_true",
                        help="MTP eager: accept all draft tokens without verification")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--preload-all", action="store_true",
                        help="Preload all experts into pinned CPU memory at init")
    parser.add_argument("--hot-experts", default="hot_experts.json")
    parser.add_argument("--no-triton", action="store_true",
                        help="Disable Triton kernels (use PyTorch fallbacks)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--profiler", choices=["chrome", "none"], default="none",
                        help="Enable torch.profiler. 'chrome' exports Chrome trace JSON")
    parser.add_argument("--profiler-warmup", type=int, default=1,
                        help="Warmup rounds before profiling (default 1, ignored if --profiler=none)")
    args = parser.parse_args()
    if args.prompts:
        args.prompts = list(args.prompts)
        args.rounds = max(args.rounds, len(args.prompts))
    else:
        args.prompts = args.prompt
    run_profile(args)


if __name__ == "__main__":
    main()
