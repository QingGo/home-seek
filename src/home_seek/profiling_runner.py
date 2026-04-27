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
import sys
import json
import time
import argparse
import threading
from collections import defaultdict

import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_encoding_dir = os.path.join(_project_root, 'weights', 'encoding')
sys.path.insert(0, os.path.abspath(_encoding_dir))

from home_seek.inference_engine import HomeSeekInferenceEngine


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
            lines.append(f"  GPU Compute: avg={g_avg:.1f}%  min={g_min:.1f}%  max={g_max:.1f}%  idle_time={idle_pct:.0f}%")
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
        self.file_loads = 0
        self.deq_times = []
        self.file_load_times = []

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

    def record_gpu_store(self, hit: bool):
        if hit:
            self.gpu_store_hits += 1
        else:
            self.gpu_store_misses += 1

    def record_file_load(self):
        self.file_loads += 1

    def record_deq_time(self, ms: float):
        self.deq_times.append(ms)

    def record_file_load_time(self, ms: float):
        self.file_load_times.append(ms)

    def summary(self):
        total_hot = self.hot_hits + self.hot_misses
        hot_rate = self.hot_hits / total_hot * 100 if total_hot > 0 else 0
        total_cache = self.cache_hits + self.cache_misses
        cache_rate = self.cache_hits / total_cache * 100 if total_cache > 0 else 0
        total_gpu = self.gpu_store_hits + self.gpu_store_misses
        gpu_rate = self.gpu_store_hits / total_gpu * 100 if total_gpu > 0 else 0

        def stats(vals):
            if not vals:
                return (0, 0, 0, 0)
            return (len(vals), min(vals), sum(vals) / len(vals), max(vals))

        deq_n, deq_min, deq_avg, deq_max = stats(self.deq_times)
        fl_n, fl_min, fl_avg, fl_max = stats(self.file_load_times)

        lines = [
            "Expert Cache Performance:",
            f"  Hot cache hits: {self.hot_hits}, misses: {self.hot_misses}  (hit rate: {hot_rate:.1f}%)",
            f"  Raw cache hits: {self.cache_hits}, misses: {self.cache_misses}  (hit rate: {cache_rate:.1f}%)",
            f"  GPU store hits: {self.gpu_store_hits}, misses: {self.gpu_store_misses}  (hit rate: {gpu_rate:.1f}%)",
            f"  File loads: {self.file_loads}",
            f"  Dequantize times (ms): n={deq_n}  avg={deq_avg:.2f}  min={deq_min:.2f}  max={deq_max:.2f}",
            f"  File load times (ms):  n={fl_n}  avg={fl_avg:.2f}  min={fl_min:.2f}  max={fl_max:.2f}",
        ]
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

    def summary(self):
        lines = ["Per-Layer Breakdown:"]
        header = f"{'Layer':<7} {'Attn(ms)':<10} {'FFN(ms)':<10} {'MHC_A(ms)':<10} {'MHC_F(ms)':<10} {'Post(ms)':<10} {'Total(ms)':<10} {'Experts'}"
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


def patch_engine(engine, timer, cache_mon, layer_trace, mem_trace):
    original_forward_attn = engine._forward_attn
    original_forward_ffn = engine._forward_ffn
    original_process_mhc = engine._process_mhc_layer
    original_mhc_post = engine._process_mhc_post
    original_load_expert = engine._load_expert_weights
    original_expert_deq = engine._load_expert_deq
    original_expert_cache_get = engine.expert_cache.get
    original_hc_head = engine._hc_head

    def traced_forward_attn(hidden_states, lw, layer_idx):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = original_forward_attn(hidden_states, lw, layer_idx)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        timer.record("attention", layer_idx, dt)
        layer_trace.attn_ms[layer_idx] += dt
        mem_trace.append(("attn_end", layer_idx, torch.cuda.memory_allocated()))
        return result

    def traced_forward_ffn(hidden_states, lw, layer_idx, input_ids=None):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result, used = original_forward_ffn(hidden_states, lw, layer_idx, input_ids)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        timer.record("ffn", layer_idx, dt)
        layer_trace.ffn_ms[layer_idx] += dt
        layer_trace.expert_ids[layer_idx] = sorted(used) if used else []
        mem_trace.append(("ffn_end", layer_idx, torch.cuda.memory_allocated()))
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

    engine._forward_attn = traced_forward_attn
    engine._forward_ffn = traced_forward_ffn
    engine._process_mhc_layer = traced_process_mhc
    engine._process_mhc_post = traced_mhc_post
    engine._hc_head = traced_hc_head
    engine._load_expert_weights = traced_load_expert
    engine._load_expert_deq = traced_expert_deq
    engine.expert_cache.get = traced_cache_get
    return engine


def run_profile(args):
    weight_dir = args.weight_dir
    prompt = args.prompt
    max_new_tokens = args.max_tokens
    temperature = args.temperature
    use_mtp = args.use_mtp

    print("=" * 70)
    print("HomeSeek Profiling Runner")
    print(f"  Model: {weight_dir}")
    print(f"  Prompt: \"{prompt}\"")
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
    )
    if use_mtp:
        engine._mtp_loaded = True
    if args.mtp_eager:
        engine._mtp_eager = True
    init_time = time.time() - t_init
    print(f"  Init done in {init_time:.2f}s")

    print("\n[2/5] Tokenizing prompt...")
    from transformers import PreTrainedTokenizerFast
    tokenizer_path = os.path.join(weight_dir, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    from encoding_dsv4 import encode_messages
    prompt_text = encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(engine.device)
    num_prompt_tokens = input_ids.shape[1]
    print(f"  Prompt tokens: {num_prompt_tokens}")

    timer = PerLayerTimer()
    cache_mon = CacheMonitor()
    layer_trace = LayerTrace(engine.config.num_hidden_layers)
    mem_trace = []
    mem_trace.append(("init", -1, torch.cuda.memory_allocated()))

    patch_engine(engine, timer, cache_mon, layer_trace, mem_trace)

    util_mon = UtilMonitor()

    print("\n[3/5] Baseline...")
    torch.cuda.reset_peak_memory_stats()
    mem_baseline = torch.cuda.memory_allocated()
    print(f"  Memory baseline: {mem_baseline / (1024**3):.2f} GB")

    print("\n[4/5] Running inference ...")
    print("-" * 70)
    t_gen_start = time.time()
    util_mon.start()

    with torch.no_grad():
        result = engine.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)

    util_mon.stop()

    print("\n[5/5] Results")
    print("-" * 70)
    out = tokenizer.decode(result["tokens"][0], skip_special_tokens=True)
    print("\nGenerated Text:")
    print(out)
    print()

    print("-" * 70)
    print("Performance Summary:")
    print(f"  Total time: {result['total_time_s']:.2f}s")
    print(f"  Generated tokens: {result['num_generated_tokens']}")
    print(f"  Throughput (with prefill): {result['new_tokens_per_second']:.2f} t/s")
    decode_tps = result.get('decode_tokens_per_second', 0)
    prefill_t = result.get('prefill_time_s', 0)
    decode_t = result.get('decode_time_s', 0)
    prompt_tok = result['num_prompt_tokens']
    decode_tok = max(1, result['num_generated_tokens'])
    has_split = 'prefill_time_s' in result and result['prefill_time_s'] > 0
    if has_split:
        print(f"  Prefill time:            {prefill_t:.2f}s  ({prompt_tok} tok, {result['prefill_tokens_per_second']:.1f} t/s)")
        print(f"  TTFT:                    {prefill_t:.2f}s")
        print(f"  Decode time:             {decode_t:.2f}s")
        print(f"  Decode throughput:       {decode_tps:.2f} t/s")
        print(f"  Decode latency:          {decode_t / decode_tok * 1000:.1f} ms/tok")
    else:
        print("  Prefill/decode:          not separated (engine v14 or older)")
        print(f"  Avg per-token (incl prefill): {result['total_time_s'] / decode_tok * 1000:.0f} ms/tok")
    print(f"  Peak memory: {result['peak_memory_gb']:.2f} GB")
    print(f"  Prompt tokens: {result['num_prompt_tokens']}")

    print(f"\n{'-' * 70}")
    print(util_mon.summary())

    print(f"\n{'-' * 70}")
    print(layer_trace.summary())

    print(f"\n{'-' * 70}")
    print(timer.summary())

    print(f"\n{'-' * 70}")
    print(cache_mon.summary())

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

    total_file_io_ms = sum(cache_mon.file_load_times)
    print(f"  Total file I/O time: {total_file_io_ms/1000:.1f}s  ({total_file_io_ms/1000/result['total_time_s']*100:.0f}% of total)")
    if total_file_io_ms / result['total_time_s'] > 0.2:
        print(f"  >> File I/O占比过高({total_file_io_ms/1000/result['total_time_s']*100:.0f}%)，专家权重从文件读取是主要瓶颈")

    if args.output:
        output_data = {
            "config": {
                "prompt": prompt,
                "prompt_tokens": num_prompt_tokens,
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
            "cache": {
                "hot_hits": cache_mon.hot_hits,
                "hot_misses": cache_mon.hot_misses,
                "file_loads": cache_mon.file_loads,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="HomeSeek End-to-End Profiling Runner")
    parser.add_argument("--weight-dir", default="weights")
    parser.add_argument("--prompt", default="介绍一下你自己")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--use-mtp", action="store_true")
    parser.add_argument("--mtp-eager", action="store_true",
                        help="Eager MTP: accept all drafts without verification (fast but risky)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--preload-all", action="store_true",
                        help="Pre-load all experts to CPU RAM during init (eliminates file I/O)")
    parser.add_argument("--hot-experts", default="hot_experts.json")
    parser.add_argument("--no-triton", action="store_true",
                        help="Disable Triton kernels (use PyTorch fallback for MHC)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_profile(args)


if __name__ == "__main__":
    main()
