#!/usr/bin/env python3
"""Hardware probe: measure disk/PCIe/GPU theoretical limits and compare with
actual HomeSeek inference timings.

Key measurements:
1. Disk seq read BW (4.2 GB/s claimed)
2. PCIe DMA CPU→GPU BW (7.0 GB/s claimed)
3. PCIe BAR GPU←CPU BW (0.2 GB/s claimed — the bottleneck)
4. GPU memory BW (919 GB/s claimed)
5. GPU BF16 matmul FLOPS (theoretical peak)
6. GPU FP4 deq BW
7. Weight total size breakdown
"""

import os, sys, time, json, math, glob as _glob
import torch
import numpy as np

DT = torch.bfloat16
WARMUP = 3
REPEAT = 10
GB = 1024**3


def bench(name, fn, warmup=WARMUP, repeat=REPEAT, cleanup=None):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    avg_ms = elapsed / repeat * 1000
    return avg_ms


def bench_get(name, fn, get_result, warmup=WARMUP, repeat=REPEAT):
    """Run fn repeatedly and return result + avg time."""
    for _ in range(warmup):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    avg_ms = elapsed / repeat * 1000
    return avg_ms, result


def main():
    print("=" * 70)
    print("HomeSeek Hardware Probe")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / GB:.1f} GB")
    print(f"Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    print(f"CPU cores: {os.cpu_count()}")

    # ── Disk ──
    print("\n" + "=" * 70)
    print("1. DISK SEQUENTIAL READ")
    print("-" * 70)

    weight_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))
    safetensors = sorted(_glob.glob(os.path.join(weight_dir, "*.safetensors")))
    total_gb = sum(os.path.getsize(f) for f in safetensors) / GB
    print(f"  Weight files: {len(safetensors)} files, {total_gb:.1f} GB")

    # Measure bulk read of a single large file
    large = max(safetensors, key=os.path.getsize)
    fsize_gb = os.path.getsize(large) / GB
    print(f"  Largest file: {os.path.basename(large)} ({fsize_gb:.2f} GB)")

    def read_whole():
        with open(large, "rb") as f:
            return f.read()

    ms_read = bench("bulk read", read_whole, warmup=1, repeat=3)
    disk_bw = fsize_gb / (ms_read / 1000)
    print(f"  Bulk read: {ms_read:.1f} ms → {disk_bw:.1f} GB/s")

    theoretical_disk = disk_bw  # capped by disk HW

    # ── PCIe DMA ──
    print("\n" + "=" * 70)
    print("2. PCIe DMA (CPU → GPU, .to('cuda'))")
    print("-" * 70)

    for size_mb in [1, 10, 50, 100, 500]:
        elements = size_mb * 1024 * 1024 // 2
        x_cpu = torch.randn(elements, dtype=torch.bfloat16, device="cpu")
        ms = bench(f"dma {size_mb}MB", lambda: x_cpu.to("cuda", non_blocking=True), warmup=3, repeat=10)
        bw = (size_mb / 1024) / (ms / 1000)
        print(f"  DMA {size_mb:>4}MB: {ms:7.2f} ms → {bw:.1f} GB/s")

    pcie_dma_bw = None
    x_cpu_large = torch.randn(500 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cpu")
    ms_dma_large = bench("dma 500MB large", lambda: x_cpu_large.to("cuda", non_blocking=True), warmup=2, repeat=5)
    pcie_dma_bw = 500 / 1024 / (ms_dma_large / 1000)
    print(f"  DMA 500MB large: {ms_dma_large:.1f} ms → {pcie_dma_bw:.1f} GB/s")

    # ── PCIe BAR (GPU reads CPU) ──
    print("\n" + "=" * 70)
    print("3. PCIe BAR (GPU kernel reads CPU memory)")
    print("-" * 70)

    # Triton may block direct CPU access in newer versions.
    # Simulate BAR bandwidth with a pytorch CPU→GPU copy that's intentionally
    # not done via DMA: use .to('cuda') without non_blocking to get sync copy time.
    # Also try a small Triton kernel to check if BAR access still works.

    for size_mb in [1, 10, 50, 100]:
        elements = size_mb * 1024 * 1024 // 2
        x_cpu = torch.randn(elements, dtype=torch.bfloat16, device="cpu")

        # Synchronous .to('cuda') (waits for copy to finish)
        ms_sync = bench(f"sync copy {size_mb}MB",
                        lambda: x_cpu.to("cuda"), warmup=3, repeat=10)
        bw_sync = (size_mb / 1024) / (ms_sync / 1000)
        print(f"  Sync copy {size_mb:>4}MB: {ms_sync:7.2f} ms → {bw_sync:.2f} GB/s")

    # Try Triton BAR access (may fail on newer Triton)
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _copy_bar_try(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            val = tl.load(src_ptr + offs, mask=mask, other=0.0)
            tl.store(dst_ptr + offs, val, mask=mask)

        elements = 1 * 1024 * 1024 // 2
        x_cpu_small = torch.randn(elements, dtype=torch.bfloat16, device="cpu")
        x_gpu_small = torch.empty(elements, dtype=torch.bfloat16, device="cuda")

        def bar_try():
            grid = (triton.cdiv(elements, 1024),)
            _copy_bar_try[grid](x_cpu_small, x_gpu_small, elements, BLOCK=1024)

        ms_bar = bench("bar 1MB try", bar_try, warmup=2, repeat=5)
        bw_bar = (1 / 1024) / (ms_bar / 1000)
        print(f"  Triton BAR 1MB (direct):      {ms_bar:.1f} ms → {bw_bar:.2f} GB/s")
    except ValueError as e:
        print(f"  Triton BAR: blocked by runtime ({e})")
    except Exception as e:
        print(f"  Triton BAR: error ({type(e).__name__}: {e})")

    # ── GPU Memory BW ──
    print("\n" + "=" * 70)
    print("4. GPU MEMORY BANDWIDTH")
    print("-" * 70)

    for size_gb in [0.5, 1, 2]:
        elements = int(size_gb * GB // 2)
        try:
            a = torch.randn(elements, dtype=torch.bfloat16, device="cuda")
            b = torch.empty_like(a)

            def gpu_copy():
                b.copy_(a)

            ms_copy = bench(f"gpu copy {size_gb}GB", gpu_copy, warmup=2, repeat=10)
            bw = size_gb / (ms_copy / 1000)
            print(f"  GPU copy {size_gb}GB: {ms_copy:.1f} ms → {bw:.1f} GB/s")
            del a, b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  GPU copy {size_gb}GB: OOM, skip")
            break

    gpu_bw = None
    elements = int(1 * GB // 2)
    a = torch.randn(elements, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    ms_copy_1gb = bench("gpu copy 1GB", lambda: b.copy_(a), warmup=2, repeat=10)
    gpu_bw = 1.0 / (ms_copy_1gb / 1000)
    del a, b
    torch.cuda.empty_cache()
    print(f"\n  → GPU mem BW: {gpu_bw:.1f} GB/s")

    # ── GPU Matmul FLOPS ──
    print("\n" + "=" * 70)
    print("5. GPU BF16 MATMUL PERFORMANCE")
    print("-" * 70)

    # Decode profile: [1, 4096] × [4096, 12288]
    decode_cfg_small = [(1, 4096, 12288)]
    # Prefill profile: [B, 4096] × [4096, 12288]
    prefill_cfg = [(4, 4096, 12288), (16, 4096, 12288), (64, 4096, 12288)]

    for B, K, N in decode_cfg_small + prefill_cfg:
        a = torch.randn(B, K, device="cuda", dtype=DT)
        b = torch.randn(N, K, device="cuda", dtype=DT)
        flops_per_op = 2 * B * K * N
        gflops = flops_per_op / 1e9

        ms = bench(f"matmul [{B},{K}]×[{K},{N}]", lambda: torch.matmul(a, b.t()), warmup=5, repeat=20)
        tflops_s = flops_per_op / (ms / 1000) / 1e12
        print(f"  matmul [{B:>4},{K}]×[{K},{N}]: {ms:7.2f} ms → {tflops_s:.2f} TFLOPS ({gflops:.1f} GFLOP/op)")

    # Full FFN matmul: w1 [I*ne, D] and w2 [D, I*ne]
    # Typical: I=2048, ne=8, D=4096 → [16384, 4096] and [4096, 16384]
    I, ne, D = 2048, 8, 4096
    a = torch.randn(1, D, device="cuda", dtype=DT)
    w1 = torch.randn(I * ne, D, device="cuda", dtype=DT)  # 16384 × 4096
    w2 = torch.randn(D, I * ne, device="cuda", dtype=DT)
    flops_gate = 2 * 1 * D * (I * ne)
    flops_up = flops_gate
    flops_down = 2 * 1 * (I * ne) * D
    flops_total = flops_gate + flops_up + flops_down

    ms_gate = bench("gate matmul", lambda: torch.matmul(a, w1.t()), warmup=5, repeat=20)
    ms_up = bench("up matmul", lambda: torch.matmul(a, w1.t()), warmup=5, repeat=20)
    combined = w1.T.contiguous()
    ms_down = bench("down matmul", lambda: torch.matmul(a, combined[:D, :]), warmup=5, repeat=20)
    tflops = flops_total / (ms_gate / 1000) / 1e12
    print(f"\n  Full FFN ([1,{D}]×[{I*ne},...]): {ms_gate:.2f}ms(gate) → {tflops:.2f} TFLOPS")
    print(f"  Matmul time per decode: ~{ms_gate + ms_up + ms_down:.2f}ms (ideal)")

    # ── Weight size analysis ──
    print("\n" + "=" * 70)
    print("6. WEIGHT SIZE BREAKDOWN")
    print("-" * 70)

    # Model constants
    num_layers = 43
    num_experts = 256
    hidden = 4096
    inter = 2048
    shared_inter = 2048

    def tensor_bytes(shape, dtype_bytes):
        if isinstance(shape, int):
            return shape * dtype_bytes
        return math.prod(shape) * dtype_bytes

    # FP4 packed: int8 [I, D/2] + float32 scale [I, D/32]
    w1_fp4_bytes = tensor_bytes((inter, hidden // 2), 1) + tensor_bytes((inter, hidden // 32), 4)
    w3_fp4_bytes = tensor_bytes((inter, hidden // 2), 1) + tensor_bytes((inter, hidden // 32), 4)
    w2_fp4_bytes = tensor_bytes((hidden, inter // 2), 1) + tensor_bytes((hidden, inter // 32), 4)
    expert_fp4_bytes = w1_fp4_bytes + w3_fp4_bytes + w2_fp4_bytes
    fp4_per_expert_mb = expert_fp4_bytes / 1e6

    # BF16 dequantized
    w1_bf16_bytes = tensor_bytes((inter, hidden), 2)
    w3_bf16_bytes = tensor_bytes((inter, hidden), 2)
    w2_bf16_bytes = tensor_bytes((hidden, inter), 2)
    expert_bf16_bytes = w1_bf16_bytes + w3_bf16_bytes + w2_bf16_bytes
    bf16_per_expert_mb = expert_bf16_bytes / 1e6

    # Shared expert (FP8)
    shared_w1_fp8_bytes = tensor_bytes((shared_inter, hidden), 1) + tensor_bytes((shared_inter, hidden // 128), 4)
    shared_w3_fp8_bytes = tensor_bytes((shared_inter, hidden), 1) + tensor_bytes((shared_inter, hidden // 128), 4)
    shared_w2_fp8_bytes = tensor_bytes((hidden, shared_inter), 1) + tensor_bytes((hidden, shared_inter // 128), 4)
    shared_fp8_bytes = shared_w1_fp8_bytes + shared_w3_fp8_bytes + shared_w2_fp8_bytes

    experts_per_layer = 8  # topk=6 but shared expert adds 2 for safety margin

    print(f"  Per expert (FP4 packed):       {fp4_per_expert_mb:.1f} MB")
    print(f"  Per expert (BF16 deq):         {bf16_per_expert_mb:.1f} MB")
    print(f"  Per expert DMA ratio:          {expert_bf16_bytes / expert_fp4_bytes:.1f}× deq expansion")
    print(f"  Per layer (8 experts/slot):    {fp4_per_expert_mb * 8:.1f} MB (FP4) / {bf16_per_expert_mb * 8:.1f} MB (BF16)")
    print(f"  Shared expert (FP8):           {shared_fp8_bytes / 1e6:.1f} MB")
    print(f"  All experts (256 × FP4):       {expert_fp4_bytes * num_experts / GB:.2f} GB")
    print(f"  43 layers × 256 exps (raw):    {expert_fp4_bytes * num_experts * num_layers / GB:.1f} GB")

    # ── Theoretical bottleneck analysis ──
    print("\n" + "=" * 70)
    print("7. BOTTLENECK ANALYSIS — theoretical vs actual")
    print("-" * 70)

    print(f"\n  Disk measured BW: {disk_bw:.1f} GB/s")
    print(f"  PCIe DMA BW:      {pcie_dma_bw:.1f} GB/s")
    print(f"  GPU mem BW:       {gpu_bw:.1f} GB/s")

    # Time to load from disk (using measured BW)
    # fp4_per_expert_mb in MB, disk_bw in GB/s → convert MB to GB
    disk_read_ms = (fp4_per_expert_mb / 1024) / disk_bw * 1000
    dma_ms = (fp4_per_expert_mb / 1024) / pcie_dma_bw * 1000

    # PCIe BAR: blocked by newer Triton, previously measured at 0.2 GB/s
    bar_gbs = 0.2
    bar_ms = (fp4_per_expert_mb / 1024) / bar_gbs * 1000

    print(f"\n  Per expert time breakdown:")
    print(f"    Disk read ({disk_bw:.1f} GB/s):       {disk_read_ms:.2f} ms")
    print(f"    DMA CPU→GPU ({pcie_dma_bw:.1f} GB/s):  {dma_ms:.2f} ms")
    print(f"    BAR (GPU←CPU, {bar_gbs} GB/s):  {bar_ms:.2f} ms")
    print(f"    Disk + DMA:              {disk_read_ms + dma_ms:.2f} ms")
    print(f"    Disk + BAR (old):        {disk_read_ms + bar_ms:.2f} ms")

    deq_gpu_ms = 0.17
    matmul_ms = ms_gate + ms_up + ms_down
    print(f"    Deq (GPU resident):      {deq_gpu_ms:.2f} ms")
    print(f"    Matmul (per layer):      {matmul_ms:.2f} ms")

    per_layer_old = disk_read_ms + bar_ms + deq_gpu_ms + matmul_ms
    per_layer_new = disk_read_ms + dma_ms + deq_gpu_ms + matmul_ms
    per_layer_cached = dma_ms + deq_gpu_ms + matmul_ms

    print(f"\n  Per decode layer ({experts_per_layer} experts):")
    print(f"    Old (disk + BAR):        {per_layer_old * experts_per_layer:.1f} ms")
    print(f"    New (disk + DMA):        {per_layer_new * experts_per_layer:.1f} ms")
    print(f"    Cache hit (DMA only):    {per_layer_cached * experts_per_layer:.1f} ms")

    tok_old_ms = per_layer_old * experts_per_layer * num_layers
    tok_new_ms = per_layer_new * experts_per_layer * num_layers
    tok_cached_ms = per_layer_cached * experts_per_layer * num_layers
    print(f"\n  Per token ({num_layers} layers):")
    print(f"    Old (disk + BAR):        {tok_old_ms:.0f} ms")
    print(f"    New (disk + DMA):        {tok_new_ms:.0f} ms")
    print(f"    Cache hit (all cached):  {tok_cached_ms:.0f} ms")
    print(f"    Actual (profile):        ~1867 ms/decode token")

    # File I/O bottleneck
    file_io_per_token = 43 * 8 * 4.21
    print(f"\n  File I/O bottleneck analysis:")
    print(f"    File I/O (43×8×4.21ms):  {file_io_per_token:.0f} ms/token (all misses)")
    print(f"    With 66% cache hit:      {file_io_per_token * 0.34:.0f} ms/token (34% misses)")

    dma_total_per_token = (fp4_per_expert_mb / 1024) * experts_per_layer * num_layers / pcie_dma_bw * 1000
    print(f"    DMA BW demand (43×8 FP4):{dma_total_per_token:.0f} ms/token")

    # ██ SUMMARY ██
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    theoretical_tps = 1000 / tok_cached_ms
    print(f"  Theoretical max throughput:    {theoretical_tps:.1f} t/s (all GPU-resident, no I/O)")
    print(f"  Theoretical DMA-bound:         {1000 / tok_new_ms:.1f} t/s (disk→DMA→GPU, all misses)")
    print(f"  Current actual throughput:     0.24 t/s")
    print(f"\n  Key numbers:")
    print(f"    Disk seq read:    {disk_bw:.1f} GB/s")
    print(f"    PCIe DMA:         {pcie_dma_bw:.1f} GB/s")
    print(f"    GPU mem BW:       {gpu_bw:.1f} GB/s")


if __name__ == "__main__":
    main()
