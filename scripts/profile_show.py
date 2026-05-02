"""Display last profile summary with intra-FFN breakdown."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/last_profile.json"
with open(path) as f:
    d = json.load(f)
p = d["performance"]
print(f"\n=== Profile: {path} ===")
print(f"  Decode throughput: {p['tokens_per_second']:.2f} t/s")
print(f"  Peak memory: {p['peak_memory_gb']:.2f} GB")
print(f"  Total time: {p['total_time_s']:.2f}s")
print(f"  Tokens generated: {p['num_generated']}")

# BW probe
bw = d.get("bw_probe")
if bw:
    print(f"\n  GPU BW: {bw['achieved_bw_gbs']:.1f}/{bw['peak_bw_gbs']:.1f} GB/s ({bw['util_pct']:.1f}% util)")

# Layer summary
layers = d.get("decode_layers") or d.get("layers", {})
if layers:
    attn = sum(layers.get("attn_ms", []))
    ffn = sum(layers.get("ffn_ms", []))
    mhc = sum(layers.get("mhc_attn_ms", [])) + sum(layers.get("mhc_ffn_ms", [])) + sum(layers.get("mhc_post_ms", []))
    print(f"\n  Layers: Attn={attn:.0f}ms  FFN={ffn:.0f}ms  MHC={mhc:.0f}ms")
    # Intra-FFN
    routing = sum(layers.get("ffn_routing_ms", []))
    load = sum(layers.get("ffn_load_ms", []))
    m1 = sum(layers.get("ffn_m1_ms", []))
    shared = sum(layers.get("ffn_shared_ms", []))
    ffn_tracked = routing + load + m1 + shared
    if ffn_tracked > 0:
        print(f"  Intra-FFN: routing={routing:.0f}ms  load={load:.0f}ms  m1={m1:.0f}ms  shared={shared:.0f}ms  overhead={ffn-ffn_tracked:.0f}ms")

# Cache summary
cache = (d.get("rounds", [{}])[-1].get("cache") if d.get("rounds") else d.get("cache", {}))
if cache:
    total = cache.get("cache_hits", 0) + cache.get("cache_misses", 0)
    hit_rate = cache.get("cache_hits", 0) / max(total, 1) * 100
    gpu_total = cache.get("gpu_store_hits", 0) + cache.get("gpu_store_misses", 0)
    gpu_rate = cache.get("gpu_store_hits", 0) / max(gpu_total, 1) * 100
    print(f"\n  Cache: {hit_rate:.0f}% hit rate  GPU_store: {gpu_rate:.0f}%  Files: {cache.get('file_n', 0)} loads")
