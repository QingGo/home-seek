"""Compare current profile against previous baseline with full detail.

Usage:
    python scripts/profile_compare.py artifacts/prev_profile.json artifacts/last_profile.json
    python scripts/profile_compare.py  # defaults to prev_profile.json vs last_profile.json
"""
import json
import sys


def load_profile(path):
    with open(path) as f:
        return json.load(f)


def pct_change(prev_val, cur_val):
    if prev_val == 0:
        return None
    return (cur_val - prev_val) / prev_val * 100


def fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val:+.1f}%"


def sum_list(lst):
    return sum(lst) if lst else 0


def compare_layers(prev_layers, cur_layers, key, label):
    p = prev_layers.get(key, [])
    c = cur_layers.get(key, [])
    if not p or not c:
        return []
    lines = []
    lines.append(f"\n  {label} per-layer:")
    diffs = [c[i] - p[i] for i in range(len(p))]
    p_sum = sum(p)
    c_sum = sum(c)
    ch = pct_change(p_sum, c_sum)
    lines.append(f"    Total: {p_sum:.0f}ms -> {c_sum:.0f}ms ({fmt_pct(ch)})")
    improved = [i for i, d in enumerate(diffs) if d < -1.0]
    degraded = [i for i, d in enumerate(diffs) if d > 1.0]
    if improved:
        lines.append(f"    Layers improved ({len(improved)}): {improved[:5]}{'...' if len(improved) > 5 else ''}")
    if degraded:
        lines.append(f"    Layers degraded ({len(degraded)}): {degraded[:5]}{'...' if len(degraded) > 5 else ''}")
    return lines


def compare_ffn_internal(prev_layers, cur_layers, decode_key="decode_layers"):
    """Compare intra-FFN breakdown between two profiles."""
    p_layers = prev_layers.get(decode_key, prev_layers.get("layers", {}))
    c_layers = cur_layers.get(decode_key, cur_layers.get("layers", {}))
    ffn_keys = [
        ("ffn_routing_ms", "FFN routing"),
        ("ffn_load_ms", "FFN expert load"),
        ("ffn_m1_ms", "FFN M1 kernel"),
        ("ffn_shared_ms", "FFN shared expert"),
    ]
    lines = []
    for key, label in ffn_keys:
        p_vals = p_layers.get(key, [])
        c_vals = c_layers.get(key, [])
        if not p_vals or not c_vals:
            continue
        p_sum = sum(p_vals)
        c_sum = sum(c_vals)
        ch = pct_change(p_sum, c_sum)
        lines.append(f"    {label:<20}: {p_sum:>8.0f}ms -> {c_sum:>8.0f}ms ({fmt_pct(ch)})")
    if lines:
        lines.insert(0, "\n  Intra-FFN breakdown (decode-only):")
    return lines


def main():
    prev_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/prev_profile.json"
    cur_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/last_profile.json"

    try:
        prev = load_profile(prev_path)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No previous profile found at {prev_path}. Run `make profile` to create a baseline, "
              f"then `cp artifacts/last_profile.json artifacts/prev_profile.json`.")
        sys.exit(1)

    try:
        cur = load_profile(cur_path)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No current profile found at {cur_path}. Run `make profile` first.")
        sys.exit(1)

    prev_perf = prev.get("performance", {})
    cur_perf = cur.get("performance", {})

    print(f"Profile Compare: {prev_path} -> {cur_path}")
    print()

    # Top-level performance
    print(f"{'Metric':<40} {'Previous':>12} {'Current':>12} {'Change':>10}")
    print("-" * 75)
    for key, label in [("tokens_per_second", "t/s"), ("ms_per_token", "ms/token"),
                        ("peak_memory_gb", "Peak VRAM (GB)"), ("total_time_s", "Total time (s)"),
                        ("num_generated", "Tokens generated")]:
        pv = prev_perf.get(key, 0)
        cv = cur_perf.get(key, 0)
        ch = pct_change(pv, cv)
        print(f"{label:<40} {pv:>12.4f} {cv:>12.4f} {fmt_pct(ch):>10}")

    # Cache comparison
    prev_rounds = prev.get("rounds", [])
    cur_rounds = cur.get("rounds", [])
    prev_cache = prev_rounds[-1].get("cache", {}) if prev_rounds else prev.get("cache", {})
    cur_cache = cur_rounds[-1].get("cache", {}) if cur_rounds else cur.get("cache", {})

    if prev_cache and cur_cache:
        print(f"\n{'Cache Metric':<40} {'Previous':>12} {'Current':>12} {'Change':>10}")
        print("-" * 75)
        cache_metrics = [
            ("cache_hits", "Cache hits"),
            ("gpu_store_hits", "GPU store hits"),
            ("hot_hits", "Hot cache hits"),
            ("file_n", "File loads"),
            ("deq_n", "Dequantize calls"),
        ]
        for key, label in cache_metrics:
            pv = prev_cache.get(key, 0)
            cv = cur_cache.get(key, 0)
            ch = pct_change(pv, cv)
            print(f"{label:<40} {pv:>12} {cv:>12} {fmt_pct(ch):>10}")

        # Hit rates
        p_total = prev_cache.get("cache_hits", 0) + prev_cache.get("cache_misses", 0)
        c_total = cur_cache.get("cache_hits", 0) + cur_cache.get("cache_misses", 0)
        p_rate = prev_cache.get("cache_hits", 0) / max(p_total, 1) * 100
        c_rate = cur_cache.get("cache_hits", 0) / max(c_total, 1) * 100
        print(f"{'  Cache hit rate':<40} {p_rate:>11.1f}% {c_rate:>11.1f}% {'':>10}")

        p_gpu = prev_cache.get("gpu_store_hits", 0) + prev_cache.get("gpu_store_misses", 0)
        c_gpu = cur_cache.get("gpu_store_hits", 0) + cur_cache.get("gpu_store_misses", 0)
        p_grate = prev_cache.get("gpu_store_hits", 0) / max(p_gpu, 1) * 100
        c_grate = cur_cache.get("gpu_store_hits", 0) / max(c_gpu, 1) * 100
        print(f"{'  GPU store hit rate':<40} {p_grate:>11.1f}% {c_grate:>11.1f}% {'':>10}")

        # File I/O total
        p_file = prev_cache.get("file_total", 0) / 1000
        c_file = cur_cache.get("file_total", 0) / 1000
        ch_file = pct_change(p_file, c_file)
        print(f"{'  File I/O total (s)':<40} {p_file:>11.2f}s {c_file:>11.2f}s {fmt_pct(ch_file):>10}")

    # Layer comparison
    prev_layers = prev_rounds[-1].get("decode_layer", {}) if prev_rounds else prev.get("decode_layers", {})
    cur_layers = cur_rounds[-1].get("decode_layer", {}) if cur_rounds else cur.get("decode_layers", {})

    if prev_layers and cur_layers:
        for key, label in [("attn_ms", "Attention"), ("ffn_ms", "FFN total"),
                           ("mhc_attn_ms", "MHC attn"), ("mhc_ffn_ms", "MHC FFN"),
                           ("mhc_post_ms", "MHC post")]:
            for line in compare_layers(prev_layers, cur_layers, key, label):
                print(line)

        # Intra-FFN comparison
        for line in compare_ffn_internal(prev_layers, cur_layers):
            print(line)

    # BW probe comparison
    prev_bw = prev.get("bw_probe")
    cur_bw = cur.get("bw_probe")
    if prev_bw and cur_bw:
        print(f"\n{'BW Metric':<40} {'Previous':>12} {'Current':>12} {'Change':>10}")
        print("-" * 75)
        for key, label in [("achieved_bw_gbs", "Achieved BW (GB/s)"),
                           ("util_pct", "BW utilization (%)")]:
            pv = prev_bw.get(key, 0)
            cv = cur_bw.get(key, 0)
            ch = pct_change(pv, cv)
            print(f"{label:<40} {pv:>12.2f} {cv:>12.2f} {fmt_pct(ch):>10}")


if __name__ == "__main__":
    main()
