import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from home_seek.inference_engine import HomeSeekInferenceEngine


def run_stress_test(weight_dir: str, quantized_dir: str, output_png: str = "memory_vs_context_length.png"):
    print("[stress_test] Running memory stress test...")

    engine = HomeSeekInferenceEngine(
        weight_dir=weight_dir,
        verbose=True,
    )

    context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    try:
        context_lengths += [65536, 131072, 262144, 524288, 1048576]
    except Exception:
        pass

    results = []
    for ctx_len in context_lengths:
        print(f"\n{'='*50}")
        print(f"Testing context length: {ctx_len}")
        print(f"{'='*50}")

        try:
            input_ids = torch.randint(0, 1000, (1, ctx_len), device=engine.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            result = engine.generate(input_ids, max_new_tokens=1)

            peak = torch.cuda.max_memory_allocated()
            allocated = torch.cuda.memory_allocated()

            entry = {
                "context_length": ctx_len,
                "peak_memory_gb": round(peak / (1024**3), 2),
                "allocated_memory_gb": round(allocated / (1024**3), 2),
                "time_s": round(result["total_time_s"], 3),
                "success": True,
            }
            print(f"  Peak: {entry['peak_memory_gb']:.2f} GB, "
                  f"Allocated: {entry['allocated_memory_gb']:.2f} GB, "
                  f"Time: {entry['time_s']:.2f}s")

        except Exception as e:
            print(f"  FAILED at {ctx_len}: {e}")
            entry = {
                "context_length": ctx_len,
                "peak_memory_gb": None,
                "allocated_memory_gb": None,
                "time_s": None,
                "success": False,
                "error": str(e),
            }

        results.append(entry)

    with open("stress_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to stress_test_results.json")

    successful = [r for r in results if r["success"]]
    if successful:
        ctx = [r["context_length"] for r in successful]
        peak = [r["peak_memory_gb"] for r in successful]

        plt.figure(figsize=(10, 6))
        plt.semilogx(ctx, peak, "bo-", linewidth=2, markersize=8)
        plt.axhline(y=22.5, color="r", linestyle="--", label="Budget (22.5 GB)")
        plt.axhline(y=24.0, color="orange", linestyle="--", label="Hard limit (24 GB)")
        plt.xlabel("Context Length (tokens)", fontsize=12)
        plt.ylabel("Peak Memory (GB)", fontsize=12)
        plt.title("Context Length vs. Peak Memory Usage", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        print(f"Chart saved to {output_png}")
    else:
        print("No successful runs to plot")

    return results


if __name__ == "__main__":
    import sys
    weight_dir = sys.argv[1] if len(sys.argv) > 1 else "weights"
    quantized_dir = sys.argv[2] if len(sys.argv) > 2 else "weights_fp4"
    run_stress_test(weight_dir, quantized_dir)
