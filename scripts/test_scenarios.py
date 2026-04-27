import csv
import torch
import numpy as np

from home_seek.inference_engine import HomeSeekInferenceEngine
from home_seek.mem_profiler import MemoryProfiler


def run_scenario(
    engine: HomeSeekInferenceEngine,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    scenario_name: str,
    num_runs: int = 2,
):
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"  Prompt tokens: {input_ids.shape[1]}, Gen tokens: {max_new_tokens}")
    print(f"{'='*60}")

    results = []
    for run in range(num_runs):
        profiler = MemoryProfiler(output_path=f"profile_{scenario_name}_run{run}.csv")
        profiler.start()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)

        result = engine.generate(input_ids, max_new_tokens=max_new_tokens)

        profiler.record("complete")
        profiler.save()

        peak_mem = torch.cuda.max_memory_allocated()
        print(f"  Run {run}: {result['num_generated_tokens']} tokens in {result['total_time_s']:.2f}s, "
              f"peak mem: {peak_mem / (1024**3):.2f} GB")

        results.append({
            "scenario": scenario_name,
            "run": run,
            "prompt_tokens": input_ids.shape[1],
            "generated_tokens": result["num_generated_tokens"],
            "total_time_s": round(result["total_time_s"], 3),
            "new_tokens_per_second": round(result["new_tokens_per_second"], 2),
            "peak_memory_gb": round(peak_mem / (1024**3), 2),
        })

        if run > 0 and run < len(results):
            same = torch.equal(results[0].get("tokens_cpu"), results[run].get("tokens_cpu"))
            print(f"  Deterministic: {same}")
            results[-1]["deterministic"] = same
        elif run == 0:
            results[-1]["deterministic"] = True

        if run == 0:
            results[-1]["tokens_cpu"] = result["tokens"].cpu()
        else:
            results[-1]["tokens_cpu"] = None

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", default="weights")
    parser.add_argument("--quantized-dir", default="weights_fp4")
    parser.add_argument("--output-csv", default="M1_Baseline_Performance.csv")
    parser.add_argument("--quick", action="store_true", help="Run only short scenarios")
    args = parser.parse_args()

    print("[test_scenarios] Initializing inference engine...")
    engine = HomeSeekInferenceEngine(
        weight_dir=args.weight_dir,
        verbose=True,
    )

    all_results = []

    scenarios = [
        ("short_128_16", 128, 16),
        ("medium_8K_16", 8192, 16),
    ]

    if not args.quick:
        scenarios += [
            ("long_128K_8", 131072, 8),
            ("xlong_1M_4", 1048576, 4),
        ]

    for name, ctx_len, gen_len in scenarios:
        input_ids = torch.randint(0, 1000, (1, ctx_len), device=engine.device)
        results = run_scenario(engine, input_ids, gen_len, name)
        all_results.extend(results)

    with open(args.output_csv, "w", newline="") as f:
        if all_results:
            fieldnames = [k for k in all_results[0].keys() if k != "tokens_cpu"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_results:
                w.writerow({k: v for k, v in r.items() if k != "tokens_cpu"})
    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
