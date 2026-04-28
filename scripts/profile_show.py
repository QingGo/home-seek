"""Display last profile summary."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/last_profile.json"
with open(path) as f:
    d = json.load(f)
p = d["performance"]
print(f"\n=== Profile saved to {path} ===")
print(f"  Decode throughput: {p['tokens_per_second']:.2f} t/s")
print(f"  Peak memory: {p['peak_memory_gb']:.2f} GB")
