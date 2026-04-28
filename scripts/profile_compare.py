"""Compare current profile against previous baseline."""
import json
import sys

prev_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/prev_profile.json"
cur_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/last_profile.json"

try:
    with open(prev_path) as f:
        prev = json.load(f)["performance"]
except (FileNotFoundError, json.JSONDecodeError):
    print(f"No previous profile found at {prev_path}. Run make profile first.")
    sys.exit(1)

with open(cur_path) as f:
    cur = json.load(f)["performance"]

print(f"{'Metric':<35} {'Previous':<12} {'Current':<12} {'Change':<10}")
print("-" * 70)
for key in ["tokens_per_second", "ms_per_token", "peak_memory_gb"]:
    pv = prev.get(key, 0)
    cv = cur.get(key, 0)
    if pv == 0:
        ch = "N/A"
    else:
        pct = (cv - pv) / pv * 100
        ch = f"{pct:+.1f}%"
    print(f"{key:<35} {pv:<12.4f} {cv:<12.4f} {ch:<10}")
