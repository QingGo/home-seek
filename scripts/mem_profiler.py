import time
import csv
import torch
from typing import Callable


class MemoryProfiler:
    def __init__(self, output_path: str = "memory_profile.csv"):
        self.output_path = output_path
        self.records = []
        self._start_time = None

    def start(self):
        torch.cuda.reset_peak_memory_stats()
        self._start_time = time.time()
        self.record("start")

    def record(self, label: str = ""):
        t = time.time() - (self._start_time or time.time())
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        self.records.append({
            "time_s": round(t, 3),
            "label": label,
            "allocated_bytes": allocated,
            "allocated_gb": allocated / (1024**3),
            "reserved_bytes": reserved,
            "reserved_gb": reserved / (1024**3),
            "peak_bytes": peak,
            "peak_gb": peak / (1024**3),
        })

    def save(self):
        if not self.records:
            return
        with open(self.output_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.records[0].keys())
            w.writeheader()
            w.writerows(self.records)
        print(f"[mem_profiler] Saved {len(self.records)} records to {self.output_path}")

    def profile(self, fn: Callable, *args, **kwargs):
        self.start()
        try:
            result = fn(*args, **kwargs)
            return result
        finally:
            self.record("done")
            self.save()
