import os
import torch
from collections import OrderedDict
from safetensors import safe_open


class AsyncPrefetchWorker:
    def __init__(self, weight_dir: str, weight_map: dict, device: str = "cuda",
                 num_prefetch: int = 6):
        self.weight_dir = weight_dir
        self.weight_map = weight_map
        self.device = torch.device(device)
        self.num_prefetch = num_prefetch

        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._prefetch_cache = {}
        self._pending_keys = []
        self._cache_order = OrderedDict()

    def prefetch(self, layer_idx: int, expert_ids: list[int]):
        if not self._stream:
            return
        keys = [(layer_idx, eid) for eid in expert_ids
                if (layer_idx, eid) not in self._prefetch_cache]
        self._pending_keys.extend(keys)
        self._execute_prefetch()

    def _execute_prefetch(self):
        if not self._pending_keys:
            return
        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(self._stream):
            cnt = 0
            while self._pending_keys and cnt < self.num_prefetch:
                key = self._pending_keys.pop(0)
                if key in self._prefetch_cache:
                    continue
                weights = self._load_raw(key[0], key[1])
                if weights:
                    self._prefetch_cache[key] = weights
                    self._cache_order[key] = None
                cnt += 1

    def get(self, layer_idx: int, eid: int):
        key = (layer_idx, eid)
        cached = self._prefetch_cache.pop(key, None)
        if cached is not None:
            self._cache_order.pop(key, None)
        return cached

    def _load_raw(self, layer_idx: int, eid: int):
        prefix = f"layers.{layer_idx}.ffn.experts.{eid}"
        result = {}
        for name in ["w1", "w3", "w2"]:
            w_key = f"{prefix}.{name}.weight"
            s_key = f"{prefix}.{name}.scale"
            fname = self.weight_map.get(w_key)
            if not fname:
                return None
            fpath = os.path.join(self.weight_dir, fname)
            try:
                with safe_open(fpath, framework="pt", device="cpu") as f:
                    result[(name, "data")] = f.get_tensor(w_key)
                    if s_key in f.keys():
                        result[(name, "scale")] = f.get_tensor(s_key)
                    else:
                        result[(name, "scale")] = None
            except Exception:
                return None
        return result

    def clear(self):
        self._pending_keys.clear()
        self._prefetch_cache.clear()
        self._cache_order.clear()

    def shutdown(self):
        self.clear()
        self._stream = None


PrefetchWorker = AsyncPrefetchWorker
