import os
import json
import torch
from safetensors import safe_open
from collections import OrderedDict


class AllExpertFP4Store:
    def __init__(self, weight_dir: str, config, device: str = "cuda",
                 max_experts: int = 256):
        self.weight_dir = weight_dir
        self.config = config
        self.device = torch.device(device)
        self._max_experts = max_experts
        self._cache = OrderedDict()
        self._build_index()

    def _build_index(self):
        idx_path = os.path.join(self.weight_dir, "model.safetensors.index.json")
        with open(idx_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        self._file_map = {}
        self._layer_files = set()
        for key, fname in weight_map.items():
            if not key.startswith("layers."):
                continue
            parts = key.split(".")
            if len(parts) < 6 or parts[2] != "ffn" or parts[3] != "experts":
                continue
            layer = int(parts[1])
            expert = int(parts[4])
            mat = ".".join(parts[5:])
            filepath = os.path.join(self.weight_dir, fname)
            self._file_map[(layer, expert, mat)] = filepath
            self._layer_files.add((layer, filepath))

    def _short_name(self, full_key: str):
        return ".".join(full_key.split(".")[5:])

    def get_expert_packed(self, layer_idx: int, expert_idx: int):
        cache_key = (layer_idx, expert_idx)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        result = self._load_single_expert(layer_idx, expert_idx)
        if result is None:
            return None

        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._max_experts:
            self._cache.popitem(last=False)
        return result

    def _safetensor_key(self, layer_idx: int, expert_idx: int, mat: str):
        return f"layers.{layer_idx}.ffn.experts.{expert_idx}.{mat}"

    def _load_single_expert(self, layer_idx: int, expert_idx: int):
        target_file = self._file_map.get((layer_idx, expert_idx, "w1.weight"))
        if target_file is None:
            return None

        try:
            with safe_open(target_file, framework="pt", device="cpu") as f:
                def load(mat):
                    k = self._safetensor_key(layer_idx, expert_idx, mat)
                    return f.get_tensor(k) if k in f.keys() else None
                return (
                    load("w1.weight"), load("w1.scale"),
                    load("w3.weight"), load("w3.scale"),
                    load("w2.weight"), load("w2.scale"),
                )
        except Exception:
            return None

    def cache_on_gpu(self, layer_idx: int, expert_idx: int,
                     gate_packed, gate_scale, up_packed, up_scale,
                     down_packed, down_scale):
        cache_key = (layer_idx, expert_idx)
        if cache_key in self._cache:
            return
        gpu_entry = (
            gate_packed.to(self.device, non_blocking=True) if gate_packed is not None else None,
            gate_scale.to(self.device, non_blocking=True) if gate_scale is not None else None,
            up_packed.to(self.device, non_blocking=True) if up_packed is not None else None,
            up_scale.to(self.device, non_blocking=True) if up_scale is not None else None,
            down_packed.to(self.device, non_blocking=True) if down_packed is not None else None,
            down_scale.to(self.device, non_blocking=True) if down_scale is not None else None,
        )
        self._cache[cache_key] = gpu_entry
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._max_experts:
            self._cache.popitem(last=False)

    def get_cache_key(self, layer_idx: int, expert_idx: int):
        cache_key = (layer_idx, expert_idx)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        return None

    def release(self, layer_idx: int, expert_idx: int):
        self._cache.pop((layer_idx, expert_idx), None)

    def clear(self):
        self._cache.clear()

    @property
    def num_cached(self):
        return len(self._cache)

    @property
    def approx_gpu_memory_mb(self):
        I = self.config.moe_intermediate_size
        D = self.config.hidden_size
        w1_bytes = I * (D // 2)
        w3_bytes = I * (D // 2)
        w2_bytes = D * (I // 2)
        per_expert_bytes = w1_bytes + w3_bytes + w2_bytes
        total = self.num_cached * per_expert_bytes
        return total / (1024 * 1024)

    def resize(self, new_max: int):
        self._max_experts = new_max
        while len(self._cache) > self._max_experts:
            self._cache.popitem(last=False)

    @property
    def max_experts(self):
        return self._max_experts
