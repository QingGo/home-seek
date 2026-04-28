from __future__ import annotations
from collections import OrderedDict
from typing import Callable
import torch


class ExpertWeightCache:
    """CPU FP4 expert cache with LRU eviction + pinned entries."""

    def __init__(self, max_experts: int = 512, device: str = "cuda",
                 hot_deq_size: int = 0):
        self.max_experts = max_experts
        self.device = torch.device(device)
        self.cache = OrderedDict()
        self.pinned = set()
        self._hot_deq = OrderedDict()
        self._max_hot_deq = hot_deq_size

    def get(self, key: str):
        if key not in self.cache:
            return None
        try:
            self.cache.move_to_end(key)
            return self.cache[key]
        except KeyError:
            return None

    def deq(self, key: str):
        if self._max_hot_deq > 0 and key in self._hot_deq:
            self._hot_deq.move_to_end(key)
            return self._hot_deq[key]
        entry = self.get(key)
        if entry is None:
            return None
        raw_w1, raw_w3, raw_w2 = entry
        w1 = self._dequantize_entry(raw_w1, self.device)
        w3 = self._dequantize_entry(raw_w3, self.device)
        w2 = self._dequantize_entry(raw_w2, self.device)
        if self._max_hot_deq > 0:
            self._hot_deq[key] = (w1, w3, w2)
            if len(self._hot_deq) > self._max_hot_deq:
                for k in list(self._hot_deq.keys()):
                    if k not in self.pinned:
                        self._hot_deq.pop(k)
                        break
                else:
                    self._hot_deq.popitem(last=False)
        return (w1, w3, w2)

    def _dequantize_entry(self, entry, device=None):
        if entry is None:
            return None
        from home_seek.inference_engine.weight_loader import load_fp8_weight, load_fp4_weight
        data, scale, fmt = entry
        dev = device if device is not None else data.device
        if fmt == "bf16":
            return data
        if fmt == "fp8":
            if str(data.device) != str(dev):
                data = data.to(dev, non_blocking=True)
                scale = scale.to(dev, non_blocking=True) if scale is not None else None
            return load_fp8_weight(data, scale)
        if fmt in ("fp4_gpu", "fp4"):
            if str(data.device) != str(dev):
                data = data.to(dev, non_blocking=True)
                scale = scale.to(dev, non_blocking=True) if scale is not None else None
            return load_fp4_weight(data, scale)
        return data.to(torch.bfloat16)

    def put(self, key: str, w1_entry, w3_entry, w2_entry, pin: bool = False):
        if key in self.cache:
            try:
                self.cache.move_to_end(key)
            except KeyError:
                pass
            return
        if not pin and len(self.cache) >= self.max_experts:
            for k in list(self.cache.keys()):
                if k not in self.pinned:
                    try:
                        self.cache.pop(k)
                        self._hot_deq.pop(k, None)
                    except KeyError:
                        pass
                    break
        self.cache[key] = (w1_entry, w3_entry, w2_entry)
        if pin:
            self.pinned.add(key)

    def put_deq(self, key: str, w1_d, w3_d, w2_d, pin: bool = False):
        self.put(key, (w1_d, None, "bf16"), (w3_d, None, "bf16"), (w2_d, None, "bf16"), pin=pin)

    def clear(self):
        pinned = {k: self.cache[k] for k in list(self.cache.keys()) if k in self.pinned}
        self.cache.clear()
        self.cache.update(pinned)
        self._hot_deq.clear()
        self.pinned.clear()
        for k in pinned:
            self.pinned.add(k)

    def trim(self, target_count: int = 64):
        excess = len(self.cache) - len(self.pinned) - target_count
        if excess <= 0:
            return
        for k in list(self.cache.keys()):
            if k in self.pinned:
                continue
            if excess <= 0:
                break
            self.cache.pop(k)
            self._hot_deq.pop(k, None)
            excess -= 1

    def __len__(self):
        return len(self.cache)


class ExpertCacheManager:
    """Unified expert cache: GPU hot → GPU LRU BF16 → CPU FP4 → disk.

    Single entry point for all expert weight access.
    """

    def __init__(self, max_gpu_hot: int = 64, max_gpu_bf16: int = 100,
                 max_cpu_fp4: int = 5120, device: str = "cuda"):
        self.device = torch.device(device)
        self._gpu_hot: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._gpu_bf16: OrderedDict = OrderedDict()
        self._max_gpu_hot = max_gpu_hot
        self._max_gpu_bf16 = max_gpu_bf16
        self._cpu_fp4 = ExpertWeightCache(max_experts=max_cpu_fp4, device=device)
        self._hot_set_by_layer: dict[int, set[int]] = {}
        self._global_hot_set: set[int] = set()
        self._num_hash_layers = 0
        self._load_from_disk: Callable | None = None
        self._make_raw_entry: Callable | None = None
        self._dequantize_fn: Callable | None = None

    def configure(self, hot_set_by_layer: dict[int, set[int]],
                  global_hot_set: set[int], num_hash_layers: int,
                  load_from_disk: Callable, make_raw_entry: Callable,
                  dequantize_fn: Callable):
        self._hot_set_by_layer = hot_set_by_layer
        self._global_hot_set = global_hot_set
        self._num_hash_layers = num_hash_layers
        self._load_from_disk = load_from_disk
        self._make_raw_entry = make_raw_entry
        self._dequantize_fn = dequantize_fn

    def get(self, layer: int, eid: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Unified expert weight retrieval: GPU hot → GPU BF16 → CPU → disk."""
        key = (layer, eid)
        cached = self._gpu_hot.get(key)
        if cached is not None:
            return cached
        cached = self._gpu_bf16.get(key)
        if cached is not None:
            self._gpu_bf16.move_to_end(key)
            return cached
        return self._load_and_cache(layer, eid)

    def _load_and_cache(self, layer: int, eid: int
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self._load_from_disk is None or self._make_raw_entry is None:
            return None
        raw = self._load_from_disk(layer, eid)
        if raw is None:
            return None
        w1_entry, w3_entry, w2_entry = raw
        if w1_entry is None:
            return None
        w1_data, w1_scale, w1_fmt = w1_entry
        w3_data, w3_scale, w3_fmt = w3_entry
        w2_data, w2_scale, w2_fmt = w2_entry
        is_fp4 = all(f in ("fp4_gpu", "fp4") for f in (w1_fmt, w3_fmt, w2_fmt))
        if not is_fp4:
            return None
        w1_data = self._to_device(w1_data); w1_scale = self._to_device(w1_scale, True)
        w3_data = self._to_device(w3_data); w3_scale = self._to_device(w3_scale, True)
        w2_data = self._to_device(w2_data); w2_scale = self._to_device(w2_scale, True)
        w1_scale = self._to_f32(w1_scale); w3_scale = self._to_f32(w3_scale); w2_scale = self._to_f32(w2_scale)
        if self._dequantize_fn is not None:
            w1_b, w3_b, w2_b = self._dequantize_fn(w1_data, w1_scale, w3_data, w3_scale, w2_data, w2_scale)
        else:
            return None
        cache_key = (layer, eid)
        layer_set = self._hot_set_by_layer.get(layer, self._global_hot_set)
        if eid in layer_set:
            self._gpu_hot[cache_key] = (w1_b, w3_b, w2_b)
            if len(self._gpu_hot) > self._max_gpu_hot:
                self._gpu_hot.pop(next(iter(self._gpu_hot)))
        else:
            self._gpu_bf16[cache_key] = (w1_b, w3_b, w2_b)
            if len(self._gpu_bf16) > self._max_gpu_bf16:
                self._gpu_bf16.pop(next(iter(self._gpu_bf16)))
        return (w1_b, w3_b, w2_b)

    def _to_device(self, t, is_scale: bool = False):
        if t is None:
            return None
        if t.device.type != self.device.type:
            return t.to(self.device, non_blocking=True)
        return t

    def _to_f32(self, t):
        if t is None:
            return None
        return t.to(torch.float32) if t.dtype != torch.float32 else t

    def warmup_hot(self, layer: int, eids: list[int]):
        """Pre-load specific experts into GPU hot cache."""
        for eid in eids:
            self.get(layer, eid)

    def clear(self):
        self._gpu_hot.clear()
        self._gpu_bf16.clear()
        self._cpu_fp4.clear()

    def cpu_cache(self) -> ExpertWeightCache:
        return self._cpu_fp4

    def stats(self) -> dict:
        return {
            "gpu_hot_entries": len(self._gpu_hot),
            "gpu_bf16_entries": len(self._gpu_bf16),
            "cpu_fp4_entries": len(self._cpu_fp4),
        }
