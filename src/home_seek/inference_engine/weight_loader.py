from __future__ import annotations

import os
import json
import logging
import torch
from safetensors import safe_open
from collections import defaultdict
from home_seek._fp4 import unpack_from_e2m1fn_x2, cast_back

_logger = logging.getLogger(__name__)


def _ue8m0_to_f32(sf: torch.Tensor) -> torch.Tensor:
    if sf.element_size() != 1:
        return sf.to(torch.float32)
    sf_u8 = sf.view(torch.uint8)
    sf_i32 = sf_u8.to(torch.int32)
    return (sf_i32 << 23).view(torch.float32)


def load_fp8_weight(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if data is None:
        return None
    if data.dtype == torch.bfloat16 or data.dtype == torch.float32:
        return data.to(torch.bfloat16)
    if data.dtype == torch.float8_e4m3fn:
        if scale is None:
            return data.to(torch.bfloat16)
        sf = _ue8m0_to_f32(scale)
        return cast_back((data, sf), 'bf16', (128, 128))
    return data.to(torch.bfloat16)


def load_fp4_weight(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if data.dtype == torch.bfloat16:
        return data
    deq = unpack_from_e2m1fn_x2(data)
    if scale.numel() == 1:
        deq = deq * scale.to(torch.float32).item()
    else:
        block_size = 32
        sf = scale.repeat_interleave(block_size, dim=1) if scale.dim() == 2 else scale
        deq = deq.to(torch.float32) * sf.to(torch.float32)
    return deq.to(torch.bfloat16)


class WeightLoader:
    def __init__(self, weight_dir: str, device: str = "cuda"):
        self.weight_dir = weight_dir
        self.device = torch.device(device)
        self.weight_map = {}
        self._mmap_cache = {}
        self._build_index()

    def _log(self, msg):
        _logger.info(msg)

    def _build_index(self):
        idx_path = os.path.join(self.weight_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                data = json.load(f)
            self.weight_map = data["weight_map"]
            self.total_size = data["metadata"]["total_size"]
            return

        safetensors_files = sorted([f for f in os.listdir(self.weight_dir) if f.endswith(".safetensors")])
        for fname in safetensors_files:
            fpath = os.path.join(self.weight_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with safe_open(fpath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        self.weight_map[key] = fname
            except Exception:
                pass

    def _open_mmap(self, fname: str):
        if fname not in self._mmap_cache:
            fpath = os.path.join(self.weight_dir, fname)
            self._mmap_cache[fname] = safe_open(fpath, framework="pt", device="cpu")
        return self._mmap_cache[fname]

    def get_weight(self, key: str, device: str = None):
        fname = self.weight_map.get(key)
        if fname is None:
            return None
        try:
            f = self._open_mmap(fname)
            tensor = f.get_tensor(key)
            if device is not None and str(device) != "cpu":
                return tensor.to(device, non_blocking=True)
            return tensor
        except Exception as e:
            self._log(f"Failed to get weight {key} from {fname}: {type(e).__name__}")
            return None

    def get_weights(self, *keys):
        results = {}
        file_keys = defaultdict(list)
        for k in keys:
            fname = self.weight_map.get(k)
            if fname:
                file_keys[fname].append(k)
        for fname, ks in file_keys.items():
            try:
                f = self._open_mmap(fname)
                for k in ks:
                    results[k] = f.get_tensor(k)
            except Exception as e:
                self._log(f"Failed batch load from {fname}: {type(e).__name__}")
        return results

    def close(self):
        self._mmap_cache.clear()

    def _safetensors_files(self):
        """Return list of all safetensor file paths in weight_dir."""
        weight_dir = self.weight_dir
        files = [os.path.join(weight_dir, f)
                 for f in sorted(os.listdir(weight_dir))
                 if f.endswith(".safetensors")]
        return [f for f in files if os.path.isfile(f)]

    def get_layer_weight(self, layer: int, weight_type: str):
        return self.get_weight(f"layers.{layer}.{weight_type}")

    def get_attn_weight(self, layer: int, name: str):
        return self.get_weight(f"layers.{layer}.attn.{name}")

    def get_ffn_weight(self, layer: int, name: str):
        return self.get_weight(f"layers.{layer}.ffn.{name}")
