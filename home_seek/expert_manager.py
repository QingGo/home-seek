import os
import json
import torch
from safetensors import safe_open
from collections import OrderedDict
from tile_reference import unpack_from_e2m1fn_x2


class ExpertMemoryManager:
    def __init__(
        self,
        quantized_dir: str,
        weight_dir: str,
        config,
        max_cache_gb: float = 6.0,
        device: str = "cuda",
    ):
        self.quantized_dir = quantized_dir
        self.weight_dir = weight_dir
        self.config = config
        self.device = torch.device(device)
        self.max_cache_bytes = int(max_cache_gb * (1024**3))

        cache = OrderedDict()
        self._load_manifest()

        self.hash_expert_ids = self._compute_hash_expert_ids()

        self.shared_gate = None
        self.shared_gate_proj = None
        self.shared_up_proj = None
        self.shared_down_proj = None

    def _load_manifest(self):
        manifest_path = os.path.join(self.quantized_dir, "quantize_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = None

    def _compute_hash_expert_ids(self):
        ids = []
        for layer_idx in range(min(3, self.config.num_hidden_layers)):
            for k in range(6):
                expert_id = (layer_idx * 256 + k) % 256
                ids.append((layer_idx, expert_id))
        return ids

    def load_shared_weights(self):
        shared_path = os.path.join(self.quantized_dir, "shared_weights.safetensors")
        if not os.path.exists(shared_path):
            shared_path = os.path.join(self.weight_dir, self._find_shared_file())
        if os.path.exists(shared_path):
            print(f"[expert_manager] Loading shared weights from {shared_path}")
            with safe_open(shared_path, framework="pt", device=str(self.device)) as f:
                for key in f.keys():
                    if "shared_expert.gate_proj" in key:
                        self.shared_gate_proj = f.get_tensor(key).to(torch.bfloat16)
                    elif "shared_expert.up_proj" in key:
                        self.shared_up_proj = f.get_tensor(key).to(torch.bfloat16)
                    elif "shared_expert.down_proj" in key:
                        self.shared_down_proj = f.get_tensor(key).to(torch.bfloat16)
                    elif "shared_expert.gate" in key:
                        self.shared_gate = f.get_tensor(key).to(torch.bfloat16)
            print(f"[expert_manager] Shared weights loaded")
        else:
            print(f"[expert_manager] WARNING: No shared weights found at {shared_path}")

    def _find_shared_file(self):
        for fname in sorted(os.listdir(self.weight_dir)):
            if fname.endswith(".safetensors"):
                fpath = os.path.join(self.weight_dir, fname)
                try:
                    with safe_open(fpath, framework="pt", device="cpu") as f:
                        keys = f.keys()
                        if any("shared_expert" in k for k in keys):
                            return fname
                except Exception:
                    continue
        return ""

    def get_expert_weight(self, layer_idx: int, expert_idx: int, precision: str = "fp4"):
        if self.manifest and precision == "fp4":
            key_pattern = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            for chunk_name in self.manifest["quantized_chunks"]:
                chunk_path = os.path.join(self.quantized_dir, chunk_name)
                try:
                    with safe_open(chunk_path, framework="pt", device=str(self.device)) as f:
                        gate_key = f"{key_pattern}.gate_proj"
                        up_key = f"{key_pattern}.up_proj"
                        down_key = f"{key_pattern}.down_proj"

                        if all(k in f.keys() for k in [f"{gate_key}.quantized", f"{up_key}.quantized", f"{down_key}.quantized"]):
                            gate_data = f.get_tensor(f"{gate_key}.quantized")
                            gate_sf = f.get_tensor(f"{gate_key}.sf")
                            up_data = f.get_tensor(f"{up_key}.quantized")
                            up_sf = f.get_tensor(f"{up_key}.sf")
                            down_data = f.get_tensor(f"{down_key}.quantized")
                            down_sf = f.get_tensor(f"{down_key}.sf")

                            gate_w = self._dequantize_fp4(gate_data, gate_sf)
                            up_w = self._dequantize_fp4(up_data, up_sf)
                            down_w = self._dequantize_fp4(down_data, down_sf)

                            return ExpertWeight(gate_w, up_w, down_w)
                except Exception as e:
                    continue

        key_pattern = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        for fname in sorted(os.listdir(self.weight_dir)):
            if not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(self.weight_dir, fname)
            try:
                with safe_open(fpath, framework="pt", device=str(self.device)) as f:
                    keys_in_file = f.keys()
                    gate_key = f"{key_pattern}.gate_proj"
                    up_key = f"{key_pattern}.up_proj"
                    down_key = f"{key_pattern}.down_proj"
                    if gate_key in keys_in_file:
                        return ExpertWeight(
                            f.get_tensor(gate_key).to(torch.bfloat16),
                            f.get_tensor(up_key).to(torch.bfloat16),
                            f.get_tensor(down_key).to(torch.bfloat16),
                        )
            except Exception:
                continue
        raise KeyError(f"Expert {key_pattern} not found")

    def _dequantize_fp4(self, data: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
        deq = unpack_from_e2m1fn_x2(data)
        block_size = 32
        sf_expanded = sf.repeat_interleave(block_size, dim=1)
        deq = deq.to(torch.float32) * sf_expanded.to(torch.float32)
        return deq.to(torch.bfloat16)


class ExpertWeight:
    __slots__ = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, gate_proj, up_proj, down_proj):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
