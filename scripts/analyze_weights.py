import json
import os
from safetensors import safe_open


def analyze_weights(weight_dir: str, output_path: str = None):
    config_path = os.path.join(weight_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"[analyze] Loaded config from {config_path}")
    else:
        cfg = {}
        print("[analyze] No config.json found, using empty config")

    inventory = {
        "config": cfg,
        "files": [],
        "tensors": {},
        "summary": {
            "total_parameters": 0,
            "shared_parameters": 0,
            "expert_parameters": 0,
            "num_expert_files": 0,
            "num_shared_files": 0,
        },
    }

    safetensors_files = sorted([f for f in os.listdir(weight_dir) if f.endswith(".safetensors")])
    print(f"[analyze] Found {len(safetensors_files)} safetensors files")

    for fname in safetensors_files:
        fpath = os.path.join(weight_dir, fname)
        file_info = {"file": fname, "tensors": []}
        file_size = os.path.getsize(fpath)
        print(f"  {fname}: {file_size / 1e9:.2f} GB")

        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                shape = list(tensor.shape)
                dtype = str(tensor.dtype)
                numel = tensor.numel()
                inventory["summary"]["total_parameters"] += numel

                entry = {
                    "name": key,
                    "shape": shape,
                    "dtype": dtype,
                    "numel": numel,
                }

                is_expert = "mlp.experts" in key or "expert" in key.lower()
                if is_expert:
                    inventory["summary"]["expert_parameters"] += numel
                else:
                    inventory["summary"]["shared_parameters"] += numel

                file_info["tensors"].append(entry)
                inventory["tensors"][key] = {"shape": shape, "dtype": dtype, "file": fname}

        inventory["files"].append(file_info)

    num_routed = cfg.get("n_routed_experts", 256)
    num_layers = cfg.get("num_hidden_layers", 43)
    hidden = cfg.get("hidden_size", 4096)
    intermediate = cfg.get("moe_intermediate_size", 2048)
    expert_params_per = 3 * hidden * intermediate
    total_expert_params = num_routed * num_layers * expert_params_per

    inventory["analysis"] = {
        "num_layers": num_layers,
        "hidden_size": hidden,
        "num_routed_experts": num_routed,
        "expert_params_per_expert": expert_params_per,
        "total_expert_params_expected": total_expert_params,
        "params_per_expert_moe_gate_up": f"{hidden}x{intermediate}",
    }

    total = inventory["summary"]["total_parameters"]
    expected = cfg.get("total_parameters", 284000000000)
    error_pct = abs(total - expected) / expected * 100
    inventory["summary"]["parameter_count_error_pct"] = round(error_pct, 2)

    print(f"\n[analyze] Total params: {total / 1e9:.2f}B")
    print(f"[analyze] Shared params: {inventory['summary']['shared_parameters'] / 1e9:.2f}B")
    print(f"[analyze] Expert params: {inventory['summary']['expert_parameters'] / 1e9:.2f}B")
    print(f"[analyze] Error vs expected ({expected / 1e9:.2f}B): {error_pct:.2f}%")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(inventory, f, indent=2)
        print(f"[analyze] Saved inventory to {output_path}")

    return inventory


if __name__ == "__main__":
    import sys
    weight_dir = sys.argv[1] if len(sys.argv) > 1 else "weights"
    output = sys.argv[2] if len(sys.argv) > 2 else "model_inventory.json"
    analyze_weights(weight_dir, output)
