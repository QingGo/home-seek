import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tile_reference import cast, unpack_from_e2m1fn_x2


def quantize_expert_weights(
    weight_dir: str,
    output_dir: str,
    inventory_path: str = None,
    block_size: tuple = (1, 32),
):
    if inventory_path and os.path.exists(inventory_path):
        with open(inventory_path) as f:
            inventory = json.load(f)
    else:
        inventory = None

    os.makedirs(output_dir, exist_ok=True)

    safetensors_files = sorted([f for f in os.listdir(weight_dir) if f.endswith(".safetensors")])
    print(f"[quantize] Found {len(safetensors_files)} safetensors files in {weight_dir}")

    expert_tensors = {}
    shared_tensors = {}

    for fname in safetensors_files:
        fpath = os.path.join(weight_dir, fname)
        print(f"  Scanning {fname}...")
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                is_expert = "mlp.experts" in key
                if is_expert:
                    expert_tensors[key] = fpath
                else:
                    shared_tensors[key] = fpath

    print(f"[quantize] Found {len(expert_tensors)} expert tensors, {len(shared_tensors)} shared tensors")

    quantized_chunks = {}
    chunk_idx = 0
    current_chunk_size = 0
    max_chunk_size = 4 * (1024**3)

    expert_keys_sorted = sorted(expert_tensors.keys())
    total = len(expert_keys_sorted)

    for i, key in enumerate(expert_keys_sorted):
        fpath = expert_tensors[key]
        with safe_open(fpath, framework="pt", device="cpu") as f:
            weight = f.get_tensor(key)

        weight = weight.contiguous().to(torch.float32)
        h, w = weight.shape

        quantized_data, sf = cast(
            weight,
            fmt="e2m1",
            block_size=block_size,
        )

        filename = f"quantized_experts_{chunk_idx:04d}.safetensors"
        if filename not in quantized_chunks:
            quantized_chunks[filename] = {}
            current_chunk_size = 0

        data_size = quantized_data.numel() * quantized_data.element_size()
        sf_size = sf.numel() * sf.element_size()
        item_size = data_size + sf_size

        if current_chunk_size + item_size > max_chunk_size and current_chunk_size > 0:
            chunk_idx += 1
            filename = f"quantized_experts_{chunk_idx:04d}.safetensors"
            quantized_chunks[filename] = {}
            current_chunk_size = 0

        quantized_chunks[filename][f"{key}.quantized"] = quantized_data
        quantized_chunks[filename][f"{key}.sf"] = sf
        current_chunk_size += item_size

        if (i + 1) % 50 == 0:
            print(f"[quantize]  {i+1}/{total} experts quantized")

    print(f"[quantize] Saving {len(quantized_chunks)} chunks...")
    for fname, tensors in quantized_chunks.items():
        fpath = os.path.join(output_dir, fname)
        save_file(tensors, fpath)
        print(f"  Saved {fname} ({len(tensors)} tensors, {os.path.getsize(fpath) / 1e9:.2f} GB)")

    shared_output = os.path.join(output_dir, "shared_weights.safetensors")
    shared_tensors_dict = {}
    for key, fpath in shared_tensors.items():
        with safe_open(fpath, framework="pt", device="cpu") as f:
            shared_tensors_dict[key] = f.get_tensor(key)
    save_file(shared_tensors_dict, shared_output)
    print(f"  Saved shared weights ({len(shared_tensors_dict)} tensors, {os.path.getsize(shared_output) / 1e9:.2f} GB)")

    manifest = {
        "quantized_chunks": list(quantized_chunks.keys()),
        "shared_file": "shared_weights.safetensors",
        "num_experts_quantized": total,
        "block_size": list(block_size),
        "format": "fp4_e2m1",
        "expert_keys": expert_keys_sorted,
    }
    manifest_path = os.path.join(output_dir, "quantize_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[quantize] Manifest saved to {manifest_path}")

    return manifest


def validate_quantization(
    weight_dir: str,
    quantized_dir: str,
    inventory_path: str = None,
    num_samples: int = 10,
    block_size: tuple = (1, 32),
):
    print("[validate] Validating quantization quality...")

    safetensors_files = sorted([f for f in os.listdir(weight_dir) if f.endswith(".safetensors")])
    expert_keys = []
    for fname in safetensors_files:
        with safe_open(os.path.join(weight_dir, fname), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mlp.experts" in key:
                    expert_keys.append((fname, key))
    print(f"[validate] Found {len(expert_keys)} expert tensors total")

    import random
    random.seed(42)
    sample_keys = random.sample(expert_keys, min(num_samples, len(expert_keys)))

    similarities = []
    for fname, key in sample_keys:
        with safe_open(os.path.join(weight_dir, fname), framework="pt", device="cpu") as f:
            original = f.get_tensor(key).to(torch.float32)

        manifest_path = os.path.join(quantized_dir, "quantize_manifest.json")
        with open(manifest_path) as mf:
            manifest = json.load(mf)

        qkey = f"{key}.quantized"
        sfkey = f"{key}.sf"
        quantized_data = None
        sf_data = None
        for chunk_name in manifest["quantized_chunks"]:
            chunk_path = os.path.join(quantized_dir, chunk_name)
            with safe_open(chunk_path, framework="pt", device="cpu") as cf:
                if qkey in cf.keys():
                    quantized_data = cf.get_tensor(qkey)
                if sfkey in cf.keys():
                    sf_data = cf.get_tensor(sfkey)
            if quantized_data is not None:
                break

        if quantized_data is None:
            print(f"  WARNING: Could not find {qkey} in quantized chunks")
            continue

        dequantized = unpack_from_e2m1fn_x2(quantized_data)
        sf_expanded = sf_data.repeat_interleave(block_size[1], dim=1)
        sf_expanded = sf_expanded[:dequantized.shape[0], :dequantized.shape[1]]
        dequantized = dequantized.to(torch.float32) * sf_expanded.to(torch.float32)
        dequantized = dequantized[:original.shape[0], :original.shape[1]]

        cos_sim = torch.nn.functional.cosine_similarity(
            original.flatten().unsqueeze(0),
            dequantized.flatten().unsqueeze(0),
        ).item()
        similarities.append(cos_sim)

        mse = ((original - dequantized) ** 2).mean().item()
        print(f"  {key}: cos_sim={cos_sim:.6f}, mse={mse:.6e}")

    avg_sim = sum(similarities) / len(similarities) if similarities else 0
    print(f"\n[validate] Average cosine similarity across {len(similarities)} experts: {avg_sim:.6f}")
    print(f"[validate] Cosine similarity >= 0.99: {'PASS' if avg_sim >= 0.99 else 'FAIL'}")

    return {"average_cosine_similarity": avg_sim, "num_validated": len(similarities)}


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "quantize"

    if action == "quantize":
        weight_dir = sys.argv[2] if len(sys.argv) > 2 else "weights"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "weights_fp4"
        inventory_path = sys.argv[4] if len(sys.argv) > 4 else "model_inventory.json"
        quantize_expert_weights(weight_dir, output_dir, inventory_path)
    elif action == "validate":
        weight_dir = sys.argv[2] if len(sys.argv) > 2 else "weights"
        quantized_dir = sys.argv[3] if len(sys.argv) > 3 else "weights_fp4"
        num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        validate_quantization(weight_dir, quantized_dir, num_samples=num_samples)
