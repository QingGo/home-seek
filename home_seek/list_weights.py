import os
from safetensors import safe_open


def build_weight_inventory(weight_dir: str):
    safetensors_files = sorted([f for f in os.listdir(weight_dir) if f.endswith(".safetensors")])
    if not safetensors_files:
        print(f"[inventory] No safetensors files found in {weight_dir}")
        return None

    print(f"[inventory] Scanning {len(safetensors_files)} files...")
    total_params = 0

    for fname in safetensors_files:
        fpath = os.path.join(weight_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024**2)
        try:
            with safe_open(fpath, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                print(f"\n  {fname} ({size_mb:.0f} MB): {len(keys)} tensors")
                for k in keys:
                    t = f.get_tensor(k)
                    numel = t.numel()
                    total_params += numel
                    dims = " × ".join(str(s) for s in t.shape)
                    print(f"    {k}: [{dims}], {t.dtype}")
        except Exception as e:
            print(f"  {fname}: ERROR - {e}")

    print(f"\n[inventory] Total parameters: {total_params / 1e9:.2f}B")
    return total_params


if __name__ == "__main__":
    import sys
    weight_dir = sys.argv[1] if len(sys.argv) > 1 else "weights"
    build_weight_inventory(weight_dir)
