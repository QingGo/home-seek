import os
import sys


def download_weights(model_id: str = "deepseek-ai/DeepSeek-V4-Flash"):
    modelscope_cache = os.environ.get("MODELSCOPE_CACHE", "")
    if modelscope_cache:
        os.environ["MODELSCOPE_CACHE"] = modelscope_cache
        if not os.path.exists(modelscope_cache):
            os.makedirs(modelscope_cache, exist_ok=True)
        print(f"[download] MODELSCOPE_CACHE={modelscope_cache}")
    else:
        modelscope_cache = os.path.join(os.getcwd(), "weights")
        os.environ["MODELSCOPE_CACHE"] = modelscope_cache
        os.makedirs(modelscope_cache, exist_ok=True)
        print(f"[download] Using default cache: {modelscope_cache}")

    try:
        from modelscope.hub.snapshot_download import snapshot_download
        print(f"[download] Downloading {model_id}...")
        model_path = snapshot_download(model_id)
        print(f"[download] Model downloaded to: {model_path}")
    except Exception as e:
        print(f"[download] modelscope failed: {e}")
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_id, cache_dir=modelscope_cache)
            print(f"[download] Model downloaded to: {model_path}")
        except Exception as e2:
            print(f"[download] huggingface_hub also failed: {e2}")
            sys.exit(1)

    output_txt = os.path.join(model_path, "weights_inventory.txt")
    with open(output_txt, "w") as f:
        for root, dirs, files in os.walk(model_path):
            for fn in sorted(files):
                fpath = os.path.join(root, fn)
                size = os.path.getsize(fpath)
                f.write(f"{fn}\t{size}\n")
    print(f"[download] Inventory saved to {output_txt}")
    return model_path


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-V4-Flash"
    download_weights(model_id)
