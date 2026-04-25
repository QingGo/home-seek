import threading
import queue
import torch
from home_seek.inference_engine import load_fp4_weight, load_fp8_weight


class PrefetchWorker:
    def __init__(self, loader, device="cuda"):
        self.loader = loader
        self.device = torch.device(device)
        self.request_queue = queue.Queue(maxsize=4)
        self.result_cache = {}
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _load_expert(self, layer_idx, eid):
        w1 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w1.weight")
        s1 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w1.scale")
        w3 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w3.weight")
        s3 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w3.scale")
        w2 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w2.weight")
        s2 = self.loader.get_weight(f"layers.{layer_idx}.ffn.experts.{eid}.w2.scale")
        if w1 is None:
            return None

        def _load_w(data, scale):
            if data is None:
                return None
            if data.dtype == torch.int8:
                return load_fp4_weight(data, scale)
            return load_fp8_weight(data, scale)

        return (_load_w(w1, s1), _load_w(w3, s3), _load_w(w2, s2))

    def _worker_loop(self):
        while self.running:
            try:
                layer_idx, eid, future_key = self.request_queue.get(timeout=0.1)
                weights = self._load_expert(layer_idx, eid)
                with self.lock:
                    self.result_cache[future_key] = weights
            except queue.Empty:
                continue

    def prefetch(self, layer_idx, eid):
        future_key = (layer_idx, eid)
        with self.lock:
            if future_key in self.result_cache:
                return self.result_cache.pop(future_key)
        try:
            self.request_queue.put_nowait((layer_idx, eid, future_key))
        except queue.Full:
            pass
        return None

    def get_cached(self, layer_idx, eid):
        future_key = (layer_idx, eid)
        with self.lock:
            return self.result_cache.pop(future_key, None)

    def shutdown(self):
        self.running = False
        self.thread.join(timeout=1.0)
