import threading
import queue
import torch


class PrefetchWorker:
    def __init__(self, loader, device="cuda", num_buffers: int = 2):
        self.loader = loader
        self.device = torch.device(device)
        self.num_buffers = num_buffers
        self.buffers = [{} for _ in range(num_buffers)]
        self.current_buffer = 0
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.lock = threading.Lock()
        self.prefetch_queue = queue.Queue(maxsize=8)
        self.completed = queue.Queue(maxsize=8)
        self.active_prefetches = set()
        self.thread.start()

    def _load_expert_weights(self, layer_idx: int, eid: int):
        prefix = f"layers.{layer_idx}.ffn.experts.{eid}"
        keys = [f"{prefix}.w1.weight", f"{prefix}.w1.scale",
                f"{prefix}.w3.weight", f"{prefix}.w3.scale",
                f"{prefix}.w2.weight", f"{prefix}.w2.scale"]
        tensors = self.loader.get_weights(*keys)
        w1 = tensors.get(keys[0]); s1 = tensors.get(keys[1])
        w3 = tensors.get(keys[2]); s3 = tensors.get(keys[3])
        w2 = tensors.get(keys[4]); s2 = tensors.get(keys[5])
        if w1 is None:
            return None
        def _load(data, scale):
            if data is None:
                return None
            if data.dtype == torch.int8:
                from home_seek.inference_engine import load_fp4_weight
                return load_fp4_weight(data, scale)
            from home_seek.inference_engine import load_fp8_weight
            return load_fp8_weight(data, scale)
        return (layer_idx, eid, _load(w1, s1), _load(w3, s3), _load(w2, s2))

    def _worker_loop(self):
        while self.running:
            try:
                item = self.prefetch_queue.get(timeout=0.05)
                if item is None:
                    continue
                layer_idx, eid, buf_idx, key = item
                result = self._load_expert_weights(layer_idx, eid)
                if result is not None:
                    self.completed.put(result)
                with self.lock:
                    self.active_prefetches.discard(key)
            except queue.Empty:
                continue

    def prefetch(self, layer_idx: int, eid: int):
        key = (layer_idx, eid)
        with self.lock:
            if key in self.active_prefetches:
                return
            self.active_prefetches.add(key)
        for buf in self.buffers:
            if key in buf:
                return
        buf_idx = (self.current_buffer + 1) % self.num_buffers
        try:
            self.prefetch_queue.put_nowait((layer_idx, eid, buf_idx, key))
        except queue.Full:
            with self.lock:
                self.active_prefetches.discard(key)

    def get_cached(self, layer_idx: int, eid: int):
        key = (layer_idx, eid)
        for buf in self.buffers:
            if key in buf:
                return buf[key]
        try:
            while True:
                result = self.completed.get_nowait()
                if result is not None:
                    _, _, w1, w3, w2 = result
                    self._store_in_buffer(layer_idx, eid, w1, w3, w2)
                    if layer_idx == result[0] and eid == result[1]:
                        return (w1, w3, w2)
        except queue.Empty:
            pass
        return None

    def _store_in_buffer(self, layer_idx: int, eid: int, w1, w3, w2):
        key = (layer_idx, eid)
        with self.lock:
            self.buffers[self.current_buffer][key] = (w1, w3, w2)

    def prefetch_next_layer(self, next_layer: int, expert_ids: list):
        if len(self.active_prefetches) >= 16:
            return
        for eid in expert_ids:
            if len(self.active_prefetches) >= 16:
                break
            self.prefetch(next_layer, eid)

    def clear(self):
        with self.lock:
            for buf in self.buffers:
                buf.clear()
            self.active_prefetches.clear()
        try:
            while True:
                self.completed.get_nowait()
        except queue.Empty:
            pass

    def shutdown(self):
        self.running = False
        self.thread.join(timeout=2.0)
