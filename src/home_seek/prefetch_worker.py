import os
import queue
import threading
import torch
from collections import OrderedDict
from safetensors import safe_open

# Share a single disk I/O mutex — only one thread touches the filesystem
# at a time to avoid contention on shared storage.  RLock so the same
# thread can safely call through _load_expert_weights → loader.get_weights
# without deadlocking.
_disk_io_lock = threading.RLock()


class AsyncPrefetchWorker:
    """Background I/O worker: loads expert weights from disk while GPU computes.

    Design:
    - Background daemon thread reads from a task queue (non-blocking push, async pop)
    - All disk I/O coordinated through a shared _disk_io_lock so the background
      thread never contends with the main thread's safetensor reads.
    - `prefetch()` just enqueues tasks — returns immediately, no GPU sync
    - Results stored in a mutex-protected cache
    - `get()` pops from cache under lock
    - `_load_expert_weights` checks prefetch cache BEFORE blocking disk read
    """

    def __init__(self, weight_dir: str, weight_map: dict, device: str = "cuda",
                 num_prefetch: int = 32):
        self.weight_dir = weight_dir
        self.weight_map = weight_map
        self.device = torch.device(device)
        self.num_prefetch = num_prefetch

        self._task_queue = queue.Queue(maxsize=256)
        self._prefetch_cache = {}
        self._cache_order = OrderedDict()
        self._cache_lock = threading.Lock()
        self._running = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ── public API ──────────────────────────────────────────────────────

    @staticmethod
    def acquire_disk_lock():
        """Call before any safetensor read in the main thread."""
        _disk_io_lock.acquire()

    @staticmethod
    def release_disk_lock():
        _disk_io_lock.release()

    def prefetch(self, layer_idx: int, expert_ids: list[int]):
        """Enqueue experts for background loading.  Non-blocking."""
        for eid in expert_ids:
            key = (layer_idx, eid)
            with self._cache_lock:
                if key in self._prefetch_cache:
                    continue
            try:
                self._task_queue.put_nowait(key)
            except queue.Full:
                break

    def get(self, layer_idx: int, eid: int):
        """Try to retrieve a prefetched expert.  Returns dict or None."""
        key = (layer_idx, eid)
        with self._cache_lock:
            cached = self._prefetch_cache.pop(key, None)
            if cached is not None:
                self._cache_order.pop(key, None)
        return cached

    def peek(self, layer_idx: int, eid: int):
        """Check if an expert is in the cache without consuming it."""
        key = (layer_idx, eid)
        with self._cache_lock:
            return key in self._prefetch_cache

    def size(self):
        with self._cache_lock:
            return len(self._prefetch_cache)

    def pending(self):
        return self._task_queue.qsize()

    def clear(self):
        with self._cache_lock:
            self._prefetch_cache.clear()
            self._cache_order.clear()
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self):
        self._running = False
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.clear()

    # ── background worker ───────────────────────────────────────────────

    def _run(self):
        while self._running:
            try:
                key = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if key is None:
                break
            layer_idx, eid = key

            # Only read from disk when the main thread is NOT doing I/O.
            if not _disk_io_lock.acquire(blocking=False):
                # Main thread is on disk → re-queue and yield
                try:
                    self._task_queue.put_nowait(key)
                except queue.Full:
                    pass
                continue

            try:
                weights = self._load_raw(layer_idx, eid)
            finally:
                _disk_io_lock.release()

            if weights:
                with self._cache_lock:
                    if key not in self._prefetch_cache:
                        self._prefetch_cache[key] = weights
                        self._cache_order[key] = None
                    # Evict oldest if cache grows beyond limit
                    max_cached = self.num_prefetch * 3
                    while len(self._prefetch_cache) > max_cached:
                        try:
                            old_key = next(iter(self._cache_order))
                            self._prefetch_cache.pop(old_key, None)
                            self._cache_order.pop(old_key, None)
                        except StopIteration:
                            break

    def _load_raw(self, layer_idx: int, eid: int):
        """Load one expert's weights from safetensors.  Returns dict or None.
        Caller MUST hold _disk_io_lock."""
        prefix = f"layers.{layer_idx}.ffn.experts.{eid}"
        result = {}
        for name in ["w1", "w3", "w2"]:
            w_key = f"{prefix}.{name}.weight"
            s_key = f"{prefix}.{name}.scale"
            fname = self.weight_map.get(w_key)
            if not fname:
                return None
            fpath = os.path.join(self.weight_dir, fname)
            try:
                with safe_open(fpath, framework="pt", device="cpu") as f:
                    result[(name, "data")] = f.get_tensor(w_key)
                    if s_key in f.keys():
                        result[(name, "scale")] = f.get_tensor(s_key)
                    else:
                        result[(name, "scale")] = None
            except Exception:
                return None
        return result


PrefetchWorker = AsyncPrefetchWorker
# Export for main thread use
disk_io_lock = _disk_io_lock
