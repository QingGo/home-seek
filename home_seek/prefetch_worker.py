import torch


class PrefetchWorker:
    def __init__(self, loader, device="cuda", num_buffers: int = 2):
        self.prefetch_queue = []

    def preload_all_experts(self, expert_keys):
        pass

    def get_cached(self, layer_idx: int, eid: int):
        return None

    def prefetch_next_layer(self, next_layer: int, expert_ids: list):
        pass

    def clear(self):
        pass

    def shutdown(self):
        pass
