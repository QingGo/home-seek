from __future__ import annotations

import torch


class LayerState:
    def __init__(self, device: str = "cuda", active_window: int = 32768):
        self.kv_latent_cache = None
        self.compressed_kv_data = None
        self.compressed_kv_idx = None
        self.compressed_count = 0
        self.archived_kv = None
        self.archived_len = 0
        self.active_window = active_window
        self.device = torch.device(device)

    def append_kv(self, kv_latent: torch.Tensor):
        if self.kv_latent_cache is None:
            self.kv_latent_cache = kv_latent
            if self.archived_len > 0:
                self.kv_latent_cache = torch.cat([self._load_archived(), self.kv_latent_cache], dim=1)
                self.archived_kv = None
                self.archived_len = 0
            return
        self.kv_latent_cache = torch.cat([self.kv_latent_cache, kv_latent], dim=1)
        if self.kv_latent_cache.shape[1] > self.active_window + 1024:
            archive_len = self.kv_latent_cache.shape[1] - self.active_window
            archive_part = self.kv_latent_cache[:, :archive_len, :].contiguous()
            self.archived_len += archive_len
            if self.archived_kv is None:
                self.archived_kv = archive_part.to("cpu", non_blocking=True)
            else:
                self.archived_kv = torch.cat([self.archived_kv, archive_part.to("cpu", non_blocking=True)], dim=0)
            self.kv_latent_cache = self.kv_latent_cache[:, archive_len:, :].contiguous()

    def _load_archived(self) -> torch.Tensor:
        if self.archived_kv is None:
            return torch.zeros(1, 0, self.kv_latent_cache.shape[-1], device=self.device, dtype=torch.bfloat16)
        return self.archived_kv.to(self.device, non_blocking=True)

    def all_kv(self):
        if self.kv_latent_cache is None:
            return None
        if self.archived_kv is not None:
            arch = self._load_archived()
            result = torch.cat([arch, self.kv_latent_cache], dim=1)
            self.archived_kv = None
            self.archived_len = 0
            return result
        return self.kv_latent_cache
