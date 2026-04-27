"""Hybrid KV Cache: paged compressed cache + SWA state + disk offloading.

Block-aligned design per arch_design.md Section 3.2:
- Block size P = lcm(4, 128) = 128 original tokens
- CSA produces P/4 = 32 compressed entries per block
- HCA produces P/128 = 1 compressed entry per block
- SWA caches last 128 tokens in circular buffers

Key classes:
- PagedCompressedBlock: one block of compressed KV
- PagedCompressedCache: manages blocks per layer type
- SWACache: sliding window KV cache
- HybridKVCache: per-layer coordinator
"""

from __future__ import annotations

import torch
import time
from typing import Dict


# Block covering 128 original tokens (LCM of 4 and 128)
BLOCK_SIZE_TOKENS = 128
_CSA_PER_BLOCK = BLOCK_SIZE_TOKENS // 4   # 32 compressed entries
_HCA_PER_BLOCK = BLOCK_SIZE_TOKENS // 128  # 1 compressed entry


class PagedCompressedBlock:
    """One block of compressed KV entries.

    For CSA layers: stores *both* compressed attention KV and indexer keys.
    For HCA layers: only compressed attention KV.
    """
    __slots__ = (
        "block_id", "data", "indexer_keys", "num_entries",
        "last_access_step", "on_gpu", "layer_type",
    )

    def __init__(self, block_id: int, data: torch.Tensor,
                 indexer_keys: torch.Tensor | None, num_entries: int,
                 layer_type: str):
        self.block_id = block_id
        self.data = data                     # [entries_per_block, head_dim] on GPU
        self.indexer_keys = indexer_keys     # [entries_per_block, idx_dim] or None
        self.num_entries = num_entries
        self.last_access_step = 0
        self.on_gpu = True
        self.layer_type = layer_type  # "csa" or "hca"

    @property
    def memory_bytes(self) -> int:
        b = self.data.numel() * self.data.element_size()
        if self.indexer_keys is not None:
            b += self.indexer_keys.numel() * self.indexer_keys.element_size()
        return b

    def to_cpu(self):
        if self.on_gpu:
            self.data = self.data.to("cpu", non_blocking=True)
            if self.indexer_keys is not None:
                self.indexer_keys = self.indexer_keys.to("cpu", non_blocking=True)
            self.on_gpu = False

    def to_gpu(self, device: torch.device):
        if not self.on_gpu:
            self.data = self.data.to(device, non_blocking=True)
            if self.indexer_keys is not None:
                self.indexer_keys = self.indexer_keys.to(device, non_blocking=True)
            self.on_gpu = True


class PagedCompressedCache:
    """Paged compressed KV cache for CSA or HCA layers.

    Manages blocks of compressed entries. Supports eviction and offloading.
    """

    def __init__(self, head_dim: int, indexer_dim: int = 0,
                 layer_type: str = "csa", max_gpu_blocks: int = 2048,
                 device: str = "cuda"):
        self.head_dim = head_dim
        self.indexer_dim = indexer_dim
        self.layer_type = layer_type
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_gpu_blocks = max_gpu_blocks

        self.entries_per_block = _CSA_PER_BLOCK if layer_type == "csa" else _HCA_PER_BLOCK

        self._blocks: Dict[int, PagedCompressedBlock] = {}
        self._block_order: list[int] = []  # LRU order
        self._next_block_id = 0
        self._gpu_block_count = 0
        self._offloaded_block_ids: set[int] = set()

        # Partial block accumulation
        self._partial_data: list[torch.Tensor] = []
        self._partial_idx_keys: list[torch.Tensor] = []
        self._partial_count = 0

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def num_entries(self) -> int:
        return sum(b.num_entries for b in self._blocks.values()) + self._partial_count

    def append(self, entry: torch.Tensor, idx_key: torch.Tensor | None = None):
        """Append one compressed entry. Automatically allocates new blocks."""
        self._partial_data.append(entry.detach().to(self.device))
        if idx_key is not None and self.indexer_dim > 0:
            self._partial_idx_keys.append(idx_key.detach().to(self.device))
        self._partial_count += 1

        if self._partial_count >= self.entries_per_block:
            self._flush_partial()

    def _flush_partial(self):
        if self._partial_count == 0:
            return
        block_id = self._next_block_id
        self._next_block_id += 1

        data = torch.cat(self._partial_data, dim=0)  # [num, head_dim]
        idx_keys = None
        if self._partial_idx_keys:
            idx_keys = torch.cat(self._partial_idx_keys, dim=0)

        block = PagedCompressedBlock(
            block_id, data, idx_keys, self._partial_count, self.layer_type)
        self._add_block(block_id, block)

        self._partial_data.clear()
        self._partial_idx_keys.clear()
        self._partial_count = 0

    def _add_block(self, block_id: int, block: PagedCompressedBlock):
        self._ensure_gpu_capacity()
        self._blocks[block_id] = block
        self._block_order.append(block_id)
        self._gpu_block_count += 1

    def _ensure_gpu_capacity(self):
        while self._gpu_block_count >= self.max_gpu_blocks and self._block_order:
            # Evict LRU block to CPU
            evict_id = self._block_order.pop(0)
            block = self._blocks.get(evict_id)
            if block and block.on_gpu:
                block.to_cpu()
                self._gpu_block_count -= 1
                self._offloaded_block_ids.add(evict_id)

    def access_block(self, block_id: int):
        """Mark block as recently accessed. Bring back from CPU if needed."""
        block = self._blocks.get(block_id)
        if block is None:
            return
        block.last_access_step = int(time.time())

        if not block.on_gpu:
            self._ensure_gpu_capacity()
            block.to_gpu(self.device)
            self._gpu_block_count += 1
            self._offloaded_block_ids.discard(block_id)

        # Move to end of LRU
        if block_id in self._block_order:
            self._block_order.remove(block_id)
        self._block_order.append(block_id)

    def get_all_entries(self) -> torch.Tensor | None:
        """Get all compressed entries (for HCA attention)."""
        self._flush_partial()
        if not self._blocks:
            return None
        sorted_blocks = sorted(self._blocks.values(), key=lambda b: b.block_id)
        tensors = [b.data for b in sorted_blocks]
        return torch.cat(tensors, dim=0)  # [total_num, head_dim]

    def get_block(self, block_id: int) -> PagedCompressedBlock | None:
        return self._blocks.get(block_id)

    def get_visible_blocks(self, up_to_entry: int) -> torch.Tensor:
        """Get compressed entries up to the given entry index (for attention)."""
        self._flush_partial()
        sorted_blocks = sorted(self._blocks.values(), key=lambda b: b.block_id)
        tensors = []
        count = 0
        for b in sorted_blocks:
            if count + b.num_entries > up_to_entry:
                take = up_to_entry - count
                if take > 0:
                    tensors.append(b.data[:take])
                break
            tensors.append(b.data)
            count += b.num_entries
            if count >= up_to_entry:
                break
        if not tensors:
            return torch.zeros(0, self.head_dim, device=self.device, dtype=torch.bfloat16)
        return torch.cat(tensors, dim=0)

    def get_all_indexer_keys(self) -> torch.Tensor | None:
        """Get all indexer keys (for Lightning Indexer scoring)."""
        if self.indexer_dim == 0:
            return None
        self._flush_partial()
        sorted_blocks = sorted(self._blocks.values(), key=lambda b: b.block_id)
        tensors = [b.indexer_keys for b in sorted_blocks if b.indexer_keys is not None]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    def reset(self):
        self._blocks.clear()
        self._block_order.clear()
        self._offloaded_block_ids.clear()
        self._gpu_block_count = 0
        self._next_block_id = 0
        self._partial_data.clear()
        self._partial_idx_keys.clear()
        self._partial_count = 0


class SWACache:
    """Sliding window KV cache for the last `window_size` tokens."""

    def __init__(self, window_size: int = 128, head_dim: int = 512,
                 device: str = "cuda"):
        self.window_size = window_size
        self.head_dim = head_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        self.k_cache = torch.zeros(1, window_size, head_dim, device=self.device, dtype=torch.bfloat16)
        self.v_cache = torch.zeros(1, window_size, head_dim, device=self.device, dtype=torch.bfloat16)
        self._write_pos = 0
        self._num_entries = 0

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append a single KV entry. k, v: [1, head_dim]."""
        pos = self._write_pos % self.window_size
        self.k_cache[0, pos] = k.squeeze(0).to(torch.bfloat16)
        self.v_cache[0, pos] = v.squeeze(0).to(torch.bfloat16)
        self._write_pos = (pos + 1) % self.window_size
        self._num_entries = min(self._num_entries + 1, self.window_size)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k, v) as [1, num_entries, head_dim]."""
        n = self._num_entries
        if self._num_entries == self.window_size and self._write_pos != 0:
            # Wrapped: reorder to chronological order
            k = torch.cat([self.k_cache[:, self._write_pos:], self.k_cache[:, :self._write_pos]], dim=1)
            v = torch.cat([self.v_cache[:, self._write_pos:], self.v_cache[:, :self._write_pos]], dim=1)
        else:
            k = self.k_cache[:, :n]
            v = self.v_cache[:, :n]
        return k, v

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._write_pos = 0
        self._num_entries = 0


class HybridKVCache:
    """Per-layer KV cache: SWA + optional compressed (CSA or HCA).

    Replaces the in-engine LayerState + CompressedKVCache with
    block-aligned paged management and optional disk offloading.
    """

    def __init__(self, compress_ratio: int, head_dim: int,
                 indexer_dim: int = 0, device: str = "cuda",
                 disk_offload_dir: str | None = None):
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.disk_offload_dir = disk_offload_dir

        self.swa = SWACache(window_size=128, head_dim=head_dim, device=device)

        self.compressed: PagedCompressedCache | None = None
        if compress_ratio > 0:
            layer_type = "csa" if compress_ratio == 4 else "hca"
            self.compressed = PagedCompressedCache(
                head_dim=head_dim, indexer_dim=indexer_dim,
                layer_type=layer_type, device=device)

        self._start_pos = 0

    @property
    def is_swa(self) -> bool:
        return self.compress_ratio == 0

    @property
    def is_csa(self) -> bool:
        return self.compress_ratio == 4

    @property
    def is_hca(self) -> bool:
        return self.compress_ratio > 0 and self.compress_ratio != 4

    def append_swa(self, k: torch.Tensor, v: torch.Tensor):
        self.swa.append(k, v)

    def append_compressed(self, entry: torch.Tensor,
                          idx_key: torch.Tensor | None = None):
        if self.compressed is not None:
            self.compressed.append(entry, idx_key)

    def append_compressed_batch(self, entries: torch.Tensor,
                                idx_keys: torch.Tensor | None = None):
        """Append multiple compressed entries at once (prefill)."""
        if self.compressed is not None:
            B, num, D = entries.shape
            for i in range(num):
                e = entries[:, i:i + 1, :].squeeze(0)
                ik = idx_keys[:, i:i + 1, :].squeeze(0) if idx_keys is not None else None
                self.compressed.append(e, ik)

    def get_swa_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.swa.get()

    def get_compressed_kv(self) -> torch.Tensor | None:
        if self.compressed is None:
            return None
        return self.compressed.get_all_entries()

    def get_indexer_keys(self) -> torch.Tensor | None:
        if self.compressed is None:
            return None
        return self.compressed.get_all_indexer_keys()

    def reset(self):
        self.swa.reset()
        if self.compressed is not None:
            self.compressed.reset()
        self._start_pos = 0

    def memory_stats(self) -> dict:
        stats = {"swa_bytes": 0, "compressed_bytes": 0, "gpu_blocks": 0}
        if hasattr(self.swa, 'k_cache'):
            stats["swa_bytes"] = (self.swa.k_cache.numel() + self.swa.v_cache.numel()) * 2
        if self.compressed is not None:
            stats["compressed_bytes"] = sum(b.memory_bytes for b in self.compressed._blocks.values())
            stats["gpu_blocks"] = self.compressed._gpu_block_count
        return stats
