import torch
from collections import OrderedDict


class KVCacheManager:
    def __init__(self, config, max_cpu_gb: float = 32.0, device="cuda"):
        self.config = config
        self.device = torch.device(device)
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.swa_window = config.sliding_window
        self.csa_ratio = 4
        self.hca_ratio = 128

        self.max_cpu_bytes = int(max_cpu_gb * (1024**3))

        self.layer_caches = {}
        self.compressed_caches = {}
        self.offloaded_pages = OrderedDict()
        self.total_offloaded_bytes = 0

    def init_cache(self, batch_size: int = 1):
        for layer_idx in range(self.num_layers):
            attn_type = self._get_attention_type(layer_idx)
            if attn_type == "swa":
                self.layer_caches[layer_idx] = SWACache(
                    batch_size, self.num_kv_heads, self.head_dim, self.swa_window, self.device
                )
                self.compressed_caches[layer_idx] = None
            else:
                self.layer_caches[layer_idx] = SWACache(
                    batch_size, self.num_kv_heads, self.head_dim, self.swa_window, self.device
                )
                self.compressed_caches[layer_idx] = CompressedCache(
                    batch_size, self.num_kv_heads, self.head_dim, self.csa_ratio, self.hca_ratio, self.device
                )

    def _get_attention_type(self, layer_idx: int) -> str:
        if layer_idx < 2:
            return "swa"
        return "hca"

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        attn_type = self._get_attention_type(layer_idx)
        self.layer_caches[layer_idx].append(k, v)

        if attn_type != "swa" and self.compressed_caches[layer_idx] is not None:
            self.compressed_caches[layer_idx].update(k, v)

    def get_swa_cache(self, layer_idx: int):
        return self.layer_caches[layer_idx].get_cache()

    def get_compressed_cache(self, layer_idx: int):
        if self.compressed_caches[layer_idx] is not None:
            return self.compressed_caches[layer_idx].get_caches()
        return None, None, None, None

    def offload_cold_pages(self, target_free_bytes: int = None):
        total = torch.cuda.memory_allocated(self.device)
        if target_free_bytes is None:
            target_free_bytes = int(total * 0.15)

        needed = total - (torch.cuda.memory_reserved(self.device) - target_free_bytes)
        if needed <= 0:
            return 0

        freed = 0
        keys_to_offload = list(self.offloaded_pages.keys())
        for key in keys_to_offload:
            if freed >= needed:
                break
            page = self.offloaded_pages.pop(key, None)
            if page is None:
                continue
            layer_idx, pos = key
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                continue
            k_page, v_page = page
            self.total_offloaded_bytes += (k_page.numel() + v_page.numel()) * k_page.element_size()
            freed += (k_page.numel() + v_page.numel()) * k_page.element_size()

        torch.cuda.empty_cache()
        return freed

    def get_memory_stats(self):
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        return {
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "offloaded_gb": self.total_offloaded_bytes / (1024**3),
        }


class SWACache:
    def __init__(self, batch_size: int, num_kv_heads: int, head_dim: int, window_size: int, device: str):
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.device = device

        self.k_cache = torch.zeros(batch_size, num_kv_heads, 0, head_dim, device=device, dtype=torch.bfloat16)
        self.v_cache = torch.zeros(batch_size, num_kv_heads, 0, head_dim, device=device, dtype=torch.bfloat16)

    def append(self, k: torch.Tensor, v: torch.Tensor):
        self.k_cache = torch.cat([self.k_cache, k], dim=-2)
        self.v_cache = torch.cat([self.v_cache, v], dim=-2)

        if self.k_cache.shape[-2] > self.window_size:
            self.k_cache = self.k_cache[:, :, -self.window_size:, :]
            self.v_cache = self.v_cache[:, :, -self.window_size:, :]

    def get_cache(self):
        return self.k_cache, self.v_cache

    def get_k(self):
        return self.k_cache

    def get_v(self):
        return self.v_cache


class CompressedCache:
    def __init__(self, batch_size: int, num_kv_heads: int, head_dim: int, csa_ratio: int, hca_ratio: int, device: str):
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.csa_ratio = csa_ratio
        self.hca_ratio = hca_ratio
        self.device = device

        self.csa_k = torch.zeros(batch_size, num_kv_heads, 0, head_dim, device=device, dtype=torch.bfloat16)
        self.csa_v = torch.zeros(batch_size, num_kv_heads, 0, head_dim, device=device, dtype=torch.bfloat16)
        self.hca_k = torch.zeros(batch_size, num_kv_heads, 0, head_dim, device=device, dtype=torch.bfloat16)
        self.hca_v = torch.zeros(batch_size, num_kv_heads, 0, head_dim, device=device, dtype=torch.bfloat16)

    def update(self, k: torch.Tensor, v: torch.Tensor):
        new_len = k.shape[-2]

        remainder_csa = new_len % self.csa_ratio
        if remainder_csa > 0:
            k_pad = k[:, :, :-remainder_csa, :]
            v_pad = v[:, :, :-remainder_csa, :]
        else:
            k_pad = k
            v_pad = v

        if k_pad.shape[-2] > 0:
            k_csa = k_pad[:, :, :k_pad.shape[-2] // self.csa_ratio * self.csa_ratio, :]
            v_csa = v_pad[:, :, :v_pad.shape[-2] // self.csa_ratio * self.csa_ratio, :]
            k_csa_compressed = k_csa.view(self.batch_size, self.num_kv_heads, -1, self.csa_ratio, self.head_dim).mean(dim=3)
            v_csa_compressed = v_csa.view(self.batch_size, self.num_kv_heads, -1, self.csa_ratio, self.head_dim).mean(dim=3)
            self.csa_k = torch.cat([self.csa_k, k_csa_compressed], dim=-2)
            self.csa_v = torch.cat([self.csa_v, v_csa_compressed], dim=-2)

        remainder_hca = new_len % self.hca_ratio
        if remainder_hca > 0:
            k_pad_hca = k[:, :, :-remainder_hca, :]
            v_pad_hca = v[:, :, :-remainder_hca, :]
        else:
            k_pad_hca = k
            v_pad_hca = v

        if k_pad_hca.shape[-2] > 0:
            k_hca = k_pad_hca[:, :, :k_pad_hca.shape[-2] // self.hca_ratio * self.hca_ratio, :]
            v_hca = v_pad_hca[:, :, :v_pad_hca.shape[-2] // self.hca_ratio * self.hca_ratio, :]
            k_hca_compressed = k_hca.view(self.batch_size, self.num_kv_heads, -1, self.hca_ratio, self.head_dim).mean(dim=3)
            v_hca_compressed = v_hca.view(self.batch_size, self.num_kv_heads, -1, self.hca_ratio, self.head_dim).mean(dim=3)
            self.hca_k = torch.cat([self.hca_k, k_hca_compressed], dim=-2)
            self.hca_v = torch.cat([self.hca_v, v_hca_compressed], dim=-2)

    def get_caches(self):
        return self.csa_k, self.csa_v, self.hca_k, self.hca_v
