"""Lightning Indexer (DeepSeek-V4 paper Section 3.2, formulas 14-17).

Selects top-k compressed KV positions for sparse attention via learned scoring.
Uses its own Compressor to build compressed indexer KV, then computes
per-head ReLU scores weighted by head-importance weights.

Key formulas:
  c_t^Q = h_t W^{DQ}                                     (14)
  q_t^I = c_t^Q W^{IUQ}                                   (15)
  w_{t,h}^I = h_t W^{IW}                                  (16)
  I_{t,s} = sum_h w_{t,h}^I * ReLU(q_{t,h}^I · K_s^Comp) (17)
"""

from __future__ import annotations

import torch

from home_seek.compressor import Compressor


class LightningIndexer:
    """Lightning Indexer for CSA layers.

    Builds compressed KV for scoring via its own Compressor, then selects
    top-k entries based on per-head weighted ReLU scores.
    """

    def __init__(
        self,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        compress_ratio: int,
        q_lora_rank: int,
        device: str = "cuda",
    ):
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.device = torch.device(device) if isinstance(device, str) else device
        self.softmax_scale = index_head_dim ** -0.5

        # Weights — set via set_weights()
        self.wq_b: torch.Tensor | None = None       # [n_heads * head_dim, q_lora]
        self.weights_proj: torch.Tensor | None = None  # [n_heads, D]
        self.compressor: Compressor | None = None

        # Compressed KV cache for scoring
        self.kv_cache: torch.Tensor | None = None  # [1, max_blocks, index_head_dim]

        # State tracking
        self.num_compressed = 0
        self.max_blocks = 0

    def set_weights(
        self,
        wq_b: torch.Tensor | None,
        weights_proj: torch.Tensor | None,
        compressor_wkv: torch.Tensor | None,
        compressor_wgate: torch.Tensor | None,
        compressor_norm: torch.Tensor | None,
        compressor_ape: torch.Tensor | None,
        max_blocks: int = 8192,
    ):
        """Set indexer weights and initialize compressor."""
        self.wq_b = wq_b
        self.weights_proj = weights_proj

        if self.compressor is None and compressor_wkv is not None:
            self.compressor = Compressor(
                ratio=self.compress_ratio,
                head_dim=self.index_head_dim,
                coff=2,  # indexer always uses CSA-style overlap
                ape=compressor_ape,
                wkv=compressor_wkv,
                wgate=compressor_wgate,
                norm_w=compressor_norm,
                device=str(self.device),
            )

        if self.kv_cache is None:
            self.max_blocks = max_blocks
            self.kv_cache = torch.zeros(1, max_blocks, self.index_head_dim,
                                        device=self.device, dtype=torch.bfloat16)

    def reset(self):
        self.num_compressed = 0
        if self.compressor is not None:
            self.compressor.reset()
        if self.kv_cache is not None:
            self.kv_cache.zero_()

    def update_compressed_kv(self, x: torch.Tensor, start_pos: int = 0):
        """Run compressor and update kv_cache.

        Parameters
        ----------
        x : torch.Tensor, [B, T, D]
            Hidden states.
        start_pos : int
            Position offset.
        """
        if self.compressor is None:
            return

        compressed = self.compressor.compress(x, start_pos)
        if compressed is None:
            return

        num_new = compressed.shape[1]  # number of new compressed entries
        if start_pos == 0:
            # Prefill: store at start
            self.kv_cache[:, :num_new] = compressed.to(torch.bfloat16)
            self.num_compressed = num_new
        else:
            # Decode: append one entry at a time
            block_idx = start_pos // self.compress_ratio
            if block_idx < self.max_blocks:
                self.kv_cache[:, block_idx:block_idx + num_new] = compressed.to(torch.bfloat16)
                self.num_compressed = max(self.num_compressed, block_idx + num_new)

    def compute_indexer(self, x: torch.Tensor, q_latent: torch.Tensor,
                        start_pos: int = 0, offset: int = 0) -> torch.Tensor:
        """Compute top-k indices for sparse attention.

        Parameters
        ----------
        x : torch.Tensor, [B, T, D]
            Hidden states.
        q_latent : torch.Tensor, [B, T, q_lora_rank]
            Low-rank query latent.
        start_pos : int
            Position offset.
        offset : int
            Offset to add to selected indices (window KV count).

        Returns
        -------
        torch.Tensor, [B, T, index_topk]
            Selected compressed KV positions (indices into full KV cache).
            Values of -1 indicate invalid positions.
        """
        if self.wq_b is None or self.weights_proj is None or self.kv_cache is None:
            return None

        B, T, _ = x.shape
        end_pos = start_pos + T
        num_visible = end_pos // self.compress_ratio

        if num_visible == 0:
            return None

        # Update compressor (new KV to index over)
        self.update_compressed_kv(x, start_pos)

        # Query projection: q_latent → [n_heads, head_dim]
        q = torch.matmul(q_latent.to(self.wq_b.dtype), self.wq_b.t())  # [B, T, n_heads * head_dim]
        q = q.view(B, T, self.index_n_heads, self.index_head_dim)       # [B, T, n_heads, head_dim]

        # Head importance weights
        weights = torch.matmul(x.float(), self.weights_proj.float().t())  # [B, T, n_heads]
        weights = weights * (self.softmax_scale * (self.index_n_heads ** -0.5))

        # Scoring: I_{t,s} = sum_h w_h * ReLU(q_h · K_s)
        # kv_cache: [1, num_visible, head_dim]
        kv_vis = self.kv_cache[:, :num_visible]  # [1, num_visible, head_dim]

        # einsum: "bshd,btd->bsht" → [B, T, n_heads, num_visible]
        index_score = torch.einsum("bthd,bnd->bthn", q.float(), kv_vis.float())
        index_score = index_score.relu_()                      # ReLU
        index_score = (index_score * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, num_visible]

        # Apply causal mask for prefill
        if start_pos == 0:
            # Each position t can only see compressed blocks up to t // ratio
            causal_mask = torch.arange(num_visible, device=self.device).unsqueeze(0) >= \
                          (torch.arange(1, T + 1, device=self.device).unsqueeze(1) // self.compress_ratio)
            index_score = index_score + torch.where(causal_mask, float("-inf"), 0.0)

        k = min(self.index_topk, num_visible)
        topk_idxs = index_score.topk(k, dim=-1)[1]  # [B, T, k]

        if start_pos == 0:
            # Mask invalid positions (where causal mask would have -inf)
            idx_range = torch.arange(1, T + 1, device=self.device).unsqueeze(1)
            causal_mask_idx = topk_idxs >= idx_range // self.compress_ratio
            topk_idxs = torch.where(causal_mask_idx, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset

        return topk_idxs.int()
