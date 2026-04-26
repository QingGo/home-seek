"""CSA and HCA KV compressors following DeepSeek-V4 paper formulas (9)-(12).

CSA (compress_ratio=4, overlap=True, coff=2):
  Dual-stream: wkv/wgate produce coff*d = 1024 dims.
  First d=512 dims = stream A (C^a / Z^a), second d=512 dims = stream B (C^b / Z^b).
  Blocks of ratio=4 tokens, overlapping: current block stream B + previous block stream A.
  Softmax-weighted fusion over 2*ratio=8 entries → compressed to d dims.

HCA (compress_ratio=128, overlap=False, coff=1):
  Single-stream: wkv/wgate produce d = 512 dims.
  Blocks of ratio=128 tokens, no overlap.
  Softmax-weighted fusion over ratio=128 entries → compressed to d dims.
"""

from __future__ import annotations

import torch

from home_seek.utils import rms_norm


class Compressor:
    """KV compressor with learned gated pooling over `ratio` consecutive tokens.

    Parameters
    ----------
    ratio : int
        Number of tokens per compression block (4 for CSA, 128 for HCA).
    head_dim : int
        Dimension of each KV head (512 for the main compressor, 128 for indexer compressor).
    coff : int
        2 for CSA (dual-stream with overlap), 1 for HCA (single-stream).
    ape : torch.Tensor
        [ratio, coff * head_dim] learnable absolute position encoding.
    wkv : torch.Tensor
        [coff * head_dim, D] linear projection for KV.
    wgate : torch.Tensor
        [coff * head_dim, D] linear projection for compression scores (Z).
    norm_w : torch.Tensor
        [head_dim] RMSNorm weight.
    device : str
        Device for state buffers.
    """

    def __init__(
        self,
        ratio: int,
        head_dim: int,
        coff: int,
        ape: torch.Tensor | None,
        wkv: torch.Tensor | None,
        wgate: torch.Tensor | None,
        norm_w: torch.Tensor | None,
        device: str = "cuda",
    ):
        self.ratio = ratio
        self.head_dim = head_dim
        self.coff = coff
        self.overlap = coff == 2  # CSA when coff=2
        self.device = torch.device(device) if isinstance(device, str) else device

        self.ape = ape
        self.wkv = wkv
        self.wgate = wgate
        self.norm_w = norm_w

        # State buffers for incremental compression during decode
        # kv_state: [coff * ratio, coff * head_dim] — accumulates partial block
        self.kv_state = torch.zeros(coff * ratio, coff * head_dim, device=self.device, dtype=torch.float32)
        # score_state: same shape, initialized to -inf
        self.score_state = torch.full((coff * ratio, coff * head_dim), float("-inf"),
                                      device=self.device, dtype=torch.float32)
        # Number of tokens accumulated in current partial block
        self.accumulated = 0

    def reset(self):
        self.kv_state.zero_()
        self.score_state.fill_(float("-inf"))
        self.accumulated = 0

    def overlap_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform blocks for overlapping CSA compression.

        Converts [B, num_blocks, ratio, 2*d] → [B, num_blocks, 2*ratio, d]
        where:
          positions ratio:2*ratio = current block's stream B (second half)
          positions 0:ratio (for i>0) = previous block's stream A (first half)
        First block gets zeros / -inf for the overlap part.
        """
        B, num_blocks, _, dim2d = tensor.shape
        d = self.head_dim
        # Reshape ratio dim into 2 streams
        # tensor: [B, num_blocks, ratio, 2*d] → [B, num_blocks, ratio, 2, d]
        t = tensor.view(B, num_blocks, self.ratio, 2, d)
        new_t = tensor.new_full((B, num_blocks, 2 * self.ratio, d), 0.0)
        # Positions ratio:2*ratio ← current block stream B (t[:, :, :, 1, :])
        new_t[:, :, self.ratio:] = t[:, :, :, 1, :]
        # Positions 0:ratio ← previous block stream A (t[:, 0:-1, :, 0, :])
        if num_blocks > 1:
            new_t[:, 1:, :self.ratio] = t[:, :-1, :, 0, :]
        return new_t

    def compress_prefill(self, x: torch.Tensor) -> torch.Tensor | None:
        """Prefill-phase compression: compress all tokens at once.

        Parameters
        ----------
        x : torch.Tensor, [B, T, D]
            Hidden states for all prompt tokens.

        Returns
        -------
        torch.Tensor | None
            [B, num_blocks, head_dim] compressed KV, or None if T < ratio.
        """
        if self.wkv is None:
            return None

        B, T, D = x.shape
        if T < self.ratio:
            return None

        x_f32 = x.float()
        # Project to coff * head_dim (compute in float32 as in reference)
        wkv_f32 = self.wkv.to(torch.float32) if self.wkv.dtype != torch.float32 else self.wkv
        wgate_f32 = self.wgate.to(torch.float32) if self.wgate.dtype != torch.float32 else self.wgate
        kv = torch.matmul(x_f32, wkv_f32.t())       # [B, T, coff*d]
        score = torch.matmul(x_f32, wgate_f32.t())   # [B, T, coff*d]

        # Add positional bias
        num_blocks = T // self.ratio
        remainder = T % self.ratio
        cutoff = T - remainder

        if remainder > 0:
            self.kv_state[:remainder] = kv[:, cutoff:]
            self.score_state[:remainder] = score[:, cutoff:] + (
                self.ape[:remainder] if self.ape is not None else 0)
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]
            self.accumulated = remainder
        else:
            self.accumulated = 0

        num_blocks = cutoff // self.ratio

        # Reshape into blocks: [B, num_blocks, ratio, coff*d]
        kv = kv.view(B, num_blocks, self.ratio, -1)
        if self.ape is not None:
            score = score.view(B, num_blocks, self.ratio, -1) + self.ape[:self.ratio].unsqueeze(0).unsqueeze(0)
        else:
            score = score.view(B, num_blocks, self.ratio, -1)

        if self.overlap:
            # CSA: overlap transform
            kv = self.overlap_transform(kv)       # [B, num_blocks, 2*ratio, d]
            score = self.overlap_transform(score)  # [B, num_blocks, 2*ratio, d]
            # Softmax over the 2*ratio entries
            kv = (kv * score.softmax(dim=2)).sum(dim=2)  # [B, num_blocks, d]
        else:
            # HCA: simple softmax over ratio entries
            kv = (kv * score.softmax(dim=2)).sum(dim=2)  # [B, num_blocks, d]

        if self.norm_w is not None:
            kv = rms_norm(kv.to(torch.bfloat16), self.norm_w.to(torch.bfloat16))

        return kv  # [B, num_blocks, d]

    def compress_decode(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        """Decode-phase incremental compression.

        Parameters
        ----------
        x : torch.Tensor, [B, 1, D]
            Hidden state for a single decode token.
        start_pos : int
            Position of this token in the full sequence.

        Returns
        -------
        torch.Tensor | None
            [B, 1, head_dim] compressed entry (when block completes), or None.
        """
        if self.wkv is None:
            return None

        x_f32 = x.float()
        wkv_f32 = self.wkv.to(torch.float32) if self.wkv.dtype != torch.float32 else self.wkv
        wgate_f32 = self.wgate.to(torch.float32) if self.wgate.dtype != torch.float32 else self.wgate
        kv = torch.matmul(x_f32, wkv_f32.t())       # [B, 1, coff*d]
        score = torch.matmul(x_f32, wgate_f32.t())   # [B, 1, coff*d]

        pos_in_block = start_pos % self.ratio
        if self.ape is not None:
            score = score + self.ape[pos_in_block:pos_in_block + 1]

        self.kv_state[self.accumulated] = kv.squeeze(0).squeeze(0)
        self.score_state[self.accumulated] = score.squeeze(0).squeeze(0)
        self.accumulated += 1

        should_compress = self.accumulated == self.ratio
        if not should_compress:
            return None

        if self.overlap:
            # Save current state for next block's overlap
            kv_a = self.kv_state[:self.ratio, :self.head_dim].clone()    # current stream A
            # Current block: use accumulated state
            cur_b = self.kv_state[self.ratio:self.ratio * 2, self.head_dim:]  # [ratio, d] stream B
            cur_b_score = self.score_state[self.ratio:self.ratio * 2, self.head_dim:]
            combined_kv = torch.cat([kv_a, cur_b], dim=0)  # [2*ratio, d]
            combined_score = torch.cat([
                self.score_state[:self.ratio, :self.head_dim],
                cur_b_score
            ], dim=0)
            compressed = (combined_kv * combined_score.softmax(dim=0)).sum(dim=0, keepdim=True)
            # Shift: stream B becomes stream A for next block
            self.kv_state[:self.ratio] = self.kv_state[self.ratio:]
            self.score_state[:self.ratio] = self.score_state[self.ratio:]
        else:
            compressed = (self.kv_state[:self.ratio] *
                          self.score_state[:self.ratio].softmax(dim=0)).sum(dim=0, keepdim=True)

        # Reset accumulation
        self.accumulated = 0

        if self.norm_w is not None:
            compressed = rms_norm(compressed.to(torch.bfloat16), self.norm_w.to(torch.bfloat16))

        return compressed.unsqueeze(0)  # [B, 1, d]

    def compress(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor | None:
        """Unified compress interface.

        Parameters
        ----------
        x : torch.Tensor, [B, T, D]
            Hidden states.
        start_pos : int
            Position offset (0 for prefill, >0 for decode).

        Returns
        -------
        torch.Tensor | None
            Compressed KV entries.
        """
        if start_pos == 0:
            return self.compress_prefill(x)
        return self.compress_decode(x, start_pos)
