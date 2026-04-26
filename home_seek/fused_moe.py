import torch
import triton
import triton.language as tl
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────
# FP4 → BF16 dequantization helper (one kernel launch for all 3 weight mats)
# ──────────────────────────────────────────────────────────────────────

def _dequantize_fp4_to_bf16(
    w1_packed: torch.Tensor,
    w1_scale: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantize all three FP4-packed weight matrices to BF16 in one step."""
    from tile_reference import unpack_from_e2m1fn_x2

    def _deq_one(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        deq = unpack_from_e2m1fn_x2(packed)
        block_size = 32
        sf = scale.repeat_interleave(block_size, dim=1) if scale.dim() == 2 else scale
        return (deq.float() * sf.float()).to(torch.bfloat16)

    w1 = _deq_one(w1_packed, w1_scale)
    w3 = _deq_one(w3_packed, w3_scale)
    w2 = _deq_one(w2_packed, w2_scale)
    return w1, w3, w2


# ──────────────────────────────────────────────────────────────────────
# Triton fused kernels (BF16 weights)
# ──────────────────────────────────────────────────────────────────────

@triton.jit
def _triton_fused_gate_up_kernel(
    hidden_ptr, w1_ptr, w3_ptr,
    gate_out_ptr, up_out_ptr,
    M, I, D, swiglu_limit,
    stride_hid_m, stride_hid_k,
    stride_w1_n, stride_w1_k,
    stride_w3_n, stride_w3_k,
    stride_gat_m, stride_gat_n,
    stride_up_m, stride_up_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    gate_acc = tl.zeros([BM, BN], dtype=tl.float32)
    up_acc = tl.zeros([BM, BN], dtype=tl.float32)

    for k in range(0, D, BK):
        mask_k = (k + offs_k) < D
        mask_m = offs_m < M

        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hid_m + (k + offs_k)[None, :] * stride_hid_k
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w1_ptrs = w1_ptr + offs_n[None, :] * stride_w1_n + (k + offs_k)[:, None] * stride_w1_k
        w1_tile = tl.load(w1_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < I), other=0.0)

        w3_ptrs = w3_ptr + offs_n[None, :] * stride_w3_n + (k + offs_k)[:, None] * stride_w3_k
        w3_tile = tl.load(w3_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < I), other=0.0)

        gate_acc += tl.dot(h, w1_tile)
        up_acc += tl.dot(h, w3_tile)

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < I
    gate_ptrs = gate_out_ptr + offs_m[:, None] * stride_gat_m + offs_n[None, :] * stride_gat_n
    up_ptrs = up_out_ptr + offs_m[:, None] * stride_up_m + offs_n[None, :] * stride_up_n
    _out_dtype = gate_out_ptr.dtype.element_ty
    tl.store(gate_ptrs, gate_acc.to(_out_dtype), mask=mask_m & mask_n)
    tl.store(up_ptrs, up_acc.to(_out_dtype), mask=mask_m & mask_n)


@triton.jit
def _triton_fused_down_kernel(
    gate_ptr, up_ptr, w2_ptr,
    out_ptr,
    M, D, I, swiglu_limit,
    stride_gat_m, stride_gat_n,
    stride_up_m, stride_up_n,
    stride_w2_n, stride_w2_k,
    stride_out_m, stride_out_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    out_acc = tl.zeros([BM, BN], dtype=tl.float32)

    for k in range(0, I, BK):
        mask_k = (k + offs_k) < I
        mask_m = offs_m < M

        gate_ptrs = gate_ptr + offs_m[:, None] * stride_gat_m + (k + offs_k)[None, :] * stride_gat_n
        gate_tile = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        up_ptrs = up_ptr + offs_m[:, None] * stride_up_m + (k + offs_k)[None, :] * stride_up_n
        up_tile = tl.load(up_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        gate_f32 = gate_tile.to(tl.float32)
        up_f32 = up_tile.to(tl.float32)
        gate_f32 = tl.minimum(gate_f32, swiglu_limit)
        up_f32 = tl.minimum(tl.maximum(up_f32, -swiglu_limit), swiglu_limit)
        activated = gate_f32 * tl.sigmoid(gate_f32.to(tl.float32)) * up_f32

        w2_ptrs = w2_ptr + offs_n[None, :] * stride_w2_n + (k + offs_k)[:, None] * stride_w2_k
        w2_tile = tl.load(w2_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < D), other=0.0)

        out_acc += tl.dot(activated.to(w2_tile.dtype), w2_tile)

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    _out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptrs, out_acc.to(_out_dtype), mask=(offs_m[:, None] < M) & (offs_n[None, :] < D))


# ──────────────────────────────────────────────────────────────────────
# Tuning and format detection
# ──────────────────────────────────────────────────────────────────────

def _tune_blocks(device) -> tuple:
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count

    if sm_count >= 100:
        return 16, 32, 64
    elif sm_count >= 70:
        return 16, 32, 32
    else:
        return 16, 16, 32


def _is_fp4_packed(data: torch.Tensor) -> bool:
    return data is not None and data.dtype == torch.int8


# ──────────────────────────────────────────────────────────────────────
# Per-inference dequantization cache (avoids repeated FP4→BF16 for
# experts reused during token-by-token decode)
# ──────────────────────────────────────────────────────────────────────

_deq_cache: dict = {}          # packed_ptr → (w1_bf16, w3_bf16, w2_bf16)
_deq_cache_order: list = []    # LRU order of keys
_DEQ_CACHE_SIZE = 64
_deq_cache_hits: int = 0
_deq_cache_misses: int = 0


def clear_deq_cache():
    """Clear the dequantization cache. Call at start of each generate()."""
    global _deq_cache, _deq_cache_order, _deq_cache_hits, _deq_cache_misses
    _deq_cache.clear()
    _deq_cache_order.clear()
    _deq_cache_hits = 0
    _deq_cache_misses = 0


def deq_cache_stats() -> tuple[int, int]:
    """Return (hits, misses) since last clear."""
    return _deq_cache_hits, _deq_cache_misses


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def fused_expert_ffn_triton(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    swiglu_limit: float = 10.0,
    w1_scale: Optional[torch.Tensor] = None,
    w3_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hidden.device.type != "cuda":
        return fused_expert_ffn_pt(hidden, w1, w3, w2, swiglu_limit)

    M = hidden.shape[0]
    H = hidden.shape[1]

    # Dequantize FP4 to BF16 first, then use unified BF16 kernels
    if _is_fp4_packed(w1):
        cache_key = w1.data_ptr()
        cached = _deq_cache.get(cache_key)
        if cached is not None:
            global _deq_cache_hits
            _deq_cache_hits += 1
            w1, w3, w2 = cached
        else:
            global _deq_cache_misses
            _deq_cache_misses += 1
            w1, w3, w2 = _dequantize_fp4_to_bf16(
                w1, w1_scale, w3, w3_scale, w2, w2_scale)
            _deq_cache[cache_key] = (w1, w3, w2)
            _deq_cache_order.append(cache_key)
            while len(_deq_cache_order) > _DEQ_CACHE_SIZE:
                old_key = _deq_cache_order.pop(0)
                _deq_cache.pop(old_key, None)

    D = w1.shape[1]
    I = w1.shape[0]

    assert H == D, f"hidden dim mismatch: hidden={H} vs weight D={D}"
    assert w2.shape[0] == D and w2.shape[1] == I, \
        f"w2 shape mismatch: got {w2.shape}, expected ({D}, {I})"

    if M == 0 or D == 0:
        return torch.zeros(M, D, device=hidden.device, dtype=hidden.dtype)

    BM, BN, BK = _tune_blocks(hidden.device)

    # Phase 1: gate_proj + up_proj (fused, shares hidden tile load)
    gate = torch.empty(M, I, device=hidden.device, dtype=hidden.dtype)
    up = torch.empty(M, I, device=hidden.device, dtype=hidden.dtype)

    grid_m = triton.cdiv(M, BM)
    grid_n = triton.cdiv(I, BN)
    grid = (grid_m, grid_n)

    _triton_fused_gate_up_kernel[grid](
        hidden,
        w1, w3,
        gate, up,
        M, I, D, swiglu_limit,
        hidden.stride(0), hidden.stride(1),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        BM=BM, BN=BN, BK=BK,
        num_stages=1,
    )

    # Phase 2: SwiGLU activation + down_proj (fused, no intermediate global mem write)
    out = torch.empty(M, D, device=hidden.device, dtype=hidden.dtype)
    grid_down_m = triton.cdiv(M, BM)
    grid_down_n = triton.cdiv(D, BN)
    grid_down = (grid_down_m, grid_down_n)

    _triton_fused_down_kernel[grid_down](
        gate, up,
        w2,
        out,
        M, D, I, swiglu_limit,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        w2.stride(0), w2.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN, BK=BK,
        num_stages=1,
    )
    return out


def fused_expert_ffn_pt(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    gate_out = hidden @ w1.t()
    up_out = hidden @ w3.t()
    g = gate_out.float().clamp(max=swiglu_limit)
    u = up_out.float().clamp(min=-swiglu_limit, max=swiglu_limit)
    activated = (g * g.sigmoid() * u).to(w1.dtype)
    return activated @ w2.t()


# ──────────────────────────────────────────────────────────────────────
# MoE orchestrator classes
# ──────────────────────────────────────────────────────────────────────

class FusedMoEFFN:
    def __init__(
        self,
        num_experts: int = 256,
        intermediate_size: int = 2048,
        hidden_size: int = 4096,
        swiglu_limit: float = 10.0,
        use_triton: bool = True,
    ):
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.swiglu_limit = swiglu_limit
        self.use_triton = use_triton and torch.cuda.is_available()

    def _get_expert_fn(self):
        if self.use_triton:
            return fused_expert_ffn_triton
        return fused_expert_ffn_pt

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        load_expert_fn: Callable,
        layer_idx: int,
    ) -> torch.Tensor:
        B, D = hidden_states.shape
        num_topk = topk_idx.shape[1]
        total_slots = B * num_topk

        if total_slots <= 64:
            return self._forward_legacy(hidden_states, topk_idx, topk_weights,
                                        load_expert_fn, layer_idx)

        try:
            from tile_kernels.torch.moe import inplace_unique_group_indices
            from tile_kernels.moe import get_fused_mapping, expand_to_fused, reduce_fused
        except ImportError:
            return self._forward_legacy(hidden_states, topk_idx, topk_weights,
                                        load_expert_fn, layer_idx)

        topk_idx_i64 = topk_idx.to(torch.int64).contiguous()
        inplace_unique_group_indices(topk_idx_i64, self.num_experts)

        alignment = 32
        num_expanded = (total_slots + (alignment - 1) * self.num_experts) // alignment * alignment
        mapping = get_fused_mapping(
            topk_idx_i64, self.num_experts, num_expanded, alignment=alignment,
        )
        (pos_to_expert, pos_to_token, pos_to_token_topk,
         token_topk_to_pos, expert_start, expert_end,
         num_tokens_per_expert, _) = mapping

        expanded_hidden = expand_to_fused(
            hidden_states.contiguous(), token_topk_to_pos, pos_to_expert)
        expanded_out = torch.zeros_like(expanded_hidden)
        expert_fn = self._get_expert_fn()

        active_experts = sorted(set(
            int(pos_to_expert[i].item())
            for i in range(pos_to_expert.shape[0])
            if pos_to_expert[i].item() >= 0
        ))

        for eid in active_experts:
            if eid < expert_start.shape[0]:
                start = int(expert_start[eid].item())
                end = int(expert_end[eid].item())
            else:
                idxs = (pos_to_expert == eid).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue
                start = int(idxs[0].item())
                end = int(idxs[-1].item()) + 1

            if start >= end or start >= expanded_hidden.shape[0]:
                continue
            end = min(end, expanded_hidden.shape[0])
            h_slice = expanded_hidden[start:end]

            weights = load_expert_fn(layer_idx, eid)
            if weights is None:
                continue

            if len(weights) == 3:
                w1_d, w3_d, w2_d = weights
                out_slice = expert_fn(h_slice, w1_d, w3_d, w2_d, self.swiglu_limit)
            elif len(weights) == 6:
                w1_d, w1_s, w3_d, w3_s, w2_d, w2_s = weights
                out_slice = expert_fn(
                    h_slice, w1_d, w3_d, w2_d, self.swiglu_limit,
                    w1_scale=w1_s, w3_scale=w3_s, w2_scale=w2_s)
            else:
                continue

            expanded_out[start:end] = out_slice.to(expanded_out.dtype)

        topk_weights_f32 = topk_weights.float() if topk_weights.dtype != torch.float32 else topk_weights
        result = reduce_fused(expanded_out, topk_weights_f32, token_topk_to_pos)
        return result.to(hidden_states.dtype)

    def _forward_legacy(
        self, hidden_states, topk_idx, topk_weights,
        load_expert_fn, layer_idx,
    ):
        B, D = hidden_states.shape
        num_topk = topk_idx.shape[1]
        result = torch.zeros_like(hidden_states)

        for k in range(num_topk):
            expert_ids = topk_idx[:, k]
            weights = topk_weights[:, k]
            unique_eids, inverse = torch.unique(expert_ids, return_inverse=True)
            for eid_idx in range(unique_eids.shape[0]):
                eid = int(unique_eids[eid_idx].item())
                if eid < 0:
                    continue
                token_mask = inverse == eid_idx
                if not token_mask.any():
                    continue
                weights_val = load_expert_fn(layer_idx, eid)
                if weights_val is None:
                    continue

                h_batch = hidden_states[token_mask].to(
                    weights_val[0].dtype if weights_val[0].dtype == torch.bfloat16
                    else torch.bfloat16)

                if len(weights_val) == 3:
                    w1_d, w3_d, w2_d = weights_val
                    activated = self._swiglu(h_batch, w1_d, w3_d)
                    out = activated.to(w2_d.dtype) @ w2_d.t()
                elif len(weights_val) == 6:
                    w1_d, w1_s, w3_d, w3_s, w2_d, w2_s = weights_val
                    out = fused_expert_ffn_triton(
                        h_batch, w1_d, w3_d, w2_d, self.swiglu_limit,
                        w1_scale=w1_s, w3_scale=w3_s, w2_scale=w2_s)
                else:
                    continue
                result[token_mask] += out * weights[token_mask].unsqueeze(-1)

        return result

    def _swiglu(self, h, w1, w3):
        gate = h @ w1.t()
        up = h @ w3.t()
        g = gate.float().clamp(max=self.swiglu_limit)
        u = up.float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return (g * g.sigmoid() * u).to(w1.dtype)


class SharedExpertFFN:
    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 2048,
                 swiglu_limit: float = 10.0, use_triton: bool = True):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        self.use_triton = use_triton and torch.cuda.is_available()

    def forward(self, hidden_states: torch.Tensor, w1, w3, w2,
                dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        B, T, D = hidden_states.shape
        h_2d = hidden_states.reshape(-1, D)
        if self.use_triton:
            out = fused_expert_ffn_triton(h_2d, w1, w3, w2, self.swiglu_limit)
        else:
            out = fused_expert_ffn_pt(h_2d, w1, w3, w2, self.swiglu_limit)
        return out.reshape(B, T, D).to(dtype)
