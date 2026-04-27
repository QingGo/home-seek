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
# Fused SwiGLU + routing + down-projection kernel (eliminates Python routing loop)
# ──────────────────────────────────────────────────────────────────────

@triton.jit
def _triton_swiglu_routedown_kernel(
    gate_ptr, up_ptr, w2_ptr, expert_w_ptr,
    out_ptr,
    B, D, I, I_total, num_e, swiglu_limit,
    stride_gate_b, stride_gate_i,
    stride_up_b, stride_up_i,
    stride_w2_d, stride_w2_i,
    stride_ew_b, stride_ew_e,
    stride_out_b, stride_out_d,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    out_acc = tl.zeros([BM, BN], dtype=tl.float32)

    for k in range(0, I_total, BK):
        offs_k = k + tl.arange(0, BK)

        mask_m = offs_m < B
        mask_k = offs_k < I_total

        gate_ptrs = gate_ptr + offs_m[:, None] * stride_gate_b + offs_k[None, :] * stride_gate_i
        gate_tile = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        up_ptrs = up_ptr + offs_m[:, None] * stride_up_b + offs_k[None, :] * stride_up_i
        up_tile = tl.load(up_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        gate_f32 = gate_tile.to(tl.float32)
        up_f32 = up_tile.to(tl.float32)
        gate_f32 = tl.minimum(gate_f32, swiglu_limit)
        up_f32 = tl.minimum(tl.maximum(up_f32, -swiglu_limit), swiglu_limit)
        activated = gate_f32 * tl.sigmoid(gate_f32.to(tl.float32)) * up_f32

        expert_idx = k // I
        ew_ptrs = expert_w_ptr + offs_m * stride_ew_b + expert_idx * stride_ew_e
        weight = tl.load(ew_ptrs, mask=offs_m < B, other=0.0).to(tl.float32)
        activated = activated * weight[:, None]

        activated_bf16 = activated.to(tl.bfloat16)

        w2_ptrs = w2_ptr + offs_k[:, None] * stride_w2_i + offs_n[None, :] * stride_w2_d
        w2_tile = tl.load(w2_ptrs, mask=(offs_k[:, None] < I_total) & (offs_n[None, :] < D), other=0.0).to(tl.bfloat16)

        out_acc += tl.dot(activated_bf16, w2_tile)

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_b + offs_n[None, :] * stride_out_d
    _out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptrs, out_acc.to(_out_dtype), mask=(offs_m[:, None] < B) & (offs_n[None, :] < D))


def _triton_batched_swiglu_routedown(
    gate: torch.Tensor,
    up: torch.Tensor,
    w2: torch.Tensor,
    expert_weights: torch.Tensor,
    I: int,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    B, I_total = gate.shape
    num_e = expert_weights.shape[1]
    D = w2.shape[0]

    out = torch.empty(B, D, device=gate.device, dtype=torch.bfloat16)

    BM, BN, BK = _tune_blocks(gate.device)

    grid_m = triton.cdiv(B, BM)
    grid_n = triton.cdiv(D, BN)

    _triton_swiglu_routedown_kernel[(grid_m, grid_n)](
        gate, up, w2, expert_weights,
        out,
        B, D, I, I_total, num_e, swiglu_limit,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        w2.stride(0), w2.stride(1),
        expert_weights.stride(0), expert_weights.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN, BK=BK,
        num_stages=1,
    )
    return out


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

    # Fused FP4 path: dequantize + GEMM in single kernel, no intermediate BF16 tensor
    if (_is_fp4_packed(w1) and _is_fp4_packed(w3) and _is_fp4_packed(w2)
            and w1_scale is not None and w3_scale is not None and w2_scale is not None):
        try:
            w1_s = w1_scale.to(torch.float32) if w1_scale.dtype != torch.float32 else w1_scale
            w3_s = w3_scale.to(torch.float32) if w3_scale.dtype != torch.float32 else w3_scale
            w2_s = w2_scale.to(torch.float32) if w2_scale.dtype != torch.float32 else w2_scale
            return _fused_fp4_expert_ffn_triton(
                hidden, w1, w1_s, w3, w3_s, w2, w2_s, swiglu_limit)
        except Exception:
            pass

    # Dequantize FP4 to BF16 (with cache), then use unified BF16 kernels
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
        """Batched decode: collect all unique experts, stack weights, one batched FFN."""
        B, D = hidden_states.shape
        num_topk = topk_idx.shape[1]

        all_eids = set()
        for b in range(B):
            for k in range(num_topk):
                eid = int(topk_idx[b, k].item())
                if eid >= 0:
                    all_eids.add(eid)

        if not all_eids:
            return torch.zeros_like(hidden_states)

        eids_sorted = sorted(all_eids)
        I_inferred = None
        loaded = {}  # eid → (w1_bf16, w3_bf16, w2_bf16)

        for eid in eids_sorted:
            weights_val = load_expert_fn(layer_idx, eid)
            if weights_val is None:
                continue

            if len(weights_val) == 6:
                w1_d, w1_s, w3_d, w3_s, w2_d, w2_s = weights_val
                device = hidden_states.device
                w1_d = w1_d.to(device, non_blocking=True)
                w1_s = w1_s.to(device, non_blocking=True)
                w3_d = w3_d.to(device, non_blocking=True)
                w3_s = w3_s.to(device, non_blocking=True)
                w2_d = w2_d.to(device, non_blocking=True)
                w2_s = w2_s.to(device, non_blocking=True)
                w1_bf16, w3_bf16, w2_bf16 = triton_dequantize_fp4_all(
                    w1_d, w1_s, w3_d, w3_s, w2_d, w2_s)
            elif len(weights_val) == 3:
                w1_d, w3_d, w2_d = weights_val
                device = hidden_states.device
                w1_d = w1_d.to(device, non_blocking=True)
                w3_d = w3_d.to(device, non_blocking=True)
                w2_d = w2_d.to(device, non_blocking=True)
                w1_bf16 = w1_d if w1_d.dtype == torch.bfloat16 else w1_d.to(torch.bfloat16)
                w3_bf16 = w3_d if w3_d.dtype == torch.bfloat16 else w3_d.to(torch.bfloat16)
                w2_bf16 = w2_d if w2_d.dtype == torch.bfloat16 else w2_d.to(torch.bfloat16)
            else:
                continue

            if I_inferred is None:
                I_inferred = w1_bf16.shape[0]
            loaded[eid] = (w1_bf16, w3_bf16, w2_bf16)

        if I_inferred is None:
            return torch.zeros_like(hidden_states)

        I = I_inferred
        loaded_eids = sorted(loaded.keys())
        num_e = len(loaded_eids)

        w1 = torch.cat([loaded[eid][0] for eid in loaded_eids], dim=0)  # [num_e*I, D]
        w3 = torch.cat([loaded[eid][1] for eid in loaded_eids], dim=0)
        w2 = torch.cat([loaded[eid][2] for eid in loaded_eids], dim=1)  # [D, num_e*I]

        gate = hidden_states @ w1.T  # [B, num_e*I]
        up = hidden_states @ w3.T

        # Build per-expert routing weights via scatter (no Python loop)
        expert_weights = torch.zeros(B, num_e, device=hidden_states.device, dtype=torch.bfloat16)
        max_eid = max(loaded_eids) if loaded_eids else 255
        eid_to_idx_t = torch.full((max_eid + 1,), -1, device=hidden_states.device, dtype=torch.int64)
        for i, eid in enumerate(loaded_eids):
            eid_to_idx_t[eid] = i

        valid = topk_idx >= 0
        clamped = topk_idx.clamp(min=0).long()
        mapped = eid_to_idx_t[clamped]
        valid = valid & (mapped >= 0)
        b_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(-1, num_topk)
        if valid.any():
            expert_weights[b_idx[valid], mapped[valid]] = topk_weights[valid].to(torch.bfloat16)

        if self.use_triton and torch.cuda.is_available() and B >= 16:
            return _triton_batched_swiglu_routedown(
                gate, up, w2, expert_weights, I, self.swiglu_limit)
        # cuBLAS path (optimal for M=1 decode; scatter handles routing without Python loop)
        g = gate.float().clamp(max=self.swiglu_limit)
        u = up.float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        activated = (g * g.sigmoid() * u).to(torch.bfloat16)
        activated_3d = activated.view(B, num_e, I)
        weighted = activated_3d * expert_weights.unsqueeze(-1)
        return weighted.reshape(B, num_e * I) @ w2.T

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
        M = h_2d.shape[0]
        # M=1 decode: cuBLAS is 8x faster than Triton (15/16 SM idle)
        if self.use_triton and M > 1:
            out = fused_expert_ffn_triton(h_2d, w1, w3, w2, self.swiglu_limit)
        else:
            out = fused_expert_ffn_pt(h_2d, w1, w3, w2, self.swiglu_limit)
        return out.reshape(B, T, D).to(dtype)


# ──────────────────────────────────────────────────────────────────────
# Triton-accelerated FP4 → BF16 dequantization (Phase 3, fixed interleaving)
# ──────────────────────────────────────────────────────────────────────
_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

_SCALE_GROUP = 32  # one scale per 32 weight columns = 16 packed bytes


@triton.jit
def _triton_dequantize_fp4_kernel(
    packed_ptr, scale_ptr,
    out_ptr,
    N, K_half,
    stride_packed_n, stride_packed_k,
    stride_scale_n, stride_scale_k,
    stride_out_n, stride_out_k,
    lut_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_HALF: tl.constexpr,
):
    """Dequantize FP4-packed int8 weights to BF16 in parallel.

    BLOCK_K_HALF = 16 (one scale group: 16 packed bytes = 32 weights per scale).
    Uses tl.join + tl.reshape for correct interleaving: out[:, 2*k]=lo, out[:, 2*k+1]=hi.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_half = pid_k * BLOCK_K_HALF + tl.arange(0, BLOCK_K_HALF)

    mask_n = offs_n < N
    mask_k_half = offs_k_half < K_half

    packed_ptrs = packed_ptr + offs_n[:, None] * stride_packed_n + offs_k_half[None, :] * stride_packed_k
    packed = tl.load(packed_ptrs, mask=mask_n[:, None] & mask_k_half[None, :], other=0).to(tl.uint8)

    lo = (packed & 0xF).to(tl.int32)
    hi = ((packed >> 4) & 0xF).to(tl.int32)

    lo_f32 = tl.load(lut_ptr + lo).to(tl.float32)
    hi_f32 = tl.load(lut_ptr + hi).to(tl.float32)

    # One scale per BLOCK_K_HALF (16 packed = 32 weights = 1 scale group)
    scale_k_idx = pid_k
    scale_ptrs = scale_ptr + offs_n[:, None] * stride_scale_n + scale_k_idx * stride_scale_k
    scale = tl.load(scale_ptrs, mask=mask_n[:, None], other=1.0).to(tl.float32)
    lo_f32 *= scale
    hi_f32 *= scale

    # Interleave using tl.join + tl.reshape:
    # [BN, KH, 1] join [BN, KH, 1] → [BN, KH, 2] → reshape → [BN, 2*KH]
    # Verified: out[:, 2*k]=lo, out[:, 2*k+1]=hi
    interleaved = tl.reshape(
        tl.join(
            tl.reshape(lo_f32.to(tl.bfloat16), (BLOCK_N, BLOCK_K_HALF, 1)),
            tl.reshape(hi_f32.to(tl.bfloat16), (BLOCK_N, BLOCK_K_HALF, 1)),
        ),
        (BLOCK_N, 2 * BLOCK_K_HALF),
    )

    offs_k_out = pid_k * (2 * BLOCK_K_HALF) + tl.arange(0, 2 * BLOCK_K_HALF)
    mask_k_out = offs_k_out < (K_half * 2)
    out_ptrs = out_ptr + offs_n[:, None] * stride_out_n + offs_k_out[None, :] * stride_out_k
    tl.store(out_ptrs, interleaved, mask=mask_n[:, None] & mask_k_out[None, :])


def triton_dequantize_fp4_to_bf16(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize a single FP4-packed weight matrix to BF16 using Triton.

    Parameters
    ----------
    w_packed : [N, K/2] int8 tensor on GPU
    w_scale : [N, K/32] float32 tensor on GPU

    Returns
    -------
    [N, K] BF16 tensor on GPU
    """
    if w_packed.device.type != "cuda":
        from tile_reference import unpack_from_e2m1fn_x2
        deq = unpack_from_e2m1fn_x2(w_packed)
        sf = w_scale.repeat_interleave(32, dim=1)
        return (deq.float() * sf.float()).to(torch.bfloat16)

    N, K_half = w_packed.shape
    K = K_half * 2

    out = torch.empty(N, K, device=w_packed.device, dtype=torch.bfloat16)
    lut = _FP4_LUT.to(w_packed.device)

    BLOCK_N = 16
    BLOCK_K_HALF = 16  # 16 packed bytes = 32 weights = 1 scale group

    grid_n = triton.cdiv(N, BLOCK_N)
    grid_k = triton.cdiv(K_half, BLOCK_K_HALF)
    grid = (grid_n, grid_k)

    _triton_dequantize_fp4_kernel[grid](
        w_packed, w_scale,
        out,
        N, K_half,
        w_packed.stride(0), w_packed.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0), out.stride(1),
        lut,
        BLOCK_N=BLOCK_N, BLOCK_K_HALF=BLOCK_K_HALF,
        num_stages=1,
    )
    return out


def triton_dequantize_fp4_all(
    w1_packed: torch.Tensor, w1_scale: torch.Tensor,
    w3_packed: torch.Tensor, w3_scale: torch.Tensor,
    w2_packed: torch.Tensor, w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantize all three FP4 weight matrices using Triton in parallel."""
    return (
        triton_dequantize_fp4_to_bf16(w1_packed, w1_scale),
        triton_dequantize_fp4_to_bf16(w3_packed, w3_scale),
        triton_dequantize_fp4_to_bf16(w2_packed, w2_scale),
    )


# ──────────────────────────────────────────────────────────────────────
# Fused FP4 dequantize + GEMM kernels (Phase 3 fixed)
# BK=32 aligns with one scale group (32 columns = 16 packed bytes).
# ──────────────────────────────────────────────────────────────────────

@triton.jit
def _triton_fp4_fused_gate_up_kernel(
    hidden_ptr,
    w1_packed_ptr, w1_scale_ptr,
    w3_packed_ptr, w3_scale_ptr,
    gate_out_ptr, up_out_ptr,
    M, I, D,
    stride_hm, stride_hk,
    stride_w1_n, stride_w1_k,
    stride_s1_n, stride_s1_k,
    stride_w3_n, stride_w3_k,
    stride_s3_n, stride_s3_k,
    stride_gm, stride_gn,
    stride_um, stride_un,
    lut_ptr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Fused gate+up projection: loads packed FP4, dequantizes inline, computes GEMM."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    gate_acc = tl.zeros([BM, BN], dtype=tl.float32)
    up_acc = tl.zeros([BM, BN], dtype=tl.float32)

    BK_HALF: tl.constexpr = BK // 2  # packed bytes per inner loop (=16)

    for k in range(0, D, BK):
        mask_k = (k + offs_k) < D
        mask_m = offs_m < M

        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + (k + offs_k)[None, :] * stride_hk
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        k_half = k // 2
        offs_kh = k_half + tl.arange(0, BK_HALF)

        # --- w1: load packed, dequantize inline ---
        # Load in [BN, BK_HALF] layout, interleave → [BN, BK], then trans → [BK, BN] for tl.dot
        w1p_ptrs = w1_packed_ptr + offs_n[:, None] * stride_w1_n + offs_kh[None, :] * stride_w1_k
        w1p = tl.load(w1p_ptrs, mask=(offs_n[:, None] < I) & (offs_kh[None, :] < D // 2), other=0).to(tl.uint8)
        w1_lo = (w1p & 0xF).to(tl.int32)
        w1_hi = ((w1p >> 4) & 0xF).to(tl.int32)
        w1_lo_f32 = tl.load(lut_ptr + w1_lo).to(tl.float32)
        w1_hi_f32 = tl.load(lut_ptr + w1_hi).to(tl.float32)
        w1_sg = k // 32
        w1s_ptrs = w1_scale_ptr + offs_n[:, None] * stride_s1_n + w1_sg * stride_s1_k
        w1s = tl.load(w1s_ptrs, mask=offs_n[:, None] < I, other=1.0).to(tl.float32)
        w1_lo_f32 *= w1s
        w1_hi_f32 *= w1s
        # Interleave: [BN, BK_HALF] → [BN, BK_HALF, 1] join → [BN, BK_HALF, 2] → reshape → [BN, BK]
        w1_inter = tl.reshape(
            tl.join(
                tl.reshape(w1_lo_f32.to(tl.bfloat16), (BN, BK_HALF, 1)),
                tl.reshape(w1_hi_f32.to(tl.bfloat16), (BN, BK_HALF, 1)),
            ),
            (BN, BK),
        )
        w1_deq = tl.trans(w1_inter, 1, 0)  # [BK, BN] for tl.dot

        # --- w3: load packed, dequantize inline ---
        w3p_ptrs = w3_packed_ptr + offs_n[:, None] * stride_w3_n + offs_kh[None, :] * stride_w3_k
        w3p = tl.load(w3p_ptrs, mask=(offs_n[:, None] < I) & (offs_kh[None, :] < D // 2), other=0).to(tl.uint8)
        w3_lo = (w3p & 0xF).to(tl.int32)
        w3_hi = ((w3p >> 4) & 0xF).to(tl.int32)
        w3_lo_f32 = tl.load(lut_ptr + w3_lo).to(tl.float32)
        w3_hi_f32 = tl.load(lut_ptr + w3_hi).to(tl.float32)
        w3_sg = k // 32
        w3s_ptrs = w3_scale_ptr + offs_n[:, None] * stride_s3_n + w3_sg * stride_s3_k
        w3s = tl.load(w3s_ptrs, mask=offs_n[:, None] < I, other=1.0).to(tl.float32)
        w3_lo_f32 *= w3s
        w3_hi_f32 *= w3s
        w3_inter = tl.reshape(
            tl.join(
                tl.reshape(w3_lo_f32.to(tl.bfloat16), (BN, BK_HALF, 1)),
                tl.reshape(w3_hi_f32.to(tl.bfloat16), (BN, BK_HALF, 1)),
            ),
            (BN, BK),
        )
        w3_deq = tl.trans(w3_inter, 1, 0)

        gate_acc += tl.dot(h, w1_deq)
        up_acc += tl.dot(h, w3_deq)

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < I
    gate_ptrs = gate_out_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    up_ptrs = up_out_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un
    _out_dtype = gate_out_ptr.dtype.element_ty
    tl.store(gate_ptrs, gate_acc.to(_out_dtype), mask=mask_m & mask_n)
    tl.store(up_ptrs, up_acc.to(_out_dtype), mask=mask_m & mask_n)


@triton.jit
def _triton_fp4_fused_down_kernel(
    gate_ptr, up_ptr,
    w2_packed_ptr, w2_scale_ptr,
    out_ptr,
    M, D, I, swiglu_limit,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_w2_n, stride_w2_k,
    stride_s2_n, stride_s2_k,
    stride_om, stride_on,
    lut_ptr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Fused SwiGLU + down projection with inline FP4 dequantization."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    out_acc = tl.zeros([BM, BN], dtype=tl.float32)

    BK_HALF: tl.constexpr = BK // 2

    for k in range(0, I, BK):
        mask_k = (k + offs_k) < I
        mask_m = offs_m < M

        gate_ptrs = gate_ptr + offs_m[:, None] * stride_gm + (k + offs_k)[None, :] * stride_gn
        gate_tile = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        up_ptrs = up_ptr + offs_m[:, None] * stride_um + (k + offs_k)[None, :] * stride_un
        up_tile = tl.load(up_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        gate_f32 = gate_tile.to(tl.float32)
        up_f32 = up_tile.to(tl.float32)
        gate_f32 = tl.minimum(gate_f32, swiglu_limit)
        up_f32 = tl.minimum(tl.maximum(up_f32, -swiglu_limit), swiglu_limit)
        activated = gate_f32 * tl.sigmoid(gate_f32.to(tl.float32)) * up_f32

        k_half = k // 2
        offs_kh = k_half + tl.arange(0, BK_HALF)

        # Load in [BN, BK_HALF], interleave → [BN, BK], trans → [BK, BN] for tl.dot
        w2p_ptrs = w2_packed_ptr + offs_n[:, None] * stride_w2_n + offs_kh[None, :] * stride_w2_k
        w2p = tl.load(w2p_ptrs, mask=(offs_n[:, None] < D) & (offs_kh[None, :] < I // 2), other=0).to(tl.uint8)
        w2_lo = (w2p & 0xF).to(tl.int32)
        w2_hi = ((w2p >> 4) & 0xF).to(tl.int32)
        w2_lo_f32 = tl.load(lut_ptr + w2_lo).to(tl.float32)
        w2_hi_f32 = tl.load(lut_ptr + w2_hi).to(tl.float32)
        w2_sg = k // 32
        w2s_ptrs = w2_scale_ptr + offs_n[:, None] * stride_s2_n + w2_sg * stride_s2_k
        w2s = tl.load(w2s_ptrs, mask=offs_n[:, None] < D, other=1.0).to(tl.float32)
        w2_lo_f32 *= w2s
        w2_hi_f32 *= w2s
        w2_inter = tl.reshape(
            tl.join(
                tl.reshape(w2_lo_f32.to(tl.bfloat16), (BN, BK_HALF, 1)),
                tl.reshape(w2_hi_f32.to(tl.bfloat16), (BN, BK_HALF, 1)),
            ),
            (BN, BK),
        )
        w2_deq = tl.trans(w2_inter, 1, 0)  # [BK, BN]

        out_acc += tl.dot(activated.to(w2_deq.dtype), w2_deq)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    _out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptrs, out_acc.to(_out_dtype), mask=(offs_m[:, None] < M) & (offs_n[None, :] < D))


def _fused_fp4_expert_ffn_triton(
    hidden: torch.Tensor,
    w1_packed: torch.Tensor, w1_scale: torch.Tensor,
    w3_packed: torch.Tensor, w3_scale: torch.Tensor,
    w2_packed: torch.Tensor, w2_scale: torch.Tensor,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """Fused FP4+GEMM: dequantization happens inside the GEMM loop, no intermediate BF16 tensor."""
    M = hidden.shape[0]
    D = hidden.shape[1]
    I = w1_packed.shape[0]

    gate = torch.empty(M, I, device=hidden.device, dtype=hidden.dtype)
    up = torch.empty(M, I, device=hidden.device, dtype=hidden.dtype)

    lut = _FP4_LUT.to(hidden.device)

    BM, BN, _ = _tune_blocks(hidden.device)
    BK_FP4 = 32  # aligned to one scale group

    grid_m = triton.cdiv(M, BM)
    grid_n = triton.cdiv(I, BN)

    _triton_fp4_fused_gate_up_kernel[(grid_m, grid_n)](
        hidden,
        w1_packed, w1_scale,
        w3_packed, w3_scale,
        gate, up,
        M, I, D,
        hidden.stride(0), hidden.stride(1),
        w1_packed.stride(0), w1_packed.stride(1),
        w1_scale.stride(0), w1_scale.stride(1),
        w3_packed.stride(0), w3_packed.stride(1),
        w3_scale.stride(0), w3_scale.stride(1),
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        lut,
        BM=BM, BN=BN, BK=BK_FP4,
        num_stages=1,
    )

    out = torch.empty(M, D, device=hidden.device, dtype=hidden.dtype)
    grid_d_m = triton.cdiv(M, BM)
    grid_d_n = triton.cdiv(D, BN)

    _triton_fp4_fused_down_kernel[(grid_d_m, grid_d_n)](
        gate, up,
        w2_packed, w2_scale,
        out,
        M, D, I, swiglu_limit,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        w2_packed.stride(0), w2_packed.stride(1),
        w2_scale.stride(0), w2_scale.stride(1),
        out.stride(0), out.stride(1),
        lut,
        BM=BM, BN=BN, BK=BK_FP4,
        num_stages=1,
    )
    return out
