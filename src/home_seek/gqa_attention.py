"""GQA fused attention kernel — eliminates 64x KV expand for n_kv=1.

Replaces the PyTorch expand + batched matmul + softmax + matmul pattern
with a fused Triton kernel that reads KV once and reuses for all H=64 heads.

Per-head program: load q[D], loop over KV blocks, online softmax, accumulate out.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_fused_attn_kernel(
    q_ptr, kv_ptr, out_ptr,
    mask_ptr, sink_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kvb, stride_kvt, stride_kvd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, H, T_q, T_kv, D,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_t * stride_qt + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=offs_d < D).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    m_i = tl.full([1], float('-inf'), tl.float32)
    l_i = tl.zeros([1], tl.float32)
    acc_o = tl.zeros([BLOCK_D], tl.float32)

    for start_kv in range(0, T_kv, BLOCK_KV):
        offs_kv = start_kv + tl.arange(0, BLOCK_KV)
        kv_ptrs = kv_ptr + pid_b * stride_kvb + offs_kv[:, None] * stride_kvt + offs_d[None, :] * stride_kvd
        mask_kv = (offs_kv[:, None] < T_kv) & (offs_d[None, :] < D)
        k = tl.load(kv_ptrs, mask=mask_kv).to(tl.float32)

        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(offs_kv < T_kv, scores, float('-inf'))

        if HAS_MASK:
            m = tl.load(mask_ptr + pid_t * T_kv + offs_kv, mask=offs_kv < T_kv)
            scores = scores + m

        m_prev = m_i
        m_local = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_local)
        alpha = tl.exp(m_prev - m_new)
        p = tl.exp(scores - m_new)
        l_i = alpha * l_i + tl.sum(p, axis=0)

        v = tl.load(kv_ptrs, mask=mask_kv).to(tl.float32)
        acc_o = alpha * acc_o + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    if HAS_SINK:
        sink_val = tl.load(sink_ptr + pid_h)
        m_prev = m_i
        m_new = tl.maximum(m_prev, sink_val)
        alpha = tl.exp(m_prev - m_new)
        l_i = alpha * l_i + tl.exp(sink_val - m_new)
        acc_o = alpha * acc_o
        m_i = m_new

    acc_o = acc_o / l_i

    out_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_t * stride_ot + offs_d * stride_od
    tl.store(out_ptrs, acc_o.to(tl.float32), mask=offs_d < D)


def gqa_fused_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    causal_mask: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    B, H, T_q, D = q.shape
    if kv.dim() == 4:
        T_kv = kv.shape[-2]
    else:
        _, T_kv, _ = kv.shape
        kv = kv.unsqueeze(1)

    out = torch.empty(B, H, T_q, D, device=q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_KV = 64

    grid = (B, H, T_q)

    has_mask = causal_mask is not None
    has_sink = attn_sink is not None and attn_sink.numel() == H

    mask_arg = causal_mask if has_mask else q
    sink_arg = attn_sink if has_sink else q

    _gqa_fused_attn_kernel[grid](
        q, kv, out,
        mask_arg, sink_arg,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(2), kv.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T_q, T_kv, D,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        HAS_MASK=has_mask,
        HAS_SINK=has_sink,
    )

    return out
