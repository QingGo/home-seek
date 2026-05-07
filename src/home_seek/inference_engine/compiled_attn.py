"""torch.compile attention kernels for M=1 decode.

Provides compiled versions of QKV projection, attention computation, and
output projection. Gated behind `eng._compiled_attn_enabled` flag.

Only active for T=1 (decode) — prefill falls back to uncompiled path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from home_seek.utils import rms_norm


# ── Lazy compilation (module-level, triggered on first use) ──

_compiled_qkv_fn = None
_compiled_compute_fn = None


def _get_compiled_attn_qkv():
    global _compiled_qkv_fn
    if _compiled_qkv_fn is None and torch.cuda.is_available():
        _compiled_qkv_fn = torch.compile(
            _attn_qkv_proj_impl, dynamic=True, mode="reduce-overhead"
        )
    return _compiled_qkv_fn or _attn_qkv_proj_impl


def _get_compiled_attn_compute():
    global _compiled_compute_fn
    if _compiled_compute_fn is None and torch.cuda.is_available():
        _compiled_compute_fn = torch.compile(
            _attn_compute_decode_impl, dynamic=True, mode="reduce-overhead"
        )
    return _compiled_compute_fn or _attn_compute_decode_impl


# ── Module-level pure tensor implementations (compile targets) ──


def _attn_qkv_proj_impl(
    hidden_states: torch.Tensor,
    wq_a: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    q_norm: torch.Tensor | None,
    kv_norm: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    num_attention_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    rms_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """QKV projection + RoPE. Pure tensor computation.

    Returns (q, kv_latent) where:
      q:         [B, H, T, head_dim]
      kv_latent: [B, T, H * (qk_rope_head_dim + qk_nope_head_dim)]
    """
    B, T, D = hidden_states.shape

    q_latent = torch.matmul(hidden_states.to(wq_a.dtype), wq_a.t())
    if q_norm is not None:
        q_latent = rms_norm(q_latent, q_norm.to(q_latent.dtype), rms_norm_eps)
    q = torch.matmul(q_latent, wq_b.t())
    q = q.view(B, T, num_attention_heads, head_dim).transpose(1, 2)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + rms_norm_eps)

    kv_latent = torch.matmul(hidden_states.to(wkv.dtype), wkv.t())
    if kv_norm is not None:
        kv_latent = rms_norm(kv_latent, kv_norm.to(kv_latent.dtype), rms_norm_eps)

    q = _apply_compiled_rope(q, freqs_cis, rd=qk_rope_head_dim, inverse=False)
    kv_latent = _apply_compiled_rope(kv_latent, freqs_cis, rd=qk_rope_head_dim, inverse=False)
    return q, kv_latent


def _attn_compute_decode_impl(
    q: torch.Tensor,
    k_all: torch.Tensor,
    attn_sink: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    wo_a: torch.Tensor | None,
    wo_b: torch.Tensor | None,
    num_attention_heads: int,
    head_dim: int,
    num_key_value_heads: int,
    qk_rope_head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    hidden_size: int,
    use_gqa_fusion: bool,
    return_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Attention computation + output projection. Pure tensor computation.

    Args:
        q:        [B, H, 1, head_dim]
        k_all:    [B, 1, T_kv, head_dim] (already expanded from K, V in GQA)
        ...
    Returns:
        out: [B, 1, hidden_size]
    """
    B, H, T_q, D = q.shape
    T_kv = k_all.shape[-2]

    if use_gqa_fusion:
        from home_seek.gqa_attention import gqa_fused_attn

        _has_sink = attn_sink is not None and attn_sink.numel() == num_attention_heads
        out = gqa_fused_attn(
            q.float(),
            k_all.to(q.dtype),
            causal_mask=None,
            attn_sink=attn_sink.float() if _has_sink else None,
        ).to(return_dtype)
    else:
        n_groups = num_attention_heads // num_key_value_heads
        scale_f = head_dim ** -0.5
        v_all = k_all

        k_exp = k_all.unsqueeze(1).expand(-1, n_groups, -1, -1, -1)
        k_expanded = k_exp.reshape(B, -1, T_kv, D)
        v_exp = v_all.unsqueeze(1).expand(-1, n_groups, -1, -1, -1)
        v_expanded = v_exp.reshape(B, -1, T_kv, D)

        attn = torch.matmul(q.float() * scale_f, k_expanded.float().transpose(-2, -1))

        if attn_sink is not None and attn_sink.numel() == num_attention_heads:
            sink_val = attn_sink.view(1, -1, 1, 1).to(attn.dtype)
            attn_with_sink = torch.cat([attn, sink_val.expand(-1, -1, T_q, -1)], dim=-1)
            P_all = F.softmax(attn_with_sink, dim=-1)
            attn_p = P_all[:, :, :, :-1].to(v_expanded.dtype)
        else:
            attn_p = F.softmax(attn, dim=-1).to(v_expanded.dtype)
        out = torch.matmul(attn_p, v_expanded)

    out = _apply_compiled_rope(out, freqs_cis, rd=qk_rope_head_dim, inverse=True)

    out = out.transpose(1, 2).contiguous()

    if wo_a is not None and wo_b is not None:
        out_g = out.view(B, T_q, o_groups, -1)
        wo_a_g = wo_a.view(o_groups, o_lora_rank, hidden_size)
        out_combined = torch.einsum('btgd,grd->btgr', out_g.to(wo_a_g.dtype), wo_a_g)
        out_combined = out_combined.reshape(B, T_q, -1)
        out = torch.matmul(out_combined.to(wo_b.dtype), wo_b.t())
    else:
        out = out.view(B, T_q, H * D)

    return out.to(return_dtype)


def _apply_compiled_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rd: int = 64,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotary position embedding — copy of engine-level apply_rotary_emb for compile."""
    if rd <= 0:
        return x
    x_rope = x[..., -rd:]
    x_pass = x[..., :-rd] if x.shape[-1] > rd else None
    T_match = x_rope.shape[-2]

    if x_rope.ndim == 4:
        h = x_rope.shape[1]
        freqs = freqs_cis[:T_match].view(1, 1, T_match, rd // 2).expand(-1, h, -1, -1)
    else:
        freqs = freqs_cis[:T_match].view(1, T_match, rd // 2)

    if inverse:
        freqs = freqs.conj()

    x_rope_complex = torch.view_as_real(
        torch.view_as_complex(x_rope.float().reshape(*x_rope.shape[:-1], -1, 2))
    )
    rotated = torch.view_as_real(
        x_rope_complex * freqs.unsqueeze(-1)
    ).flatten(-2)

    if x_pass is not None:
        return torch.cat([x_pass.float(), rotated.to(x_pass.dtype)], dim=-1).to(x.dtype)
    return rotated.to(x.dtype)


# ── Public dispatch functions ──


def _attn_qkv_proj(
    hidden_states: torch.Tensor,
    wq_a: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    q_norm: torch.Tensor | None,
    kv_norm: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    num_attention_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    rms_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = _get_compiled_attn_qkv()
    return fn(hidden_states, wq_a, wq_b, wkv, q_norm, kv_norm, freqs_cis,
              num_attention_heads, head_dim, qk_rope_head_dim, rms_norm_eps)


def _attn_compute_decode(
    q: torch.Tensor,
    k_all: torch.Tensor,
    attn_sink: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    wo_a: torch.Tensor | None,
    wo_b: torch.Tensor | None,
    num_attention_heads: int,
    head_dim: int,
    num_key_value_heads: int,
    qk_rope_head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    hidden_size: int,
    use_gqa_fusion: bool,
    return_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    fn = _get_compiled_attn_compute()
    return fn(q, k_all, attn_sink, freqs_cis, wo_a, wo_b,
              num_attention_heads, head_dim, num_key_value_heads,
              qk_rope_head_dim, o_groups, o_lora_rank, hidden_size,
              use_gqa_fusion, return_dtype)
