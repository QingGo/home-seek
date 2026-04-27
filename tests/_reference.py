import torch
from typing import Optional, Union


def swiglu_forward(
    x: torch.Tensor,
    pos_to_token_topk: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    swiglu_clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float32)

    num_expanded_tokens, hidden2 = x.shape
    assert hidden2 % 2 == 0
    hidden = hidden2 // 2

    if pos_to_token_topk is not None:
        assert pos_to_token_topk.dim() == 1
        assert pos_to_token_topk.shape[0] == num_expanded_tokens
        assert topk_weights is not None
        assert topk_weights.dim() == 2

    x_fp32 = x.float()
    x_left = x_fp32[:, :hidden]
    x_right = x_fp32[:, hidden:]

    if swiglu_clamp_value is not None:
        if clamped_count is not None:
            clamped_count[0] += (x_left > swiglu_clamp_value).sum()
            clamped_count[1] += (x_right > swiglu_clamp_value).sum()
            clamped_count[2] += (x_right < -swiglu_clamp_value).sum()
        x_left = torch.clamp(x_left, max=swiglu_clamp_value)
        x_right = torch.clamp(x_right, min=-swiglu_clamp_value, max=swiglu_clamp_value)

    out = x_left / (1.0 + torch.exp(-x_left)) * x_right

    if pos_to_token_topk is not None:
        num_tokens, num_topk = topk_weights.shape
        pos_mask = pos_to_token_topk >= 0
        token_indices = torch.div(pos_to_token_topk[pos_mask], num_topk, rounding_mode='floor')
        topk_indices = pos_to_token_topk[pos_mask] % num_topk
        w_expanded = torch.zeros(num_expanded_tokens, device=x.device, dtype=torch.float32)
        w_expanded[pos_mask] = topk_weights[token_indices, topk_indices].float()
        out = out * w_expanded.unsqueeze(1)

    return out


@torch.compile
def elementwise_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return a * b + c


def reduce_fused(
    x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    topk_weights: Optional[torch.Tensor],
    token_topk_to_pos: torch.Tensor,
    fp8_format: str = '',
    sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(x, tuple):
        x, x_sf = x
    else:
        x_sf = None

    num_expanded_tokens, hidden = x.shape
    num_tokens, num_topk = token_topk_to_pos.shape

    out_dtype = torch.float8_e4m3fn if fp8_format == 'e4m3' else x.dtype

    if num_tokens == 0:
        return torch.empty((0, hidden), dtype=out_dtype, device=x.device)

    reduced = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x.device)
    valid = token_topk_to_pos >= 0

    for k in range(num_topk):
        pos_k = token_topk_to_pos[:, k]
        mask_k = valid[:, k]

        if not mask_k.any():
            continue

        safe_pos = pos_k.clamp(min=0)
        rows = x[safe_pos].float()

        s = torch.ones(num_tokens, dtype=torch.float32, device=x.device)
        if topk_weights is not None:
            s = topk_weights[:, k].clone()
        if x_sf is not None:
            s = s * x_sf[safe_pos]

        result = elementwise_fma(rows, s.unsqueeze(1), reduced)
        reduced = torch.where(mask_k.unsqueeze(1), result, reduced)

    if sf is not None:
        reduced = reduced * sf[0].item()

    return reduced.to(out_dtype)


def expand_to_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden = x.shape
    num_expanded_tokens = pos_to_expert.shape[0]

    out = torch.zeros((num_expanded_tokens, hidden), dtype=x.dtype, device=x.device)

    pos_flat = token_topk_to_pos.reshape(-1)
    mask = pos_flat >= 0
    valid_pos = pos_flat[mask]
    num_topk = token_topk_to_pos.shape[1]
    x_repeated = x.unsqueeze(1).expand(-1, num_topk, -1).reshape(-1, hidden)
    out[valid_pos] = x_repeated[mask]

    return out
