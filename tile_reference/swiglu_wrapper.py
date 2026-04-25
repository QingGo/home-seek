import torch
from typing import Optional


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
