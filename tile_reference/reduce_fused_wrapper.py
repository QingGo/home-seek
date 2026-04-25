import torch
from typing import Optional, Union
from tile_reference.quant_types import QuantTensor


@torch.compile
def elementwise_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return a * b + c


def reduce_fused(
    x: Union[torch.Tensor, QuantTensor],
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
