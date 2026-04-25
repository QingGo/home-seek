import torch


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
