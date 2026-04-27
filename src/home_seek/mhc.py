import torch
import torch.nn.functional as F


def mhc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    B, S, mix_dim = mixes.shape
    pre = torch.sigmoid(mixes[:, :, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[:, :, hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult])
    comb = mixes[:, :, 2*hc_mult:] * hc_scale[2] + hc_base[2*hc_mult:]
    comb = comb.view(B, S, hc_mult, hc_mult)
    comb = F.softmax(comb, dim=-1) + eps
    row_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (row_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        col_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (col_sum + eps)
        row_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (row_sum + eps)
    return pre, post, comb
