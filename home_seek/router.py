import torch
import torch.nn.functional as F
from tile_reference import stable_topk


def compute_expert_affinity(hidden_states: torch.Tensor, gate_weight: torch.Tensor, top_k: int = 6):
    logits = torch.matmul(hidden_states.to(gate_weight.dtype), gate_weight.t())
    logits = logits.float()
    scores = F.softplus(logits).sqrt()
    topk_indices = stable_topk(scores, top_k)
    topk_weights = scores.gather(1, topk_indices)
    topk_sum = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    topk_weights = topk_weights / topk_sum
    return topk_indices, topk_weights


def compute_expert_affinity_with_bias(
    hidden_states: torch.Tensor, gate_weight: torch.Tensor,
    gate_bias: torch.Tensor, top_k: int = 6,
    routed_scaling_factor: float = 1.5,
):
    logits = torch.matmul(hidden_states.to(gate_weight.dtype), gate_weight.t())
    if gate_bias is not None:
        logits = logits + gate_bias.to(logits.dtype)
    logits = logits.float()
    scores = F.softplus(logits).sqrt()
    topk_indices = stable_topk(scores, top_k)
    topk_weights_raw = scores.gather(1, topk_indices)
    topk_sum = topk_weights_raw.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    topk_weights = topk_weights_raw / topk_sum * routed_scaling_factor
    return topk_indices, topk_weights


def compute_shared_expert_affinity(hidden_states: torch.Tensor, gate_weight: torch.Tensor):
    logits = torch.matmul(hidden_states.to(gate_weight.dtype), gate_weight.t())
    return torch.sigmoid(logits.float())
