import torch
from typing import Callable


def fused_expert_ffn_triton(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    return fused_expert_ffn_pt(hidden, w1, w3, w2, swiglu_limit)


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
            w1_d, w3_d, w2_d = weights

            out_slice = expert_fn(h_slice, w1_d, w3_d, w2_d, self.swiglu_limit)
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
                w1_d, w3_d, w2_d = weights_val
                h_batch = hidden_states[token_mask].to(w1_d.dtype)
                activated = self._swiglu(h_batch, w1_d, w3_d)
                out = activated.to(w2_d.dtype) @ w2_d.t()
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
