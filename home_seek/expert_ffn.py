import torch
from tile_reference import swiglu_forward


def swiglu_expert_forward(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    hidden_states = hidden_states.to(dtype=torch.bfloat16)
    gate_proj = gate_proj.to(dtype=torch.bfloat16)
    up_proj = up_proj.to(dtype=torch.bfloat16)
    down_proj = down_proj.to(dtype=torch.bfloat16)

    gate_out = torch.matmul(hidden_states, gate_proj.t())
    up_out = torch.matmul(hidden_states, up_proj.t())

    x = torch.cat([gate_out, up_out], dim=-1).contiguous()
    activated = swiglu_forward(x)

    out = torch.matmul(activated.to(dtype=torch.bfloat16), down_proj.t())
    return out.to(dtype=hidden_states.dtype)
