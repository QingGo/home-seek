import torch
import torch.nn.functional as F


class MLAAttention:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = torch.device(device)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.num_kv_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window
        self.index_topk = config.index_topk
        self.index_head_dim = config.index_head_dim
        self.index_n_heads = config.index_n_heads

    def _get_projected_shapes(self, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, o_proj):
        pass

    def forward_mla(
        self,
        hidden_states: torch.Tensor,
        q_a_proj: torch.Tensor,
        q_b_proj: torch.Tensor,
        kv_a_proj: torch.Tensor,
        kv_b_proj: torch.Tensor,
        o_proj: torch.Tensor,
        past_k=None,
        past_v=None,
    ) -> torch.Tensor:
        B, T, D = hidden_states.shape

        q_a = torch.matmul(hidden_states.to(q_a_proj.dtype), q_a_proj.t())  # [B, T, q_lora_rank]
        q_expanded = torch.matmul(q_a, q_b_proj.t())  # [B, T, num_heads * head_dim]
        q = q_expanded.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]

        kv = torch.matmul(hidden_states.to(kv_a_proj.dtype), kv_a_proj.t())
        kv_expanded = torch.matmul(kv, kv_b_proj.t())
        kv_combined = kv_expanded.view(B, T, self.num_kv_heads, -1).transpose(1, 2)
        k = kv_combined[:, :, :, :self.head_dim].contiguous()
        v = kv_combined[:, :, :, self.head_dim:].contiguous()

        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        T_kv = k.shape[-2]
        window = min(self.sliding_window, T_kv)
        k_sliding = k[:, :, -window:, :]
        v_sliding = v[:, :, -window:, :]

        attn = torch.zeros(B, self.num_heads, T, T_kv, device=q.device, dtype=torch.float32)
        scale = self.head_dim ** -0.5

        num_groups = self.num_heads // self.num_kv_heads
        for g in range(num_groups):
            q_g = q[:, g * self.num_kv_heads:(g + 1) * self.num_kv_heads]
            scores = torch.matmul(q_g.float() * scale, k.float().transpose(-2, -1))
            attn[:, g * self.num_kv_heads:(g + 1) * self.num_kv_heads] = scores

        attn_probs = F.softmax(attn, dim=-1).to(v.dtype)
        out = torch.zeros(B, self.num_heads, T, self.head_dim, device=q.device, dtype=v.dtype)
        for g in range(num_groups):
            attn_g = attn_probs[:, g * self.num_kv_heads:(g + 1) * self.num_kv_heads]
            out[:, g * self.num_kv_heads:(g + 1) * self.num_kv_heads] = torch.matmul(attn_g, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = torch.matmul(out.to(o_proj.dtype), o_proj.t())
        return out.to(hidden_states.dtype), k, v


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x_normed).to(x.dtype)
