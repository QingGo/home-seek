"""Shared utilities for the inference engine."""

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x_normed).to(x.dtype)
