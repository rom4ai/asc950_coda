from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only on systems without torch.
    torch = None
    F = None


def _require_torch():
    if torch is None:
        raise ImportError("baseline_pt.py requires PyTorch to run the reference kernels")
    return torch


def _as_bf16(tensor):
    _require_torch()
    return tensor if tensor.dtype == torch.bfloat16 else tensor.to(torch.bfloat16)


def rms_norm(x, weight, eps: float = 1e-6):
    """Reference RMSNorm with BF16 input/output and FP32 reduction."""
    _require_torch()
    x_bf16 = _as_bf16(x)
    weight_bf16 = _as_bf16(weight)

    variance = x_bf16.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x_bf16.float() * torch.rsqrt(variance + eps)
    return (normalized * weight_bf16.float()).to(torch.bfloat16)


def gemm_residual_rmsnorm(x, matmul_weight, residual, norm_weight, eps: float = 1e-6):
    """Unfused matmul, residual add, RMSNorm sequence."""
    _require_torch()
    gemm_out = _as_bf16(x) @ _as_bf16(matmul_weight)
    residual_added = gemm_out + _as_bf16(residual)
    return rms_norm(residual_added, norm_weight, eps=eps)


def matmul_add_rmsnorm(x, matmul_weight, residual, norm_weight, eps: float = 1e-6):
    """Alias for the unfused GEMM + residual add + RMSNorm reference path."""
    return gemm_residual_rmsnorm(x, matmul_weight, residual, norm_weight, eps=eps)


def swiglu_unfused(x, gate_weight, up_weight):
    """Unfused gate matmul, up matmul, and SiLU(gate) * up."""
    _require_torch()
    gate = _as_bf16(x) @ _as_bf16(gate_weight)
    up = _as_bf16(x) @ _as_bf16(up_weight)
    return (F.silu(gate) * up).to(torch.bfloat16)


def moe_gate_up_swiglu(x, gate_weight, up_weight):
    """Alias for the unfused MoE expert gate/up SwiGLU reference path."""
    return swiglu_unfused(x, gate_weight, up_weight)


__all__ = [
    "rms_norm",
    "gemm_residual_rmsnorm",
    "matmul_add_rmsnorm",
    "swiglu_unfused",
    "moe_gate_up_swiglu",
]
