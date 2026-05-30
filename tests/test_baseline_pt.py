import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from baseline_pt import gemm_residual_rmsnorm, rms_norm, swiglu_unfused


def test_rms_norm_matches_reference_formula_and_returns_bf16():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    weight = torch.ones(4, dtype=torch.bfloat16)

    out = rms_norm(x, weight, eps=1e-6)

    expected = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16
    )
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)


def test_gemm_residual_rmsnorm_uses_matmul_then_residual_then_norm():
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(torch.bfloat16)
    w = torch.eye(4, dtype=torch.bfloat16)
    residual = torch.ones((2, 4), dtype=torch.bfloat16)
    norm_weight = torch.ones(4, dtype=torch.bfloat16)

    out = gemm_residual_rmsnorm(x, w, residual, norm_weight)

    expected = rms_norm((x @ w) + residual, norm_weight)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-2, rtol=1e-2)


def test_swiglu_unfused_uses_separate_gate_and_up_matmuls():
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(torch.bfloat16)
    gate_w = torch.eye(4, dtype=torch.bfloat16)
    up_w = (2 * torch.eye(4, dtype=torch.float32)).to(torch.bfloat16)

    out = swiglu_unfused(x, gate_w, up_w)

    expected = (F.silu(x @ gate_w) * (x @ up_w)).to(torch.bfloat16)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=2e-2)
