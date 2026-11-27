# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang_omni.models.auk.modulation import _fused, gated_residual, scale_shift


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(1, 7, 32), (3, 17, 1536), (4, 683, 1536)])
@pytest.mark.parametrize(
    "x_dtype,param_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float32, torch.float16),
    ],
)
@torch.inference_mode()
def test_modulation_preserves_eager_rounding(shape, x_dtype, param_dtype):
    torch.manual_seed(42)
    b, _, d = shape
    x = torch.randn(shape, device="cuda", dtype=x_dtype)
    update = torch.randn(shape, device="cuda", dtype=param_dtype)
    # Real AdaLN parameters are noncontiguous chunks of a [B,6D] projection.
    shift, scale, gate, *_ = torch.randn(
        b, 6 * d, device="cuda", dtype=param_dtype
    ).chunk(6, dim=-1)
    expected_scale_shift = x * (1 + scale[:, None]) + shift[:, None]
    expected_residual = x + gate[:, None] * update
    for actual, expected in [
        (_fused(x, scale, shift, modulate=True), expected_scale_shift),
        (scale_shift(x, scale, shift), expected_scale_shift),
        (_fused(x, gate, update, modulate=False), expected_residual),
        (gated_residual(x, gate, update), expected_residual),
    ]:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_modulation_keeps_autograd(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(2, 7, 32, device=device, requires_grad=True)
    params = torch.randn(2, 96, device=device, requires_grad=True)
    shift, scale, gate = params.chunk(3, dim=-1)
    update = torch.randn_like(x, requires_grad=True)
    actual = gated_residual(scale_shift(x, scale, shift), gate, update)
    expected = x * (1 + scale[:, None]) + shift[:, None] + gate[:, None] * update
    actual_grads = torch.autograd.grad(actual.sum(), (x, params, update))
    expected_grads = torch.autograd.grad(expected.sum(), (x, params, update))
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_modulation_handles_sliced_activations():
    x = torch.randn(3, 19, 32, device="cuda")[:, 2:-2]
    update = torch.randn(3, 19, 32, device="cuda", dtype=torch.bfloat16)[:, 2:-2]
    shift, scale, gate = torch.randn(3, 96, device="cuda", dtype=torch.bfloat16).chunk(
        3, -1
    )
    expected_scale_shift = x * (1 + scale[:, None]) + shift[:, None]
    expected_residual = x + gate[:, None] * update
    torch.testing.assert_close(
        scale_shift(x, scale, shift),
        expected_scale_shift,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        gated_residual(x, gate, update), expected_residual, rtol=0, atol=0
    )
