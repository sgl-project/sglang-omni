# SPDX-License-Identifier: Apache-2.0
"""The vendor RMSNorm patch stays on the fused-op dispatch path."""

from __future__ import annotations

import pytest
import torch
from sglang.kernels import fused_op

from sglang_omni.vendor.sglang.layers import RMSNorm


@pytest.fixture(autouse=True)
def _cuda_dispatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fused_op, "_platform_key", lambda: "cuda")


def _inputs(residual_dtype: torch.dtype, post_dtype: torch.dtype | None):
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    residual = torch.randn(4, 8, dtype=residual_dtype)
    post = None if post_dtype is None else torch.randn(4, 8, dtype=post_dtype)
    return x, residual, post


def test_residual_dtype_mismatch_takes_the_native_path():
    norm = RMSNorm(8, eps=1e-6)
    x, residual, _ = _inputs(torch.float32, None)

    out, out_residual = norm(x.clone(), residual.clone())

    expected, expected_residual = norm.forward_native(x.clone(), residual.clone())
    assert torch.equal(out, expected)
    assert torch.equal(out_residual, expected_residual)


def test_post_residual_addition_dtype_mismatch_takes_the_native_path():
    norm = RMSNorm(8, eps=1e-6)
    x, residual, post = _inputs(torch.bfloat16, torch.float32)

    out, out_residual = norm(
        x.clone(), residual.clone(), post_residual_addition=post.clone()
    )

    expected, expected_residual = norm.forward_native(
        x.clone(), residual.clone(), post_residual_addition=post.clone()
    )
    assert torch.equal(out, expected)
    assert torch.equal(out_residual, expected_residual)


def test_zero_tokens_keep_the_upstream_contract_across_dtypes():
    norm = RMSNorm(8, eps=1e-6)
    x = torch.empty(0, 8, dtype=torch.bfloat16)
    residual = torch.empty(0, 8, dtype=torch.float32)
    post = torch.empty(0, 8, dtype=torch.float32)

    out, out_residual = norm(x, residual, post_residual_addition=post)

    assert out is x
    assert out_residual.dtype == torch.float32


def test_forward_kwargs_reach_the_native_fallback():
    norm = RMSNorm(8, eps=1e-6)
    x, residual, _ = _inputs(torch.float32, None)

    out, out_residual = norm(
        x.clone(), residual.clone(), quant_linear=torch.nn.Linear(8, 8)
    )

    expected, expected_residual = norm.forward_native(x.clone(), residual.clone())
    assert torch.equal(out, expected)
    assert torch.equal(out_residual, expected_residual)
