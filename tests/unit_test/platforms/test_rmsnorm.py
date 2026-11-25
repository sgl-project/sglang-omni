# SPDX-License-Identifier: Apache-2.0
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import reference_rmsnorm
from sglang_omni.platforms.npu import NPUOmniPlatform

HIDDEN_SIZE = 64
VARIANCE_EPSILON = 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_rmsnorm_matches_torch(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(7)
    hidden_states = torch.randn(4, HIDDEN_SIZE, generator=generator).to(dtype)
    weight = (torch.rand(HIDDEN_SIZE, generator=generator) + 0.5).to(dtype)
    expected = F.rms_norm(
        hidden_states.float(), (HIDDEN_SIZE,), weight.float(), eps=VARIANCE_EPSILON
    ).to(dtype)

    normalized = reference_rmsnorm(hidden_states, weight, VARIANCE_EPSILON)

    assert normalized.dtype == dtype
    torch.testing.assert_close(normalized, expected)


@pytest.mark.accelerator
def test_cuda_platform_serves_the_aot_rmsnorm_kernel() -> None:
    sgl_kernel = pytest.importorskip("sgl_kernel")

    assert CUDAOmniPlatform().get_rmsnorm() is sgl_kernel.rmsnorm


def test_the_npu_add_rmsnorm_adapter_returns_the_summed_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``torch_npu.npu_add_rms_norm`` returns ``(normalized, rstd, residual)``;
    the adapter has to surface the last tensor rather than ``rstd``."""
    normalized = torch.randn(4, HIDDEN_SIZE)
    rstd = torch.randn(4, 1)
    residual_sum = torch.randn(4, HIDDEN_SIZE)
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_add_rms_norm = lambda hidden_states, residual, weight, eps: (
        normalized,
        rstd,
        residual_sum,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    added, summed = NPUOmniPlatform().get_add_rmsnorm()(
        torch.randn(4, HIDDEN_SIZE),
        torch.randn(4, HIDDEN_SIZE),
        torch.rand(HIDDEN_SIZE) + 0.5,
        VARIANCE_EPSILON,
    )

    assert added is normalized
    assert summed is residual_sum
