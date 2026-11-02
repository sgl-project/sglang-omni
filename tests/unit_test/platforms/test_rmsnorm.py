# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F

from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import reference_rmsnorm

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
