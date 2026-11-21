# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.zonos2.fused_norm import bind_norm_kernels, reference_rms_norm
from sglang_omni.platforms.npu import NPUOmniPlatform

WIDTH = 64
EPS = 1e-6


class PlatformWithoutFusedAdd:
    """A platform that ships no fused add kernel."""

    def get_add_rmsnorm(self):
        return None

    def get_rmsnorm(self):
        return reference_rms_norm


def test_platforms_without_a_fused_add_kernel_keep_torch() -> None:
    """Binding must leave both call sites on ``F.rms_norm``: the plain norm is
    bit-identical to the call it replaced, and the residual variant adds before
    normalizing.
    """
    add_kernel, norm_kernel = bind_norm_kernels(PlatformWithoutFusedAdd())
    torch.manual_seed(0)
    x = torch.randn(5, 7, WIDTH)
    residual = torch.randn(5, 7, WIDTH)
    weight = torch.randn(WIDTH)

    assert torch.equal(norm_kernel(x, None, EPS), F.rms_norm(x, (WIDTH,), None, EPS))
    assert torch.equal(
        norm_kernel(x, weight, EPS), F.rms_norm(x, (WIDTH,), weight, EPS)
    )

    normed, summed = add_kernel(x, residual, weight, EPS)
    expected_sum = x + residual

    assert torch.equal(summed, expected_sum)
    assert torch.equal(normed, F.rms_norm(expected_sum, (WIDTH,), weight, EPS))


def test_the_npu_platform_binds_the_fused_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The binding has to come from the platform itself: a plain norm reaches
    ``npu_rms_norm`` with the cached unit gamma, and the residual call returns
    the kernel's summed residual rather than ``rstd``.
    """
    gammas: list[torch.Tensor] = []
    normalized = torch.randn(4, WIDTH)
    rstd = torch.randn(4, 1)
    residual_sum = torch.randn(4, WIDTH)
    torch_npu = types.ModuleType("torch_npu")

    def npu_rms_norm(x, weight, eps):
        gammas.append(weight)
        return x

    def npu_add_rms_norm(x, residual, weight, eps):
        return normalized, rstd, residual_sum

    torch_npu.npu_rms_norm = npu_rms_norm
    torch_npu.npu_add_rms_norm = npu_add_rms_norm
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    add_kernel, norm_kernel = bind_norm_kernels(NPUOmniPlatform())
    x = torch.ones(4, WIDTH)

    norm_kernel(x, None, EPS)
    norm_kernel(x, None, EPS)

    assert len(gammas) == 2
    assert torch.equal(gammas[0], torch.ones(WIDTH))
    assert gammas[0].dtype == x.dtype and gammas[0].device == x.device
    assert gammas[1] is gammas[0]

    added, summed = add_kernel(x, x, torch.ones(WIDTH), EPS)

    assert added is normalized
    assert summed is residual_sum
