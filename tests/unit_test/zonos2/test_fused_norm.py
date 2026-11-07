# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import sglang_omni.models.zonos2.fused_norm as fused_norm
from sglang_omni.models.zonos2.fused_norm import (
    _resolve_kernels,
    add_rms_norm,
    no_affine_gamma,
    rms_norm,
)

WIDTH = 64
EPS = 1e-6


@pytest.fixture(autouse=True)
def no_platform_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to the reference normalization on the running platform.

    The real bindings are picked at import time and an accelerator kernel
    rejects CPU tensors, so each case opts into the kernel it exercises.
    """
    monkeypatch.setattr(fused_norm, "_RMSNORM_KERNEL", fused_norm._reference_rms_norm)
    monkeypatch.setattr(fused_norm, "_ADD_RMSNORM_KERNEL", None)


def test_rms_norm_without_a_platform_kernel_matches_torch() -> None:
    """Platforms that ship no fused kernel keep the plain normalization, so the
    wrapper must be bit-identical to the call it replaced.
    """
    torch.manual_seed(0)
    x = torch.randn(5, 7, WIDTH)
    weight = torch.randn(WIDTH)

    assert torch.equal(rms_norm(x, None, EPS), F.rms_norm(x, (WIDTH,), None, EPS))
    assert torch.equal(rms_norm(x, weight, EPS), F.rms_norm(x, (WIDTH,), weight, EPS))


def test_rms_norm_hands_the_stand_in_gamma_to_the_platform_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kernel cannot take ``weight=None``, so the no-affine call site must
    reach it with the cached all-ones gamma instead.
    """
    seen: list[tuple[torch.Tensor, ...]] = []

    def kernel(x, weight, eps):
        seen.append((x, weight, eps))
        return x

    monkeypatch.setattr(
        fused_norm, "_RMSNORM_KERNEL", fused_norm._platform_rms_norm(kernel)
    )
    x, weight = torch.ones(3, WIDTH), torch.ones(WIDTH)

    rms_norm(x, None, EPS)
    rms_norm(x, weight, EPS)

    assert len(seen) == 2
    assert seen[0][1] is no_affine_gamma(x)
    assert seen[0][2] == EPS
    assert seen[1][1] is weight


def test_add_rms_norm_without_a_fused_kernel_sums_then_normalizes() -> None:
    """The return contract is ``(normed_of_sum, sum)``; the second value is the
    residual stream the caller keeps adding to, and it must stay un-normalized.
    """
    torch.manual_seed(0)
    x = torch.randn(5, WIDTH)
    residual = torch.randn(5, WIDTH)
    weight = torch.randn(WIDTH)

    normed, summed = add_rms_norm(x, residual, weight, EPS)
    expected_sum = x + residual

    assert torch.equal(summed, expected_sum)
    assert torch.equal(normed, F.rms_norm(expected_sum, (WIDTH,), weight, EPS))


def test_add_rms_norm_prefers_the_fused_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On devices that provide the fused add kernel the separate add is skipped,
    and the caller receives the kernel's own residual sum.
    """
    seen: list[tuple[torch.Tensor, ...]] = []

    def fused(x, residual, weight, eps):
        seen.append((x, residual, weight, eps))
        summed = x + residual
        return summed * 2, summed

    monkeypatch.setattr(fused_norm, "_ADD_RMSNORM_KERNEL", fused)
    x, residual, weight = torch.ones(4, WIDTH), torch.ones(4, WIDTH), torch.ones(WIDTH)

    normed, summed = add_rms_norm(x, residual, weight, EPS)

    assert len(seen) == 1
    assert seen[0][0] is x and seen[0][1] is residual
    assert seen[0][2] is weight and seen[0][3] == EPS
    assert torch.equal(summed, torch.full((4, WIDTH), 2.0))
    assert torch.equal(normed, torch.full((4, WIDTH), 4.0))


def test_norm_kernel_is_taken_only_with_the_fused_add_kernel() -> None:
    """The platform kernel is adopted only when the platform ships both, so a
    platform that provides a plain normalization keeps the torch path.
    """

    class _NormOnly:
        def get_rmsnorm(self):
            return "norm"

        def get_add_rmsnorm(self):
            return None

    class _Both:
        def get_rmsnorm(self):
            return "norm"

        def get_add_rmsnorm(self):
            return "add"

    add_kernel, norm_kernel = _resolve_kernels(_NormOnly())
    assert add_kernel is None
    assert norm_kernel is fused_norm._reference_rms_norm

    add_kernel, norm_kernel = _resolve_kernels(_Both())
    assert add_kernel == "add"
    assert norm_kernel is not fused_norm._reference_rms_norm
    assert callable(norm_kernel)


def test_no_affine_gamma_is_ones_and_cached_per_shape() -> None:
    """The stand-in gamma is all ones, which makes it a no-op scale, and it is
    reused for a given (device, width, dtype) so it is allocated once rather
    than per call (allocation inside a captured region is not replayable).
    """
    x = torch.zeros(3, 128)
    other_width = torch.zeros(3, 256)

    gamma = no_affine_gamma(x)
    assert torch.equal(gamma, torch.ones(128))
    assert gamma.device == x.device and gamma.dtype == x.dtype
    assert no_affine_gamma(x) is gamma
    assert no_affine_gamma(other_width) is not gamma
