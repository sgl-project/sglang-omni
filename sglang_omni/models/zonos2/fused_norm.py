# SPDX-License-Identifier: Apache-2.0
"""RMSNorm calls dispatched to the platform's kernels.

``F.rms_norm`` decomposes into several elementwise kernels per call, which on
ZONOS2 decode adds up to ~140 launches per step; the platform kernels fold that
into one, and the fused add variant also returns the residual sum so the
separate add disappears.

The platform kernel replaces ``F.rms_norm`` only where the platform also
provides the fused add variant, i.e. where it ships both. On the remaining
platforms the plain normalization is left untouched, so this module adds no
kernel change there.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.interface import (
    OmniPlatform,
    RMSNormAddKernel,
    RMSNormKernel,
)


def _reference_rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    """The plain normalization, kept for platforms without fused kernels."""
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


# Gamma stand-in for the no-affine case, which the kernels reject. Cached per
# (device, width, dtype) so the tensor exists before graph capture instead of
# being allocated inside the captured region.
_NO_AFFINE_GAMMA: dict[tuple[torch.device, int, torch.dtype], torch.Tensor] = {}


def no_affine_gamma(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.shape[-1], x.dtype)
    gamma = _NO_AFFINE_GAMMA.get(key)
    if gamma is None:
        gamma = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        _NO_AFFINE_GAMMA[key] = gamma
    return gamma


def _platform_rms_norm(kernel: RMSNormKernel) -> RMSNormKernel:
    """Adapt a platform kernel to the ``(x, weight, eps)`` call signature.

    The kernels reject ``weight=None``, so a no-affine call reaches them with
    the cached all-ones gamma instead.
    """

    def rms_norm_with_gamma(
        x: torch.Tensor, weight: torch.Tensor | None, eps: float
    ) -> torch.Tensor:
        return kernel(x, no_affine_gamma(x) if weight is None else weight, eps)

    return rms_norm_with_gamma


def _resolve_kernels(
    platform: OmniPlatform,
) -> tuple[RMSNormAddKernel | None, RMSNormKernel]:
    """Return ``(add_kernel, norm_kernel)`` for ``platform``.

    The platform norm kernel is only taken when the fused add kernel exists, so
    the two always come from the same platform; elsewhere the plain
    normalization is the norm kernel.
    """
    add_kernel = platform.get_add_rmsnorm()
    if add_kernel is None:
        return None, _reference_rms_norm
    return add_kernel, _platform_rms_norm(platform.get_rmsnorm())


# Resolved once: both kernels are stateless and the platform never changes
# under a running process.
_ADD_RMSNORM_KERNEL, _RMSNORM_KERNEL = _resolve_kernels(current_platform)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim; ``weight=None`` applies no affine scale."""
    return _RMSNORM_KERNEL(x, weight, eps)


def add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual = x + residual`` then RMSNorm; returns ``(normed, residual)``."""
    if _ADD_RMSNORM_KERNEL is not None:
        return _ADD_RMSNORM_KERNEL(x, residual, weight, eps)
    residual = x + residual
    return rms_norm(residual, weight, eps), residual
