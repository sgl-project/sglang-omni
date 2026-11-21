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


def reference_rms_norm(
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
    else:
        pass
    return gamma


def platform_rms_norm(kernel: RMSNormKernel) -> RMSNormKernel:
    """Adapt a platform kernel to the ``(x, weight, eps)`` call signature.

    The kernels reject ``weight=None``, so a no-affine call reaches them with
    the cached all-ones gamma instead.
    """

    def rms_norm_with_gamma(
        x: torch.Tensor, weight: torch.Tensor | None, eps: float
    ) -> torch.Tensor:
        return kernel(x, no_affine_gamma(x) if weight is None else weight, eps)

    return rms_norm_with_gamma


def plain_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual = x + residual`` then RMSNorm; returns ``(normed, residual)``."""
    residual = x + residual
    return reference_rms_norm(residual, weight, eps), residual


def bind_norm_kernels(
    platform: OmniPlatform,
) -> tuple[RMSNormAddKernel, RMSNormKernel]:
    """Bind both norm call sites once for ``platform``.

    The platform norm kernel is only taken when the fused add kernel exists, so
    the two always come from the same platform; everywhere else both call sites
    stay on ``F.rms_norm`` and no wrapper is added.
    """
    add_kernel = platform.get_add_rmsnorm()
    if add_kernel is None:
        return plain_add_rms_norm, reference_rms_norm
    else:
        return add_kernel, platform_rms_norm(platform.get_rmsnorm())


# Resolved once: both kernels are stateless and the platform never changes
# under a running process.
add_rms_norm, rms_norm = bind_norm_kernels(current_platform)
