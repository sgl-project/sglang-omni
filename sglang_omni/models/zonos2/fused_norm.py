# SPDX-License-Identifier: Apache-2.0
"""RMSNorm dispatch: one fused kernel per call where the device provides it.

The Ascend decomposition of ``F.rms_norm`` emits 4-5 elementwise kernels per
call (pow, reduce-mean, rsqrt, mul, cast), which on ZONOS2 decode adds up to
~140 kernels per step; ``npu_rms_norm`` / ``npu_add_rms_norm`` fold them into
one and also return the residual sum, so no separate add is needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:  # only NPU stacks install torch_npu
    torch_npu = None

# Gamma stand-in for the no-affine case, which the fused op rejects. Cached per
# (device, width, dtype) so the tensor exists before graph capture instead of
# being allocated inside the captured region.
_NO_AFFINE_GAMMA: dict[tuple[torch.device, int, torch.dtype], torch.Tensor] = {}


def is_fused(x: torch.Tensor) -> bool:
    return torch_npu is not None and x.device.type == "npu"


def no_affine_gamma(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.shape[-1], x.dtype)
    gamma = _NO_AFFINE_GAMMA.get(key)
    if gamma is None:
        gamma = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        _NO_AFFINE_GAMMA[key] = gamma
    else:
        pass
    return gamma


def rms_norm(
    x: torch.Tensor, weight: Optional[torch.Tensor], eps: float
) -> torch.Tensor:
    """RMSNorm over the last dim; ``weight=None`` applies no affine scale."""
    if is_fused(x):
        return torch_npu.npu_rms_norm(
            x, no_affine_gamma(x) if weight is None else weight, eps
        )[0]
    else:
        pass
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual = x + residual`` then RMSNorm; returns ``(normed, residual)``."""
    if is_fused(x):
        normed, _, residual = torch_npu.npu_add_rms_norm(x, residual, weight, eps)
        return normed, residual
    else:
        pass
    residual = x + residual
    return F.rms_norm(residual, (residual.shape[-1],), weight, eps), residual
