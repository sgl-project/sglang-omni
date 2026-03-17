"""Vendor wrapper for sglang.srt.models.utils.

Centralize third-party imports and apply monkey patches here.

Patch applied to apply_qk_norm:
  - Skip fused_inplace_qknorm when q_norm.cast_x_before_out_mul is True,
    ensuring HF-compatible RMSNorm cast order for QK normalization.
This patch can be removed once upstream SGLang merges the equivalent change.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sglang.srt.models.utils as _sglang_models_utils
import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)

# ---------------------------------------------------------------------------
# apply_qk_norm monkey-patch: skip fused QK norm when cast_x_before_out_mul
# ---------------------------------------------------------------------------
_orig_apply_qk_norm = _sglang_models_utils.apply_qk_norm


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if getattr(q_norm, "cast_x_before_out_mul", False):
        allow_inplace = False
    return _orig_apply_qk_norm(
        q,
        k,
        q_norm,
        k_norm,
        head_dim,
        alt_stream=alt_stream,
        allow_inplace=allow_inplace,
    )


# Patch the source module so any direct imports also see the change.
_sglang_models_utils.apply_qk_norm = apply_qk_norm

__all__ = [
    "apply_qk_norm",
    "create_fused_set_kv_buffer_arg",
    "enable_fused_set_kv_buffer",
]
