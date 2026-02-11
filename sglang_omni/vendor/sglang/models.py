"""Vendor wrapper for sglang.srt.models.utils.

Centralize third-party imports and apply optional monkey patches here.
"""

from __future__ import annotations

from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)

# Optional monkey patches (example):
# def apply_qk_norm(*args, **kwargs):
#     return args[0]

__all__ = [
    "apply_qk_norm",
    "create_fused_set_kv_buffer_arg",
    "enable_fused_set_kv_buffer",
]
