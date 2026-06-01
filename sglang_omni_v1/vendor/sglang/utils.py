"""Vendor wrapper for sglang.srt.utils.

Centralize third-party imports and apply optional monkey patches here.
"""

from __future__ import annotations

from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    is_cuda,
    is_flashinfer_available,
    is_non_idle_and_non_empty,
    is_npu,
    make_layers,
)

# Optional monkey patches (example):
# def is_cuda() -> bool:
#     return False

__all__ = [
    "LazyValue",
    "add_prefix",
    "is_cuda",
    "is_flashinfer_available",
    "is_non_idle_and_non_empty",
    "is_npu",
    "make_layers",
]
