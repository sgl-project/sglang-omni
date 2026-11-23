"""Vendor wrapper for sglang.srt.utils.

Centralize third-party imports and apply optional monkey patches here.
"""

from __future__ import annotations

from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_layers
from sglang.srt.models.utils import is_non_idle_and_non_empty
from sglang.srt.utils.common import LazyValue
from sglang.srt.utils.device import is_cuda, is_flashinfer_available, is_musa, is_npu

# Optional monkey patches (example):
# def is_cuda() -> bool:
#     return False

__all__ = [
    "LazyValue",
    "add_prefix",
    "is_cuda",
    "is_flashinfer_available",
    "is_musa",
    "is_non_idle_and_non_empty",
    "is_npu",
    "make_layers",
]
