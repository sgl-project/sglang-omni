"""Vendor wrapper for sglang.srt.distributed.

Centralize third-party imports and apply optional monkey patches here.
"""

from __future__ import annotations

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.distributed.utils import get_group_rank_size

# Optional monkey patches (example):
# def tensor_model_parallel_all_reduce(x):
#     return x

__all__ = [
    "get_group_rank_size",
    "tensor_model_parallel_all_reduce",
]
