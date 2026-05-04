"""Vendor wrapper for sglang.srt.distributed.

Centralize third-party imports and apply optional monkey patches here.
"""

from __future__ import annotations

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)

# Optional monkey patches (example):
# def tensor_model_parallel_all_reduce(x):
#     return x

__all__ = [
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "tensor_model_parallel_all_reduce",
]
