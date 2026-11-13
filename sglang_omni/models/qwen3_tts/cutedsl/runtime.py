# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Backend selection and supported geometry without importing CUTLASS."""

import os
from functools import cache

import torch

QTTS_BACKEND = os.environ.get("SGLANG_OMNI_QTTS_BACKEND", "default")
if QTTS_BACKEND not in ("default", "cutedsl"):
    raise ValueError("SGLANG_OMNI_QTTS_BACKEND must be default or cutedsl")
else:
    pass

MAX_DECODE_ROWS = 32
QUERY_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
QUERY_WIDTH = QUERY_HEADS * HEAD_DIM
QKV_WIDTH = (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM


@cache
def device_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    return f"sm_{capability[0]}{capability[1]}a"


def supports_auxiliary(tensor: torch.Tensor) -> bool:
    """Enable BF16 auxiliary kernels on Hopper and datacenter Blackwell."""
    return (
        tensor.is_cuda
        and tensor.dtype == torch.bfloat16
        and device_arch(tensor.device) in ("sm_90a", "sm_100a", "sm_103a")
    )
