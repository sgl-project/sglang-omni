# SPDX-License-Identifier: Apache-2.0
"""Tensor conversion at the native MLX adapter boundary."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    tensor = tensor.detach().cpu()
    if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.float()
    return mx.array(tensor.numpy())


def mlx_to_torch(array: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(array.astype(mx.float32))))
