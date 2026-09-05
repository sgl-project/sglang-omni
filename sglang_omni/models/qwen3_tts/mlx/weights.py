# SPDX-License-Identifier: Apache-2.0
"""Weight-layout helpers for the MLX Qwen3-TTS ports."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# PyTorch and MLX disagree on convolution weight layout, and the two
# convolution kinds disagree with each other:
#   Conv1d           torch [out, in, kernel]  -> mlx [out, kernel, in]
#   ConvTranspose1d  torch [in, out, kernel]  -> mlx [out, kernel, in]
_CONV_PERMUTATIONS: tuple[tuple[type, tuple[int, int, int]], ...] = (
    # ConvTranspose1d first: it subclasses nothing shared, but keep the more
    # specific check ahead of Conv1d in case that changes upstream.
    (nn.ConvTranspose1d, (1, 2, 0)),
    (nn.Conv1d, (0, 2, 1)),
)


def align_conv_weights(
    weights: dict[str, mx.array],
    model: Any,
) -> dict[str, mx.array]:
    """Permute checkpoint convolution weights into MLX layout.

    The permutation is chosen from each module's *type*, not from the shape,
    because a transposed convolution with equal input and output channels has
    the same shape under either permutation and cannot be told apart
    numerically. A tensor already in MLX layout is left alone, so re-loading a
    converted checkpoint is a no-op.
    """
    expected = {
        key: value.shape
        for key, value in tree_flatten(model.parameters())
        if value.ndim == 3
    }

    permutation_by_key: dict[str, tuple[int, int, int]] = {}
    for path, module in model.named_modules():
        for module_type, permutation in _CONV_PERMUTATIONS:
            if isinstance(module, module_type):
                key = f"{path}.weight" if path else "weight"
                permutation_by_key[key] = permutation
                break

    aligned = dict(weights)
    for key, permutation in permutation_by_key.items():
        value = aligned.get(key)
        target = expected.get(key)
        if value is None or target is None or value.ndim != 3:
            continue
        if value.shape == target:
            continue
        permuted = mx.transpose(value, permutation)
        if permuted.shape == target:
            aligned[key] = permuted
    return aligned
