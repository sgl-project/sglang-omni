# SPDX-License-Identifier: Apache-2.0
"""Shared reverse (de-delay) transform: ``[L, N]`` delayed -> ``[L-(N-1), N]`` raw codes.

Forward (delay) direction is model-specific and stays in each model.
"""

from __future__ import annotations

import torch


def reverse_delay_pattern(
    delayed: torch.Tensor, *, allow_short: bool = False
) -> torch.Tensor:
    """``[L, N]`` delayed -> ``[L-(N-1), N]`` raw. When L<N: allow_short raises (False) / returns [0,N] (True)."""
    if delayed.ndim != 2:
        raise ValueError(
            f"delayed codes must be 2-D [L, N], got shape {tuple(delayed.shape)}"
        )
    length, num_codebooks = delayed.shape
    rows = length - (num_codebooks - 1)
    if rows <= 0:
        if allow_short:
            return delayed.new_empty((0, num_codebooks))
        raise ValueError(
            f"delayed has L={length}, N={num_codebooks}; need L >= N so at "
            f"least one data row can be recovered."
        )
    out = torch.empty((rows, num_codebooks), device=delayed.device, dtype=delayed.dtype)
    for c in range(num_codebooks):
        out[:, c] = delayed[c : c + rows, c]
    return out
