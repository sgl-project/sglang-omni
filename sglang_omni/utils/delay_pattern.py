# SPDX-License-Identifier: Apache-2.0
"""Shared delay-pattern transforms for multi-codebook TTS codecs."""

from __future__ import annotations

from typing import Literal

import torch


def reverse_delay_pattern(
    delayed: torch.Tensor, *, on_short: Literal["raise", "empty"] = "raise"
) -> torch.Tensor:
    """Undo a codebook delay pattern: ``[L, N]`` (L >= N) -> ``[L - (N - 1), N]``.

    Column ``c`` is read with a ``c``-row offset so the per-codebook delays line
    back up. ``on_short`` controls the ``L < N`` case where no full data row can
    be recovered: ``"raise"`` (default; Higgs) raises ``ValueError``; ``"empty"``
    (MOSS) returns an empty ``[0, N]`` tensor.
    """
    if delayed.ndim != 2:
        raise ValueError(
            f"delayed must be 2-D [L, N], got shape {tuple(delayed.shape)}"
        )
    length, n = delayed.shape
    rows = length - (n - 1)
    if rows <= 0:
        if on_short == "empty":
            return delayed.new_empty((0, n))
        raise ValueError(
            f"delayed has L={length}, N={n}; need L >= N so at least one data "
            f"row can be recovered."
        )
    out = torch.empty((rows, n), device=delayed.device, dtype=delayed.dtype)
    for c in range(n):
        out[:, c] = delayed[c : c + rows, c]
    return out


__all__ = ["reverse_delay_pattern"]
