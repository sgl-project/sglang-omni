# SPDX-License-Identifier: Apache-2.0
"""Shared delay-pattern (de-delay) transform for multi-codebook audio codes.

The *reverse* delay transform is the only shareable direction: it maps a
delayed ``[L, N]`` layout (codebook ``c`` shifted by ``c`` steps) back to the
raw ``[L - (N - 1), N]`` codes. Higgs and MOSS-TTS implemented the same
column-recovery loop; this is the single shared copy. The *forward* direction
is model-specific (Higgs pads with its codec specials; MOSS builds the delay
incrementally during AR decode) and stays in each model.
"""

from __future__ import annotations

import torch


def reverse_delay_pattern(
    delayed: torch.Tensor, *, allow_short: bool = False
) -> torch.Tensor:
    """``[L, N]`` delayed (``L >= N``) → ``[L - (N - 1), N]`` raw codes.

    ``allow_short=False`` (Higgs): raise ``ValueError`` when ``L < N``.
    ``allow_short=True`` (MOSS): return an empty ``[0, N]`` tensor when ``L < N``.
    """
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
