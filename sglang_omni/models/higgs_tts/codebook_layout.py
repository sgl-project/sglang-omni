# SPDX-License-Identifier: Apache-2.0
"""Higgs codec-codebook layout helpers."""

from __future__ import annotations

from typing import Any

import torch

# Codec-vocab specials (inside the [N*V] codebook space, NOT the text vocab).
BOC_ID = 1024
EOC_ID = 1025


def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    """``[T, N]`` raw codes -> ``[T + N - 1, N]`` delayed, BOC/EOC padded."""
    if codes_TN.ndim != 2:
        raise ValueError(
            f"codes_TN must be 2-D [T, N], got shape {tuple(codes_TN.shape)}"
        )
    T, N = codes_TN.shape
    out = torch.full(
        (T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype
    )
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    """``[L, N]`` delayed (L >= N) -> ``[L - (N - 1), N]`` raw codes."""
    if delayed_LN.ndim != 2:
        raise ValueError(
            f"delayed_LN must be 2-D [L, N], got shape {tuple(delayed_LN.shape)}"
        )
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        raise ValueError(
            f"delayed_LN has L={L}, N={N}; need L >= N so at least one "
            f"data row can be recovered."
        )
    out = torch.empty((T, N), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(N):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


def to_codes_TN(raw: Any, num_codebooks: int) -> torch.Tensor | None:
    """Coerce client-supplied ``reference_codes`` to a ``[T, N]`` int64 tensor."""
    if raw is None:
        return None
    t = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw)
    if t.numel() == 0:
        return None
    if t.ndim != 2 or t.shape[1] != num_codebooks:
        raise ValueError(
            f"reference_codes must have shape [T, {num_codebooks}], got {tuple(t.shape)}"
        )
    return t.to(torch.long)
