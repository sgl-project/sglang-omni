# SPDX-License-Identifier: Apache-2.0
"""Radix-cache token ids for ZONOS2 frames.

The scheduler keys the radix tree on one id per frame. A frame's 9 codes (plus
the constant text-pad column) must hash to a single id so a radix match implies
identical audio — and continuing frames fold below the vocab boundary while a
finished request emits a sentinel above it, which is how the scheduler detects
EOS. Pure int64 ops (host-side here; B is small).
"""

from __future__ import annotations

import torch

# Continuing-frame keys fold strictly below this; a generated id at/above it
# finishes the request (Req vocab-boundary check). Also the EOS sentinel.
RADIX_HASH_SPACE = 151643
EOS_SENTINEL = RADIX_HASH_SPACE

_MOD = 2147483647  # 2**31 - 1
_BASE = 1000000007


def poly_row_hash(rows: torch.Tensor) -> torch.Tensor:
    """Order-sensitive polynomial hash of each ``[B, C]`` int row -> ``[B]`` int64."""
    work = rows.to(torch.int64)
    acc = torch.zeros(work.shape[0], dtype=torch.int64, device=work.device)
    for c in range(work.shape[1]):
        acc = torch.remainder(acc * _BASE + torch.remainder(work[:, c], _MOD), _MOD)
    return torch.remainder(acc, RADIX_HASH_SPACE)
