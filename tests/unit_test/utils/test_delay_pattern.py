# SPDX-License-Identifier: Apache-2.0
"""Contract test for the shared ``reverse_delay_pattern`` util (RFC #661 quick win).

Covers the de-delay round-trip and the two ``on_short`` policies that Higgs and
MOSS rely on (raise vs return-empty). Needs torch; CPU only, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.utils.delay_pattern import reverse_delay_pattern


def _apply_delay(codes: torch.Tensor) -> torch.Tensor:
    """Reference forward delay: ``[T, N]`` -> ``[T + N - 1, N]`` (column c offset by c)."""
    t, n = codes.shape
    out = torch.full((t + n - 1, n), -1, dtype=codes.dtype)
    for c in range(n):
        out[c : c + t, c] = codes[:, c]
    return out


def test_reverse_round_trips_the_forward_delay() -> None:
    codes = torch.arange(12).reshape(4, 3)  # T=4, N=3
    recovered = reverse_delay_pattern(_apply_delay(codes))
    assert torch.equal(recovered, codes)


def test_short_input_raises_by_default() -> None:
    with pytest.raises(ValueError):
        reverse_delay_pattern(torch.zeros(2, 3, dtype=torch.long))  # L=2 < N=3


def test_short_input_returns_empty_when_requested() -> None:
    out = reverse_delay_pattern(torch.zeros(2, 3, dtype=torch.long), on_short="empty")
    assert out.shape == (0, 3)


def test_non_2d_input_raises() -> None:
    with pytest.raises(ValueError):
        reverse_delay_pattern(torch.zeros(3, dtype=torch.long))
