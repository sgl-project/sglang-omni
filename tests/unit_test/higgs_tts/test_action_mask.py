# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`delay_pattern_action_mask` (RL trainable-action mask).

The mask selects exactly the ``T x N`` real-audio parallelogram of a delayed code
matrix -- the inverse geometry of :func:`apply_delay_pattern`. CPU-only.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.utils import (
    BOC_ID,
    EOC_ID,
    apply_delay_pattern,
    delay_pattern_action_mask,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("t_raw,n", [(1, 8), (3, 4), (10, 8), (5, 2), (1, 1), (20, 3)])
def test_mask_is_inverse_of_apply_delay_pattern(t_raw: int, n: int):
    """Mask == the non-BOC/EOC cells of a real delayed matrix == ``c<=r<c+T``."""
    torch.manual_seed(t_raw * 100 + n)
    raw_TN = torch.randint(0, 1024, (t_raw, n), device=DEVICE)  # real codes only
    delayed = apply_delay_pattern(raw_TN)
    assert delayed.shape == (t_raw + n - 1, n)

    mask = delay_pattern_action_mask(delayed)

    is_real = (delayed != BOC_ID) & (delayed != EOC_ID)
    assert torch.equal(mask, is_real)
    assert int(mask.sum().item()) == t_raw * n


def test_T_recovered_from_cb0_eoc_only():
    """``T`` comes from codebook 0's EOC, not an EOC appearing in another cb."""
    n, t_raw = 4, 5
    delayed = apply_delay_pattern(torch.randint(0, 1024, (t_raw, n), device=DEVICE))
    delayed[1, 2] = EOC_ID  # spurious early EOC in cb2; cb0's EOC must still win

    mask = delay_pattern_action_mask(delayed)

    r = torch.arange(delayed.shape[0], device=DEVICE).unsqueeze(1)
    c = torch.arange(n, device=DEVICE).unsqueeze(0)
    assert torch.equal(mask, (c <= r) & (r < c + t_raw))


def test_no_cb0_eoc_means_T_equals_L():
    """Length-truncated generation (cb0 never emits EOC) ⇒ ``T = L``."""
    n, L = 3, 5
    delayed = torch.randint(0, 1024, (L, n), device=DEVICE)
    for c in range(n):
        delayed[:c, c] = BOC_ID  # leading delay triangle, no EOC anywhere

    mask = delay_pattern_action_mask(delayed)

    r = torch.arange(L, device=DEVICE).unsqueeze(1)
    c = torch.arange(n, device=DEVICE).unsqueeze(0)
    assert torch.equal(mask, c <= r)  # r < c + L always holds
