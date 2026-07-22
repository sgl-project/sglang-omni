# SPDX-License-Identifier: Apache-2.0
"""Codec-content geometry is deliberately separate from the RL action mask."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.utils import (
    BOC_ID,
    EOC_ID,
    apply_delay_pattern,
    delay_pattern_codec_content_mask,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("t_raw,n", [(1, 8), (3, 4), (10, 8), (5, 2), (1, 1)])
def test_content_mask_inverts_complete_delay_pattern(t_raw: int, n: int):
    raw = torch.randint(0, 1024, (t_raw, n), device=DEVICE)
    delayed = apply_delay_pattern(raw)

    mask = delay_pattern_codec_content_mask(delayed)

    assert torch.equal(mask, (delayed != BOC_ID) & (delayed != EOC_ID))
    assert int(mask.sum()) == t_raw * n


def test_content_mask_depends_on_shape_not_special_token_values():
    t_raw, n = 5, 4
    delayed = apply_delay_pattern(torch.randint(0, 1024, (t_raw, n), device=DEVICE))
    delayed[1, 1] = EOC_ID
    delayed[3, 0] = BOC_ID

    mask = delay_pattern_codec_content_mask(delayed)
    rows = torch.arange(delayed.shape[0], device=DEVICE).unsqueeze(1)
    channels = torch.arange(n, device=DEVICE).unsqueeze(0)
    assert torch.equal(mask, (channels <= rows) & (rows < channels + t_raw))


def test_length_truncation_only_marks_fully_dedelayable_rows():
    length, n = 7, 4
    delayed = torch.randint(0, 1024, (length, n), device=DEVICE)
    t_raw = length - (n - 1)

    mask = delay_pattern_codec_content_mask(delayed)

    rows = torch.arange(length, device=DEVICE).unsqueeze(1)
    channels = torch.arange(n, device=DEVICE).unsqueeze(0)
    assert torch.equal(mask, (channels <= rows) & (rows < channels + t_raw))
