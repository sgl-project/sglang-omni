# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniCPM-o variable-length DiT execution."""

import torch

from sglang_omni.models.minicpm_o.components.token2wav.dit import CausalConvBlock


def test_packed_causal_conv_preserves_sequence_boundaries() -> None:
    torch.manual_seed(0)
    block = CausalConvBlock(4, 4).eval()
    rows = [torch.randn(length, 4) for length in (3, 5, 2)]

    expected = torch.cat([block(row.unsqueeze(0)).squeeze(0) for row in rows])
    lengths = torch.tensor([len(row) for row in rows])
    total_length = int(lengths.sum())
    sequence_ids = torch.repeat_interleave(torch.arange(len(rows)), lengths)
    positions = torch.arange(total_length) + (sequence_ids + 1) * 2
    valid = torch.zeros(total_length + len(rows) * 2, dtype=torch.bool)
    valid[positions] = True
    actual = block.forward_packed(torch.cat(rows), positions, valid)

    torch.testing.assert_close(actual, expected)
