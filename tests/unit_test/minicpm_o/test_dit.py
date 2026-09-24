# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniCPM-o variable-length DiT execution."""

from unittest.mock import patch

import torch

from sglang_omni.models.minicpm_o.components.token2wav.dit import (
    CausalConvBlock,
    DiTBlock,
    FinalLayer,
)


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


def test_packed_adaln_projects_once_per_sequence() -> None:
    torch.manual_seed(0)
    lengths = torch.tensor([2, 3, 1])
    sequence_ids = torch.repeat_interleave(torch.arange(3), lengths)
    total_length = int(lengths.sum())
    per_sequence = torch.randn(3, 8)
    per_frame = per_sequence[sequence_ids]
    x = torch.randn(total_length, 8)
    positions = torch.arange(total_length) + (sequence_ids + 1) * 2
    valid = torch.zeros(total_length + 3 * 2, dtype=torch.bool)
    valid[positions] = True
    cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))

    block = DiTBlock(hidden_size=8, num_heads=2, head_dim=4).eval()
    conditioning_batch_sizes = []
    hook = block.adaLN_modulation.register_forward_pre_hook(
        lambda module, inputs: conditioning_batch_sizes.append(inputs[0].shape[0])
    )
    with patch.object(block.attn, "forward_packed", side_effect=lambda x, *_: x):
        actual = block.forward_packed(
            x, per_sequence, sequence_ids, cu_seqlens, 3, positions, valid
        )
        expected = block.forward_packed(
            x, per_frame, torch.arange(total_length), cu_seqlens, 3, positions, valid
        )
    hook.remove()
    assert conditioning_batch_sizes == [3, total_length]
    torch.testing.assert_close(actual, expected)

    final_layer = FinalLayer(hidden_size=8, out_channels=4).eval()
    actual = final_layer.forward_packed(x, per_sequence, sequence_ids)
    expected = final_layer(x, per_frame)
    torch.testing.assert_close(actual, expected)
