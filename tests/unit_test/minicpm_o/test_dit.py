# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniCPM-o variable-length DiT execution."""

from unittest.mock import patch

import pytest
import torch

from sglang_omni.models.minicpm_o.components.token2wav.dit import (
    CausalConvBlock,
    DiT,
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


def test_compile_wraps_packed_block_method_only() -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=8)
    calls: list[int] = []

    def compiled_forward(
        block: DiTBlock,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        sequence_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_length: int,
        conv_positions: torch.Tensor,
        conv_valid: torch.Tensor,
    ) -> torch.Tensor:
        calls.append(x.shape[0])
        return x + 1

    with patch("torch.compile", return_value=compiled_forward) as compile_packed:
        dit.enable_compiled_packed_blocks()
    compile_packed.assert_called_once_with(
        DiTBlock.forward_packed,
        dynamic=True,
        fullgraph=True,
        options={"triton.cudagraphs": False},
    )
    dit.warmup_compiled_packed_blocks()
    assert len(calls) == 2 and calls[0] != calls[1]
    calls.clear()
    block = dit.blocks[0]
    x = torch.ones(3, 8)
    actual = block.forward_packed(
        x,
        torch.zeros(1, 8),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([0, 3], dtype=torch.int32),
        3,
        torch.arange(3),
        torch.ones(3, dtype=torch.bool),
    )
    torch.testing.assert_close(actual, x + 1)
    dense = block.forward(
        x.unsqueeze(0), torch.zeros(1, 1, 8), torch.ones(1, 3, dtype=torch.bool)
    )
    assert dense.shape == (1, 3, 8)
    assert calls == [3]


def test_compile_materialization_failure_propagates() -> None:
    dit = DiT(in_channels=16, out_channels=4, depth=1, hidden_size=8)
    with patch.object(
        dit,
        "forward_packed",
        side_effect=RuntimeError("compiler materialization failed"),
    ):
        with pytest.raises(RuntimeError, match="compiler materialization failed"):
            dit.warmup_compiled_packed_blocks()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_packed_block_matches_eager_with_changed_lengths() -> None:
    torch.manual_seed(31)
    dit = (
        DiT(
            in_channels=16,
            out_channels=4,
            depth=1,
            num_heads=2,
            head_dim=16,
            hidden_size=32,
        )
        .cuda()
        .eval()
    )
    block = dit.blocks[0]
    torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.01)
    dit.enable_compiled_packed_blocks()

    for lengths in ((6, 4), (9, 5), (6, 4)):
        total = sum(lengths)
        x = torch.randn(total, 32, device="cuda", dtype=torch.bfloat16)
        conditioning = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
        sequence_ids = torch.repeat_interleave(
            torch.arange(2, device="cuda"), torch.tensor(lengths, device="cuda")
        )
        cu_seqlens = torch.tensor(
            [0, lengths[0], total], device="cuda", dtype=torch.int32
        )
        conv_positions = torch.arange(total, device="cuda") + (sequence_ids + 1) * 2
        conv_valid = torch.zeros(total + 4, device="cuda", dtype=torch.bool)
        conv_valid[conv_positions] = True
        arguments = (
            x,
            conditioning,
            sequence_ids,
            cu_seqlens,
            max(lengths),
            conv_positions,
            conv_valid,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = DiTBlock.forward_packed(block, *arguments)
            actual = block.forward_packed(*arguments)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
