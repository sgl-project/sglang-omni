# SPDX-License-Identifier: Apache-2.0
"""LFM3 short-conv decode against the MambaPool slot layout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

short_conv = pytest.importorskip("sglang_omni.models.lfm3_audio.short_conv")


def test_conv_decode_padding_rows_write_only_the_dummy_slot() -> None:
    # Slot 0 is the pool's dummy; slots 1..4 belong to requests. A decode graph pads its batch with slot -1.
    torch.manual_seed(0)
    channels, kernel = 3, 7
    conv = SimpleNamespace(conv_weight=torch.randn(channels, kernel), conv_bias=None)
    conv_state = torch.randn(5, channels, kernel - 1)
    live = conv_state.clone()
    inputs = torch.randn(4, channels)
    slots = torch.tensor([2, 4, -1, -1])

    outputs = short_conv.Lfm3ShortConv.conv_decode(conv, inputs, conv_state, slots)

    for row, slot in enumerate([2, 4]):
        window = torch.cat([live[slot], inputs[row, :, None]], dim=-1)
        torch.testing.assert_close(outputs[row], (window * conv.conv_weight).sum(-1))
        torch.testing.assert_close(conv_state[slot], window[:, 1:])
    for slot in (1, 3):
        torch.testing.assert_close(conv_state[slot], live[slot])


def reference_conv_prefill(
    conv: SimpleNamespace,
    inputs: torch.Tensor,
    conv_state: torch.Tensor,
    slots: list[int],
    starts: list[int],
    continues: list[bool],
) -> torch.Tensor:
    """One request at a time: its cached state (or zeros) followed by its inputs."""
    kernel = conv.conv_weight.shape[1]
    outputs = []
    for row, slot in enumerate(slots):
        start, end = starts[row], starts[row + 1]
        if start == end:
            continue
        else:
            pass
        state = (
            conv_state[slot].clone()
            if continues[row]
            else torch.zeros(inputs.shape[1], kernel - 1)
        )
        window = torch.cat([state, inputs[start:end].T], dim=-1)
        outputs.append(
            torch.nn.functional.conv1d(
                window[None],
                conv.conv_weight[:, None],
                conv.conv_bias,
                groups=inputs.shape[1],
            )[0]
        )
        conv_state[slot] = window[:, -(kernel - 1) :]
    return torch.cat(outputs, dim=-1).T


def test_conv_prefill_matches_one_request_at_a_time() -> None:
    torch.manual_seed(0)
    channels, kernel = 4, 7
    conv = SimpleNamespace(
        conv_weight=torch.randn(channels, kernel), conv_bias=torch.randn(channels)
    )
    starts = [0, 5, 5, 6, 15]  # a prompt, an empty segment, one token, a prompt
    slots = [3, 1, 6, 2]
    continues = [False, True, True, False]
    inputs = torch.randn(starts[-1], channels)
    conv_state = torch.randn(7, channels, kernel - 1)
    expected_state = conv_state.clone()
    expected = reference_conv_prefill(
        conv, inputs, expected_state, slots, starts, continues
    )

    outputs = short_conv.Lfm3ShortConv.conv_prefill(
        SimpleNamespace(conv_kernel=kernel, **vars(conv)),
        inputs,
        conv_state,
        torch.tensor(slots),
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(continues),
    )

    torch.testing.assert_close(outputs, expected)
    torch.testing.assert_close(conv_state, expected_state)
