# SPDX-License-Identifier: Apache-2.0
"""LFM3 short-conv decode against the MambaPool slot layout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

backbone = pytest.importorskip("sglang_omni.models.lfm3_audio.backbone")


def test_conv_decode_padding_rows_write_only_the_dummy_slot() -> None:
    # Slot 0 is the pool's dummy; slots 1..4 belong to requests. A decode graph pads its batch with slot -1.
    torch.manual_seed(0)
    channels, kernel = 3, 7
    conv = SimpleNamespace(conv_weight=torch.randn(channels, kernel), conv_bias=None)
    conv_state = torch.randn(5, channels, kernel - 1)
    live = conv_state.clone()
    inputs = torch.randn(4, channels)
    slots = torch.tensor([2, 4, -1, -1])

    outputs = backbone.Lfm3ShortConv.conv_decode(conv, inputs, conv_state, slots)

    for row, slot in enumerate([2, 4]):
        window = torch.cat([live[slot], inputs[row, :, None]], dim=-1)
        torch.testing.assert_close(outputs[row], (window * conv.conv_weight).sum(-1))
        torch.testing.assert_close(conv_state[slot], window[:, 1:])
    for slot in (1, 3):
        torch.testing.assert_close(conv_state[slot], live[slot])
