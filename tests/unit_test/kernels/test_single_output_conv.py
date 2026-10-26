# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.kernels.single_output_conv import clamped_single_output_conv
from sglang_omni.utils.channels_last_conv import channels_last_weight


def make_conv(channels: int, dtype: torch.dtype) -> torch.nn.Conv1d:
    conv = torch.nn.Conv1d(channels, 1, 7).to(device="cuda", dtype=dtype)
    conv.weight.data = channels_last_weight(conv)
    return conv


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels", [8, 96])
def test_single_output_conv_is_the_clamped_conv_rounded_once(
    dtype: torch.dtype, channels: int
) -> None:
    torch.manual_seed(0)
    conv = make_conv(channels, dtype)
    padded = torch.randn(3, 300 + 6, channels, device="cuda", dtype=dtype)

    with torch.inference_mode():
        actual = clamped_single_output_conv(padded, conv, 300)
        expected = (
            F.conv1d(
                padded.double().transpose(1, 2),
                conv.weight.double(),
                conv.bias.double(),
            )
            .clamp(min=-1, max=1)
            .to(dtype)
        )

    assert actual.shape == (3, 1, 300) and actual.is_contiguous()
    assert (expected.abs() == 1).any() and (expected.abs() < 1).any()
    torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_single_output_conv_gives_a_position_the_same_bits_in_any_split() -> None:
    torch.manual_seed(0)
    conv = make_conv(96, torch.bfloat16)
    padded = torch.randn(3, 1000 + 6, 96, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        whole = clamped_single_output_conv(padded, conv, 1000)
        parts = torch.cat(
            [
                clamped_single_output_conv(padded[:, : 417 + 6], conv, 417),
                clamped_single_output_conv(padded[:, 417:], conv, 583),
            ],
            dim=-1,
        )
        row = clamped_single_output_conv(padded[1:2], conv, 1000)

    assert torch.equal(parts, whole)
    assert torch.equal(row, whole[1:2])
