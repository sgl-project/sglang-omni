# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.platforms import current_platform
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
    channels_last_folded_weight,
    channels_last_weight,
)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="channels-last convs run on NVIDIA CUDA only"
)
@pytest.mark.parametrize("length", [1, 5, 16])
@pytest.mark.parametrize("dilation,groups", [(1, 1), (3, 1), (9, 1), (1, 2), (3, 2)])
def test_channels_last_conv1d_matches_the_causal_conv1d(
    monkeypatch: pytest.MonkeyPatch, length: int, dilation: int, groups: int
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(length * 10 + dilation)
    conv = torch.nn.Conv1d(4, 6, 7, dilation=dilation, groups=groups).cuda()
    left_context = (7 - 1) * dilation
    inputs = torch.randn(2, length, 4, device="cuda")
    expected = F.conv1d(
        F.pad(inputs.transpose(1, 2), (left_context, 0)),
        conv.weight,
        conv.bias,
        dilation=dilation,
        groups=groups,
    ).transpose(1, 2)

    padded = F.pad(inputs, (0, 0, left_context, (-length) % dilation))
    actual = channels_last_conv1d(
        padded, conv, channels_last_weight(conv), conv.bias, length
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="channels-last convs run on NVIDIA CUDA only"
)
@pytest.mark.parametrize("length", [60, 61])
@pytest.mark.parametrize("stride,kernel_size", [(3, 6), (15, 30)])
def test_channels_last_conv1d_matches_the_strided_causal_conv1d(
    monkeypatch: pytest.MonkeyPatch, length: int, stride: int, kernel_size: int
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(stride)
    conv = torch.nn.Conv1d(4, 6, kernel_size, stride=stride).cuda()
    inputs = torch.randn(2, length, 4, device="cuda")
    expected = F.conv1d(
        F.pad(inputs.transpose(1, 2), (stride - 1, 0)),
        conv.weight,
        conv.bias,
        stride=stride,
    ).transpose(1, 2)

    outputs = (stride - 1 + length - kernel_size) // stride + 1
    padded = F.pad(inputs, (0, 0, stride - 1, (-(stride - 1 + length)) % stride))
    actual = channels_last_conv1d(
        padded,
        conv,
        channels_last_folded_weight(conv.weight.data, stride),
        conv.bias,
        outputs,
    )

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="channels-last convs run on NVIDIA CUDA only"
)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("stride,padding,output_padding", [(2, 0, 0), (4, 1, 1)])
def test_channels_last_conv_transpose1d_matches_the_conv_transpose1d(
    monkeypatch: pytest.MonkeyPatch,
    has_bias: bool,
    stride: int,
    padding: int,
    output_padding: int,
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(stride)
    conv = torch.nn.ConvTranspose1d(
        4, 6, 8, stride=stride, padding=padding, output_padding=output_padding
    ).cuda()
    bias = conv.bias if has_bias else None
    inputs = torch.randn(2, 9, 4, device="cuda")
    expected = F.conv_transpose1d(
        inputs.transpose(1, 2),
        conv.weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    ).transpose(1, 2)

    actual = channels_last_conv_transpose1d(
        inputs, conv, channels_last_weight(conv), bias
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "conv",
    [
        torch.nn.Conv1d(4, 6, 7),
    ],
)
def test_channels_last_weight_has_channels_last_strides_and_leaves_the_module(
    conv: torch.nn.Conv1d | torch.nn.ConvTranspose1d,
) -> None:
    weight = channels_last_weight(conv)

    assert torch.equal(weight, conv.weight)
    assert weight.transpose(1, 2).is_contiguous()
    assert conv.weight.is_contiguous()
