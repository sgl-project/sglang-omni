# SPDX-License-Identifier: Apache-2.0
"""The fused positional conv against the module's grouped Conv1d and Mish."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.causal_conv import (
    GROUP_CONV_KERNEL_CHANNELS,
    FusedConvPositionEmbedding,
    group_conv_mish,
    pack_group_conv_weight,
)

pytestmark = pytest.mark.accelerator

GROUPS = 16
TAPS = 31


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual.double() - expected)
        / torch.linalg.vector_norm(expected)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_channels", GROUP_CONV_KERNEL_CHANNELS)
@pytest.mark.parametrize("frames", [31, 32, 95, 1000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_group_conv_mish_is_as_close_to_float64_as_the_module(
    group_channels: int, frames: int, dtype: torch.dtype
) -> None:
    channels = GROUPS * group_channels
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(channels, channels, TAPS, groups=GROUPS).cuda().to(dtype)
    with torch.no_grad():
        conv.bias.normal_(0, 0.5)
    x = torch.randn(2, frames, channels, device="cuda").to(dtype)

    with torch.inference_mode():
        fused = group_conv_mish(x, pack_group_conv_weight(conv), conv.bias)
        module = F.mish(conv(x.transpose(1, 2))).transpose(1, 2)
        float64 = F.mish(
            F.conv1d(
                x.double().transpose(1, 2),
                conv.weight.double(),
                conv.bias.double(),
                groups=GROUPS,
            )
        ).transpose(1, 2)

    assert fused.shape == (2, frames - TAPS + 1, channels) and fused.dtype == dtype
    assert relative_error(fused, float64) <= relative_error(module, float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_fused_position_embedding_is_as_close_to_float64_as_the_module() -> None:
    cosyvoice_modules = pytest.importorskip("cosyvoice.flow.DiT.modules")
    torch.manual_seed(0)
    original = cosyvoice_modules.CausalConvPositionEmbedding(256).cuda().eval()
    float64 = copy.deepcopy(original).double()
    original.to(torch.bfloat16)
    fused = FusedConvPositionEmbedding(original)
    x = torch.randn(2, 77, 256, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        fused_output = fused(x)
        module_output = original(x)
        reference = float64(x.double())

    assert fused_output.shape == module_output.shape
    assert relative_error(fused_output, reference) <= relative_error(
        module_output, reference
    )
