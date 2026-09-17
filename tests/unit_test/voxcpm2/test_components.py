# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 component regressions: BF16 arithmetic."""

import torch

from sglang_omni.models.voxcpm2.components.minicpm import (
    MiniCPMRMSNorm,
    MiniCPMSiluAndMul,
)


def test_silu_rounds_before_multiplying_the_up_projection():
    gate_up = torch.tensor(
        [[-7.34375, -6.71875, 0.3125, 0.42578125]], dtype=torch.bfloat16
    )
    expected = torch.tensor(
        [[-0.00148773193359375, -0.0034637451171875]], dtype=torch.bfloat16
    )
    torch.testing.assert_close(MiniCPMSiluAndMul()(gate_up), expected, atol=0, rtol=0)


def test_normalization_rounds_before_multiplying_its_weight():
    norm = MiniCPMRMSNorm(4).bfloat16()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.375, 1.375, -3.5, 8.5]))
    values = torch.tensor([[0.25, 0.5, 1.0, 2.0]], dtype=torch.bfloat16)
    expected = torch.tensor(
        [[0.0810546875, 0.59765625, -3.03125, 14.75]], dtype=torch.bfloat16
    )
    torch.testing.assert_close(norm(values), expected, atol=0, rtol=0)
