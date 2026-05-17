# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.quantization import (
    convert_fp8_weight_scale_inv_for_sglang,
)


@pytest.mark.parametrize(
    "target_name",
    [
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.w13_weight_scale_inv",
        "model.layers.0.mlp.experts.w2_weight_scale_inv",
    ],
)
def test_convert_fp8_weight_scale_inv_inverts_large_checkpoint_scales(
    target_name: str,
) -> None:
    source = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32)

    converted = convert_fp8_weight_scale_inv_for_sglang(
        target_name,
        source,
    )

    assert torch.allclose(
        converted,
        torch.tensor([[0.5, 0.25], [0.125, 0.0625]], dtype=torch.float32),
    )
    assert torch.equal(source, torch.tensor([[2.0, 4.0], [8.0, 16.0]]))


def test_convert_fp8_weight_scale_inv_inverts_by_loader_contract_not_magnitude() -> (
    None
):
    checkpoint_scale_inv = torch.tensor([0.125, 0.25], dtype=torch.float32)

    converted = convert_fp8_weight_scale_inv_for_sglang(
        "linear.weight_scale_inv",
        checkpoint_scale_inv,
    )

    assert torch.allclose(converted, torch.tensor([8.0, 4.0], dtype=torch.float32))


def test_convert_fp8_weight_scale_inv_leaves_non_scale_weight() -> None:
    weight = torch.tensor([2.0, 4.0], dtype=torch.float32)

    assert convert_fp8_weight_scale_inv_for_sglang("linear.weight", weight) is weight


@pytest.mark.parametrize(
    "scale",
    [
        torch.tensor([], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([float("inf")], dtype=torch.float32),
        torch.tensor([float("nan")], dtype=torch.float32),
    ],
)
def test_convert_fp8_weight_scale_inv_rejects_invalid_scale(
    scale: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        convert_fp8_weight_scale_inv_for_sglang("linear.weight_scale_inv", scale)


def test_convert_fp8_weight_scale_inv_rejects_non_floating_scale() -> None:
    with pytest.raises(TypeError):
        convert_fp8_weight_scale_inv_for_sglang(
            "linear.weight_scale_inv",
            torch.tensor([2, 4], dtype=torch.int32),
        )
