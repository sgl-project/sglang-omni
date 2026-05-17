# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.quantization import (
    convert_fp8_weight_scale_inv_for_sglang,
)


def test_convert_fp8_weight_scale_inv_inverts_large_block_scales() -> None:
    source = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32)

    converted = convert_fp8_weight_scale_inv_for_sglang(
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv",
        source,
    )

    assert torch.allclose(
        converted,
        torch.tensor([[0.5, 0.25], [0.125, 0.0625]], dtype=torch.float32),
    )
    assert torch.equal(
        source,
        torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float32),
    )


def test_convert_fp8_weight_scale_inv_handles_moe_scale_names() -> None:
    for name in (
        "model.layers.0.mlp.experts.w13_weight_scale_inv",
        "model.layers.0.mlp.experts.w2_weight_scale_inv",
    ):
        converted = convert_fp8_weight_scale_inv_for_sglang(
            name, torch.tensor([128.0], dtype=torch.float32)
        )
        assert torch.allclose(converted, torch.tensor([1.0 / 128.0]))


def test_convert_fp8_weight_scale_inv_leaves_non_scale_or_existing_scale() -> None:
    weight = torch.tensor([2.0, 4.0], dtype=torch.float32)
    runtime_scale = torch.tensor([0.125, 0.25], dtype=torch.float32)

    assert convert_fp8_weight_scale_inv_for_sglang("linear.weight", weight) is weight
    assert (
        convert_fp8_weight_scale_inv_for_sglang(
            "linear.weight_scale_inv", runtime_scale
        )
        is runtime_scale
    )


def test_convert_fp8_weight_scale_inv_leaves_non_floating_tensors() -> None:
    packed = torch.tensor([2, 4], dtype=torch.int32)

    assert (
        convert_fp8_weight_scale_inv_for_sglang("linear.weight_scale_inv", packed)
        is packed
    )


@pytest.mark.parametrize(
    "scale",
    [
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
