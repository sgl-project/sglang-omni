# SPDX-License-Identifier: Apache-2.0
"""Tests for FP8 ``weight_scale_inv`` reciprocal preprocessing."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.quantization import convert_fp8_weight_scale_inv, is_fp8_block_quant


class TestIsFp8BlockQuant:
    """Tests for ``is_fp8_block_quant`` detection."""

    def test_true_for_block_fp8(self) -> None:
        assert is_fp8_block_quant(
            {"quant_method": "fp8", "weight_block_size": [128, 128]}
        )

    def test_false_without_block_size(self) -> None:
        assert not is_fp8_block_quant({"quant_method": "fp8"})

    def test_false_for_non_fp8(self) -> None:
        assert not is_fp8_block_quant(
            {"quant_method": "auto-round", "weight_block_size": [128, 128]}
        )

    def test_false_for_none(self) -> None:
        assert not is_fp8_block_quant(None)


class TestConvertFp8WeightScaleInv:
    """Tests for ``convert_fp8_weight_scale_inv``."""

    def test_regular_weight_passes_through(self) -> None:
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = convert_fp8_weight_scale_inv("model.weight", weight)
        assert result is weight

    def test_scale_inv_is_reciprocated(self) -> None:
        scale_inv = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)
        result = convert_fp8_weight_scale_inv(
            "model.layers.0.mlp.gate_proj.weight_scale_inv", scale_inv
        )
        expected = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_original_tensor_not_mutated(self) -> None:
        scale_inv = torch.tensor([2.0, 4.0], dtype=torch.float32)
        _ = convert_fp8_weight_scale_inv(
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv", scale_inv
        )
        assert torch.equal(scale_inv, torch.tensor([2.0, 4.0]))

    def test_empty_scale_raises(self) -> None:
        scale_inv = torch.tensor([], dtype=torch.float32)
        with pytest.raises(ValueError, match="Invalid empty FP8 scale tensor"):
            convert_fp8_weight_scale_inv("layer.weight_scale_inv", scale_inv)

    def test_zero_scale_raises(self) -> None:
        scale_inv = torch.tensor([2.0, 0.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="Invalid zero FP8 scale tensor"):
            convert_fp8_weight_scale_inv("layer.weight_scale_inv", scale_inv)

    def test_inf_scale_raises(self) -> None:
        scale_inv = torch.tensor([2.0, float("inf")], dtype=torch.float32)
        with pytest.raises(ValueError, match="Invalid non-finite FP8 scale tensor"):
            convert_fp8_weight_scale_inv("layer.weight_scale_inv", scale_inv)

    def test_nan_scale_raises(self) -> None:
        scale_inv = torch.tensor([2.0, float("nan")], dtype=torch.float32)
        with pytest.raises(ValueError, match="Invalid non-finite FP8 scale tensor"):
            convert_fp8_weight_scale_inv("layer.weight_scale_inv", scale_inv)

    def test_non_float_scale_raises(self) -> None:
        scale_inv = torch.tensor([1, 2, 3], dtype=torch.int32)
        with pytest.raises(TypeError, match="must be floating point"):
            convert_fp8_weight_scale_inv("layer.weight_scale_inv", scale_inv)
