# SPDX-License-Identifier: Apache-2.0
"""Tests for the quantization discovery + weight-preprocessor entry points."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.quantization import (
    get_weight_preprocessor,
    quant_method_name,
    resolve_quant_config,
)


class TestExtractQuantizationConfig:
    """Tests for ``resolve_quant_config`` composite-config walking."""

    def test_reads_root_quantization_config(self) -> None:
        config = SimpleNamespace(
            quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]}
        )
        result = resolve_quant_config(config)
        assert result is not None
        assert result["quant_method"] == "fp8"

    def test_reads_nested_thinker_config(self) -> None:
        thinker = SimpleNamespace(
            quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]}
        )
        config = SimpleNamespace(quantization_config=None, thinker_config=thinker)
        result = resolve_quant_config(config)
        assert result is not None
        assert result["quant_method"] == "fp8"

    def test_object_shaped_config_converted_to_dict(self) -> None:
        quant_config = SimpleNamespace(
            quant_method="fp8", weight_block_size=[128, 128], bits=8
        )
        config = SimpleNamespace(quantization_config=quant_config)
        result = resolve_quant_config(config)
        assert isinstance(result, dict)
        assert result["quant_method"] == "fp8"

    def test_reads_compression_config(self) -> None:
        config = SimpleNamespace(
            quantization_config=None,
            compression_config={"quant_method": "compressed-tensors"},
        )
        result = resolve_quant_config(config)
        assert result is not None
        assert result["quant_method"] == "compressed-tensors"

    def test_cyclic_config_does_not_infinite_loop(self) -> None:
        config = SimpleNamespace(quantization_config=None)
        config.text_config = config

        assert resolve_quant_config(config) is None


class TestQuantMethodName:
    """Tests for ``quant_method_name`` normalization."""

    def test_normalizes_underscore_to_hyphen(self) -> None:
        assert quant_method_name({"quant_method": "auto_round"}) == "auto-round"

    def test_lowercases(self) -> None:
        assert quant_method_name({"quant_method": "FP8"}) == "fp8"


class TestResolveWeightPreprocessor:
    """Tests for ``get_weight_preprocessor`` fixed dispatch."""

    def test_identity_when_no_quantization(self) -> None:
        config = SimpleNamespace(model_type="qwen3")
        preprocess = get_weight_preprocessor(config)

        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = preprocess("model.layers.0.weight", weight)
        assert result is weight

    def test_fp8_block_returns_reciprocal_preprocessor(self) -> None:
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        }
        preprocess = get_weight_preprocessor(
            SimpleNamespace(**config), fp8_scale_inverted=True
        )

        # Regular weights pass through.
        weight = torch.tensor([[1.0, 2.0]])
        assert torch.equal(preprocess("model.weight", weight), weight)

        # Scales are reciprocated.
        scale = torch.tensor([2.0, 4.0])
        result = preprocess("model.layers.0.weight_scale_inv", scale)
        assert torch.allclose(result, torch.tensor([0.5, 0.25]))

    def test_fp8_block_default_is_identity(self) -> None:
        # Without opting in, block-FP8 is left to SGLang's native handling.
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        }
        preprocess = get_weight_preprocessor(SimpleNamespace(**config))
        scale = torch.tensor([2.0, 4.0])
        assert torch.equal(preprocess("model.layers.0.weight_scale_inv", scale), scale)

    def test_fp8_without_block_size_is_identity(self) -> None:
        # Per-tensor FP8 is handled entirely by SGLang; no Omni preprocessing.
        config = SimpleNamespace(
            quantization_config={"quant_method": "fp8"},
        )
        preprocess = get_weight_preprocessor(config)
        scale = torch.tensor([2.0])
        assert torch.equal(preprocess("layer.weight_scale_inv", scale), scale)

    def test_auto_round_is_identity(self) -> None:
        config = SimpleNamespace(
            quantization_config={"quant_method": "auto-round", "bits": 4},
        )
        preprocess = get_weight_preprocessor(config)
        weight = torch.tensor([[1.0, 2.0]])
        assert torch.equal(preprocess("model.layers.0.weight", weight), weight)

    def test_nested_fp8_config_is_detected(self) -> None:
        thinker = SimpleNamespace(
            quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]}
        )
        config = SimpleNamespace(quantization_config=None, thinker_config=thinker)
        preprocess = get_weight_preprocessor(config, fp8_scale_inverted=True)

        scale = torch.tensor([2.0])
        assert torch.allclose(
            preprocess("layers.0.weight_scale_inv", scale), torch.tensor([0.5])
        )
