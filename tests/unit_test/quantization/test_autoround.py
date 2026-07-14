# SPDX-License-Identifier: Apache-2.0
"""Tests for stage-local checkpoint name normalization (e.g. AutoRound)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.quantization import (
    needs_quant_config_normalization,
    normalize_quant_config,
)


def _make_model_config(
    architecture: str | None,
    quantization_config: object,
) -> SimpleNamespace:
    """Build a minimal ``model_config`` stub for normalization tests."""
    hf_config = SimpleNamespace(
        architectures=[architecture] if architecture is not None else [],
        quantization_config=quantization_config,
    )
    return SimpleNamespace(hf_config=hf_config)


class TestNeedsStageLocalNormalization:
    """Tests for the ``needs_quant_config_normalization`` dispatch."""

    def test_true_for_auto_round(self) -> None:
        assert needs_quant_config_normalization({"quant_method": "auto-round"})

    def test_true_for_underscore_variant(self) -> None:
        assert needs_quant_config_normalization({"quant_method": "auto_round"})

    def test_false_for_fp8(self) -> None:
        assert not needs_quant_config_normalization({"quant_method": "fp8"})

    def test_false_for_none(self) -> None:
        assert not needs_quant_config_normalization(None)


class TestNormalizeStageLocalCheckpointConfig:
    """Tests for stripping the stage prefix from block names / extra_config."""

    def test_strips_thinker_prefix_from_block_names(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_strips_talker_prefix_from_block_names(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "talker.model.layers",
        }
        model_config = _make_model_config("Qwen3OmniTalker", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_asr_architecture_strips_thinker_prefix(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config(
            "Qwen3ASRForConditionalGeneration", quant_config
        )

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_strips_prefix_from_comma_separated_list(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers,thinker.model.experts",
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers,model.experts"

    def test_normalizes_block_name_list_input(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": ["thinker.model.layers", "model.shared"],
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == [
            "model.layers",
            "model.shared",
        ]

    def test_strips_prefix_from_extra_config_keys(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                r"thinker\.model\.layers\.0": {"bits": 8},
                "thinker.model.layers.1": {"bits": 4},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            "model.layers.1": {"bits": 4},
        }

    def test_extra_config_normalized_when_block_names_already_stripped(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "model.layers",
            "extra_config": {
                r"thinker\.model\.layers\.0": {"bits": 8},
                "thinker.model.layers.1": {"bits": 4},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"
        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            "model.layers.1": {"bits": 4},
        }

    def test_extra_config_prefix_anchored_at_pattern_start(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                # leading prefix -> should be stripped
                r"thinker\.model\.layers\.0": {"bits": 8},
                # prefix inside an alternation -> must be preserved
                r"(?:thinker|decoder)\.model\.layers\.1": {"bits": 4},
                # prefix as a substring of another name -> must be preserved
                r"thinker_audio\.model\.layers\.2": {"bits": 4},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8},
            r"(?:thinker|decoder)\.model\.layers\.1": {"bits": 4},
            r"thinker_audio\.model\.layers\.2": {"bits": 4},
        }

    def test_strips_prefix_from_leading_wildcard_pattern(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
            "extra_config": {
                r".*thinker\.model\.layers\.\d+\.mlp\.gate.*": {"bits": 8},
            },
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["extra_config"] == {
            r".*model\.layers\.\d+\.mlp\.gate.*": {"bits": 8},
        }

    def test_no_change_when_block_names_lack_prefix(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "model.layers",
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_unknown_architecture_leaves_block_names_unchanged(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config("SomeOtherForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "thinker.model.layers"

    def test_normalizes_config_that_lives_only_on_nested_stage_attr(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        thinker_config = SimpleNamespace(quantization_config=quant_config)
        hf_config = SimpleNamespace(
            architectures=["Qwen3OmniThinkerForCausalLM"],
            thinker_config=thinker_config,
        )
        model_config = SimpleNamespace(hf_config=hf_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"
        assert thinker_config.quantization_config["block_name_to_quantize"] == (
            "model.layers"
        )

    def test_missing_hf_config_is_noop(self) -> None:
        model_config = SimpleNamespace(hf_config=None)
        normalize_quant_config(model_config)

    def test_missing_block_name_to_quantize_is_noop(self) -> None:
        quant_config = {"quant_method": "auto-round"}
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config == {"quant_method": "auto-round"}

    def test_non_dict_quantization_config_raises(self) -> None:
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", "not-a-dict")
        with pytest.raises(TypeError, match="unsupported type"):
            normalize_quant_config(model_config)


class TestObjectShapedConfig:
    """Object-shaped quant configs are converted and written back."""

    def test_object_quant_config_converted_and_written_back(self) -> None:
        quant_config = SimpleNamespace(
            quant_method="auto-round",
            block_name_to_quantize="thinker.model.layers",
            bits=4,
        )
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["block_name_to_quantize"] == (
            "model.layers"
        )

    def test_object_with_to_dict_converted_and_written_back(self) -> None:
        class HasToDict:
            def __init__(self):
                self.quant_method = "auto-round"
                self.block_name_to_quantize = "thinker.model.layers"
                self.bits = 4

            def to_dict(self):
                return {
                    "quant_method": self.quant_method,
                    "block_name_to_quantize": self.block_name_to_quantize,
                    "bits": self.bits,
                }

        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", HasToDict())

        normalize_quant_config(model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["block_name_to_quantize"] == (
            "model.layers"
        )

    def test_object_extra_config_normalized_when_blocks_already_stripped(self) -> None:
        quant_config = SimpleNamespace(
            quant_method="auto-round",
            block_name_to_quantize="model.layers",
            extra_config={r"thinker\.model\.layers\.0": {"bits": 8}},
        )
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8}
        }

    def test_unsupported_quant_config_type_raises(self) -> None:
        model_config = _make_model_config(
            "Qwen3OmniThinkerForCausalLM", ["not", "a", "dict"]
        )
        with pytest.raises(TypeError, match="unsupported type"):
            normalize_quant_config(model_config)
