# SPDX-License-Identifier: Apache-2.0
"""Tests for stage-local checkpoint name normalization (e.g. AutoRound)."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
)

from sglang_omni.quantization import (
    needs_quant_config_normalization,
    normalize_quant_config,
)


def make_model_config(
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

    def test_true_for_underscore_variant(self) -> None:
        assert needs_quant_config_normalization({"quant_method": "auto_round"})


class TestNormalizeStageLocalCheckpointConfig:
    """Tests for stripping the stage prefix from block names / extra_config."""

    def test_strips_talker_prefix_from_block_names(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "talker.model.layers",
        }
        model_config = make_model_config("Qwen3OmniTalker", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_asr_architecture_strips_thinker_prefix(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = make_model_config(
            "Qwen3ASRForConditionalGeneration", quant_config
        )

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_strips_prefix_from_comma_separated_list(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers,thinker.model.experts",
        }
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers,model.experts"

    def test_normalizes_block_name_list_input(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": ["thinker.model.layers", "model.shared"],
        }
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

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
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

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
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

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
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["extra_config"] == {
            r".*model\.layers\.\d+\.mlp\.gate.*": {"bits": 8},
        }

    def test_unknown_architecture_leaves_block_names_unchanged(self) -> None:
        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = make_model_config("SomeOtherForCausalLM", quant_config)

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

    def test_missing_block_name_to_quantize_is_noop(self) -> None:
        quant_config = {"quant_method": "auto-round"}
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config == {"quant_method": "auto-round"}

    def test_non_dict_quantization_config_raises(self) -> None:
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", "not-a-dict")
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
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

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

        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", HasToDict())

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
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["extra_config"] == {
            r"model\.layers\.0": {"bits": 8}
        }


class TestNormalizeCompressedTensorsIgnoreList:
    """Tests for stripping the stage prefix from a compressed-tensors `ignore`."""

    def test_strips_thinker_prefix_from_ignore_entries(self) -> None:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": [
                "thinker.audio_tower.layers.0.fc1",
                "thinker.visual.blocks.0.mlp.down_proj",
                "lm_head",
            ],
        }
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["ignore"] == [
            "audio_tower.layers.0.fc1",
            "visual.blocks.0.mlp.down_proj",
            "lm_head",
        ]

    def test_leaves_another_stage_prefix_alone(self) -> None:
        """A talker entry cannot match a thinker module either way, so stripping
        it would only invent a name the stage does not have."""
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": ["talker.model.layers.0.mlp.gate"],
        }
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["ignore"] == ["talker.model.layers.0.mlp.gate"]

    def test_compressed_tensors_needs_normalization(self) -> None:
        assert needs_quant_config_normalization({"quant_method": "compressed_tensors"})

    def test_unprefixed_ignore_list_is_untouched(self) -> None:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": ["audio_tower.layers.0.fc1"],
        }
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["ignore"] == ["audio_tower.layers.0.fc1"]

    def test_object_ignore_list_is_written_back(self) -> None:
        """An object-valued quant config is copied before mutation, so the
        normalized ignore list only reaches SGLang via the write-back."""
        quant_config = SimpleNamespace(
            quant_method="compressed-tensors",
            format="pack-quantized",
            ignore=["thinker.audio_tower.layers.0.fc1"],
        )
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert isinstance(model_config.hf_config.quantization_config, dict)
        assert model_config.hf_config.quantization_config["ignore"] == [
            "audio_tower.layers.0.fc1"
        ]

    def test_rewrites_stage_prefix_as_optional_in_regex_entries(self) -> None:
        """compressed-tensors also accepts `re:` entries, which SGLang matches
        with `re.match`. Deleting the prefix from a regex is unsound, so each
        occurrence is made optional instead and the rest is left as written."""
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": [
                r"re:thinker\.audio_tower\..*",
                r"re:^thinker\.visual\..*",
                "re:thinker.audio_tower.*",
                r"re:.*thinker\.audio_tower\.conv.*",
                r"re:.*thinker.audio_tower.*",
                r"re:thinker\.audio_tower\..*|thinker\.visual\..*",
                r"re:talker\.model\..*",
                "re:lm_head",
            ],
        }
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["ignore"] == [
            r"re:(?:thinker\.)?audio_tower\..*",
            r"re:^(?:thinker\.)?visual\..*",
            "re:(?:thinker.)?audio_tower.*",
            r"re:.*(?:thinker\.)?audio_tower\.conv.*",
            r"re:.*(?:thinker.)?audio_tower.*",
            r"re:(?:thinker\.)?audio_tower\..*|(?:thinker\.)?visual\..*",
            r"re:talker\.model\..*",
            "re:lm_head",
        ]

    @pytest.mark.parametrize(
        "raw, layer",
        [
            (r"re:thinker\.audio_tower\..*", "audio_tower.layers.0.fc1"),
            (r"re:^thinker\.visual\..*", "visual.blocks.0.mlp.fc1"),
            ("re:thinker.audio_tower.*", "audio_tower.layers.0.fc1"),
            (r"re:.*thinker\.audio_tower\.conv.*", "audio_tower.conv2d1"),
            (r"re:.*thinker.audio_tower.*", "audio_tower.conv2d1"),
            (
                r"re:thinker\.audio_tower\..*|thinker\.visual\..*",
                "visual.blocks.0.mlp.fc1",
            ),
        ],
        ids=[
            "escaped",
            "anchored",
            "unescaped_dot",
            "escaped_leading_wildcard",
            "unescaped_leading_wildcard",
            "second_alternative",
        ],
    )
    def test_rewritten_regex_matches_a_stage_local_layer(
        self, raw: str, layer: str
    ) -> None:
        """The point of the rewrite: SGLang's own matcher must exclude the
        stage-local module that the raw checkpoint entry missed, and must still
        exclude the prefixed spelling."""
        assert not check_equal_or_regex_match(layer, [raw])

        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": [raw],
        }
        normalize_quant_config(
            make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)
        )
        rewritten = quant_config["ignore"]

        assert check_equal_or_regex_match(layer, rewritten)
        assert check_equal_or_regex_match(f"thinker.{layer}", rewritten)
        assert not check_equal_or_regex_match(
            "model.layers.0.self_attn.q_proj", rewritten
        )

    @pytest.mark.parametrize("raw", ["re:thinker.*", "re:^thinker.*", r"re:thinker\.*"])
    def test_quantifier_owned_separator_is_left_as_written(self, raw: str) -> None:
        """In `thinker.*` the dot belongs to the quantifier, so the entry has no
        unambiguous stage prefix to make optional. Deleting one produced the
        uncompilable `re:*`; it is now reported and left alone instead."""
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": [raw],
        }
        normalize_quant_config(
            make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)
        )

        assert quant_config["ignore"] == [raw]
        re.compile(quant_config["ignore"][0][len("re:") :])

    def test_quantifier_owned_separator_is_reported_not_salvaged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The quantifier guard has to skip the occurrence outright. Without it
        the rewrite still produces an uncompilable pattern and only the compile
        fallback saves the load, which is defence in depth, not the design."""
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": ["re:thinker.*"],
        }
        with caplog.at_level(logging.WARNING, logger="sglang_omni.quantization"):
            normalize_quant_config(
                make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)
            )

        assert quant_config["ignore"] == ["re:thinker.*"]
        assert "cannot be rewritten" in caplog.text
        assert "did not survive" not in caplog.text

    def test_every_rewritten_entry_compiles(self) -> None:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": [
                r"re:thinker\.audio_tower\..*",
                "re:thinker.*",
                r"re:^thinker\.(audio_tower|visual)\..*",
                r"re:.*thinker.audio_tower.*",
                "thinker.audio_tower.fc1",
            ],
        }
        normalize_quant_config(
            make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)
        )

        for entry in quant_config["ignore"]:
            if entry.startswith("re:"):
                re.compile(entry[len("re:") :])
            else:
                pass

    def test_non_string_ignore_entries_are_preserved(self) -> None:
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": ["thinker.audio_tower.fc1", None, 7],
        }
        model_config = make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        normalize_quant_config(model_config)

        assert quant_config["ignore"] == ["audio_tower.fc1", None, 7]
