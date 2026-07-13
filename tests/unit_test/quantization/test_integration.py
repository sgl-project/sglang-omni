# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the Omni quantization adapters in model_worker."""

from __future__ import annotations

from types import SimpleNamespace


def _make_model_config(
    architecture: str, quantization_config: object
) -> SimpleNamespace:
    hf_config = SimpleNamespace(
        architectures=[architecture],
        quantization_config=quantization_config,
    )
    return SimpleNamespace(hf_config=hf_config)


class TestApplyOmniQuantizationAdapters:
    """Tests for ``model_worker._apply_omni_quantization_adapters``."""

    def test_auto_round_triggers_stage_normalization(self) -> None:
        from sglang_omni.model_runner import model_worker

        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", quant_config)

        model_worker._apply_omni_quantization_adapters(model_config)

        # Stage prefix must be stripped for SGLang's AutoRoundConfig matching.
        assert quant_config["block_name_to_quantize"] == "model.layers"

    def test_fp8_does_not_normalize_block_names(self) -> None:
        from sglang_omni.model_runner import model_worker

        quant_config = {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            # An unrelated field that must not be rewritten.
            "block_name_to_quantize": "thinker.model.layers",
        }
        model_config = _make_model_config("Qwen3OmniTalker", quant_config)

        model_worker._apply_omni_quantization_adapters(model_config)

        # FP8 does not use stage-local block-name matching; leave it untouched.
        assert quant_config["block_name_to_quantize"] == "thinker.model.layers"

    def test_no_quantization_is_noop(self) -> None:
        from sglang_omni.model_runner import model_worker

        model_config = _make_model_config("Qwen3OmniThinkerForCausalLM", None)

        # Must not raise.
        model_worker._apply_omni_quantization_adapters(model_config)

    def test_nested_auto_round_config_is_detected(self) -> None:
        from sglang_omni.model_runner import model_worker

        quant_config = {
            "quant_method": "auto-round",
            "block_name_to_quantize": "thinker.model.layers",
        }
        thinker = SimpleNamespace(quantization_config=quant_config)
        hf_config = SimpleNamespace(
            architectures=["Qwen3OmniThinkerForCausalLM"],
            quantization_config=quant_config,
            thinker_config=thinker,
        )
        model_config = SimpleNamespace(hf_config=hf_config)

        model_worker._apply_omni_quantization_adapters(model_config)

        assert quant_config["block_name_to_quantize"] == "model.layers"
