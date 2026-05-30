# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig


def test_whisper_asr_config_uses_single_batched_stage() -> None:
    config = WhisperASRPipelineConfig(model_path="openai/whisper-large-v3")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_whisper_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("WhisperForConditionalGeneration")
        is WhisperASRPipelineConfig
    )
