# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.zonos2.config import ZONOS2PipelineConfig
from sglang_omni.models.zonos2.payload_types import ZONOS2State
from sglang_omni.models.zonos2.stages import create_sglang_tts_engine_executor


def test_zonos2_config_is_registered():
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Zonos2ForConditionalGeneration")
        is ZONOS2PipelineConfig
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("ZONOS2ForConditionalGeneration")
        is ZONOS2PipelineConfig
    )


def test_zonos2_pipeline_shape():
    config = ZONOS2PipelineConfig(model_path="Zyphra/ZONOS2")

    assert config.entry_stage == "text_frontend"
    assert [stage.name for stage in config.stages] == [
        "text_frontend",
        "speaker_embedding",
        "tts_engine",
        "vocoder",
    ]
    assert config.stages[0].next == "speaker_embedding"
    assert config.stages[1].next == "tts_engine"
    assert config.stages[2].next == "vocoder"
    assert config.stages[3].terminal is True


def test_zonos2_example_config_loads():
    manager = ConfigManager.from_file("examples/configs/zonos2.yaml")

    assert isinstance(manager.config, ZONOS2PipelineConfig)
    assert manager.config.model_path == "Zyphra/ZONOS2"


def test_zonos2_state_round_trip_defaults():
    state = ZONOS2State(text="hello", ref_audio="speaker.wav", ref_text="hi")

    restored = ZONOS2State.from_dict(state.to_dict())

    assert restored.text == "hello"
    assert restored.ref_audio == "speaker.wav"
    assert restored.ref_text == "hi"
    assert restored.frame_width == 9
    assert restored.codebook_size == 1024
    assert restored.sample_rate == 44100


def test_zonos2_runtime_factories_are_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="ZONOS2 runtime support"):
        create_sglang_tts_engine_executor()
