# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.config import build_process_topology_plan, build_stage_placement_plan
from sglang_omni.models.fun_cosyvoice3 import CAPABILITIES
from sglang_omni.models.fun_cosyvoice3.config import FunCosyVoice3PipelineConfig
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_fun_cosyvoice3_config_and_registry_contract() -> None:
    config = FunCosyVoice3PipelineConfig(model_path="model")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert [stage.process for stage in config.stages] == [
        "pipeline",
        "pipeline",
        "pipeline",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.required_speech_reference_count == 1
    assert config.speech_reference_text_excludes_instructions is True
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}
    assert config.generation_sglang_role_to_stage() == {"generation": "tts_engine"}
    assert config.mem_fraction_role_to_stage() == {"talker": "tts_engine"}
    assert config.process_safe_edges() == frozenset({("tts_engine", "vocoder")})
    assert CAPABILITIES.supports_streaming_vocoder is False

    build_process_topology_plan(config, build_stage_placement_plan(config))
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FunCosyVoice3SGLangModel")
        is FunCosyVoice3PipelineConfig
    )


def test_fun_cosyvoice3_state_round_trip_preserves_wire_contract() -> None:
    state = FunCosyVoice3State(
        text="hello",
        language="en",
        instructions="speak brightly",
        ref_text="reference",
        stream=True,
        speed=1.25,
        seed=7,
        generation_kwargs={"max_new_tokens": 32},
        flow_embedding=torch.tensor([[1.0, 2.0]]),
        flow_prompt_speech_token=torch.tensor([[10, 11]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.ones(1, 2, 80),
        audio_codes=torch.tensor([[20], [21]], dtype=torch.long),
    )

    wire = state.to_dict()
    restored = FunCosyVoice3State.from_dict(wire)

    assert wire["flow_embedding"] == [[1.0, 2.0]]
    assert restored.text == state.text
    assert restored.language == state.language
    assert restored.instructions == state.instructions
    assert restored.stream is True
    assert restored.speed == 1.25
    assert restored.seed == 7
    assert restored.generation_kwargs == {"max_new_tokens": 32}
    assert restored.flow_prompt_speech_token == [[10, 11]]
    assert restored.flow_prompt_speech_feat[0][0] == [1.0] * 80
    assert restored.audio_codes == [[20], [21]]
