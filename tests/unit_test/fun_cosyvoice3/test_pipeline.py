# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.runtime import (
    resolve_stage_factory_args,
    resolve_stage_typed_kwargs,
)
from sglang_omni.models.fun_cosyvoice3 import CAPABILITIES
from sglang_omni.models.fun_cosyvoice3.config import (
    FunCosyVoice3IsolatedVocoderPipelineConfig,
    FunCosyVoice3PipelineConfig,
)
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


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
    assert type(config).stage_config_cls("tts_engine").engine_stage
    assert config.process_local_edges() == frozenset({("preprocessing", "tts_engine")})
    assert CAPABILITIES.supports_streaming_vocoder is False
    assert all(stage.gpu_memory_fraction is None for stage in config.stages)

    vocoder = next(stage for stage in config.stages if stage.name == "vocoder")
    assert vocoder.factory.dtype == "bfloat16"
    assert vocoder.factory.model_extra == {
        "flow_batch_bucket_frames": 50,
        "flow_batch_admission_frames": 2000,
        "enable_dit_torch_compile": False,
    }

    build_compiled_process_topology(config)
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FunCosyVoice3SGLangModel")
        is FunCosyVoice3PipelineConfig
    )


def test_fun_cosyvoice3_isolated_vocoder_variant_gives_the_flow_decoder_a_process() -> (
    None
):
    config = FunCosyVoice3IsolatedVocoderPipelineConfig(model_path="model")
    stages = {stage.name: stage for stage in config.stages}

    assert stages["vocoder"].process == "vocoder"
    # Preprocessing stays with the engine: it hands prepared requests over through
    # module state, which is why process_local_edges pins that edge.
    assert stages["preprocessing"].process == "pipeline"
    assert config.process_local_edges() == frozenset({("preprocessing", "tts_engine")})

    assert stages["tts_engine"].gpu_memory_fraction == pytest.approx(0.80)
    assert stages["vocoder"].gpu_memory_fraction == pytest.approx(0.12)

    placement = build_stage_placement_plan(config)
    assert placement.gpus[0].total_gpu_memory_fraction == pytest.approx(0.92)
    assert placement.gpus[0].missing_fraction_stage_names == ()

    topology = build_compiled_process_topology(config)
    assert topology.stage_to_process == {
        "preprocessing": "pipeline",
        "tts_engine": "pipeline",
        "vocoder": "vocoder",
    }


def test_fun_cosyvoice3_bare_process_override_is_refused_without_budgets() -> None:
    """The variant exists because the dotted override alone cannot launch."""
    merged = ConfigManager(
        FunCosyVoice3PipelineConfig(model_path="model")
    ).merge_config({"vocoder.process": "vocoder"})
    with pytest.raises(ValueError, match="without a declared total footprint"):
        build_compiled_process_topology(merged)


def test_fun_cosyvoice3_isolated_vocoder_forwards_the_engine_budget() -> None:
    config = FunCosyVoice3IsolatedVocoderPipelineConfig(model_path="model")
    stage = next(stage for stage in config.stages if stage.name == "tts_engine")
    args = resolve_stage_factory_args(stage, config, gpu_id=0)
    assert args["total_gpu_memory_fraction"] == pytest.approx(0.80)

    builder = FunCosyVoice3EngineBuilder(total_gpu_memory_fraction=0.80)
    assert builder.infra_kwargs()["total_gpu_memory_fraction"] == pytest.approx(0.80)
    # The stage budget sizes KV; the model's own static fraction is left alone.
    assert builder.generation_defaults(dtype="bfloat16")["mem_fraction_static"] == 0.85
    # No declared budget means the single-process call is untouched.
    assert FunCosyVoice3EngineBuilder().infra_kwargs() == {}


def test_fun_cosyvoice3_flow_factory_overrides_use_typed_path() -> None:
    config = FunCosyVoice3PipelineConfig(model_path="model")
    manager = ConfigManager(config)
    merged = manager.merge_config(
        {
            "vocoder.factory.flow_batch_bucket_frames": 100,
            "vocoder.factory.flow_batch_admission_frames": 4000,
        }
    )
    vocoder = next(stage for stage in merged.stages if stage.name == "vocoder")

    assert vocoder.factory.model_extra == {
        "flow_batch_bucket_frames": 100,
        "flow_batch_admission_frames": 4000,
        "enable_dit_torch_compile": False,
    }
    args = resolve_stage_typed_kwargs(vocoder)
    assert args["flow_batch_bucket_frames"] == 100
    assert args["flow_batch_admission_frames"] == 4000


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
