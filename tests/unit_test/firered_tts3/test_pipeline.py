# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-driven pipeline topology, configuration validation, and state transport."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch
from sglang.srt.models.registry import ModelRegistry
from transformers import AutoConfig

from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.sources import patches_from_shared_block
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.hf_config import (
    FireRedTTS3Config,
    detect_fireredtts3_variant,
    register_fireredtts3_hf_config,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.utils.json import JsonValue


def test_firered_tts3_pipeline_is_registered() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FireRedTTS3ForConditionalGeneration")
        is FireRedTTS3PipelineConfig
    )


@pytest.mark.parametrize(
    "variant,markers,stage_names,reference_count,text_required,dual_stream",
    [
        (
            "base",
            {"spk_in_dim": 512},
            ["preprocessing", "reference_encode", "talker", "vocoder"],
            1,
            True,
            False,
        ),
        (
            "instruct",
            {"use_lm_head": True},
            ["preprocessing", "reference_encode", "talker", "vocoder"],
            None,
            False,
            False,
        ),
        (
            "speaker",
            {
                "text_latent_interleave_ratio": [2, 1],
                "speaker_token": "<|f531_speaker|>",
                "use_lm_head": True,
            },
            ["preprocessing", "talker", "vocoder"],
            0,
            False,
            True,
        ),
    ],
)
def test_checkpoint_config_drives_pipeline_topology(
    tmp_path: Path,
    variant: str,
    markers: dict[str, JsonValue],
    stage_names: list[str],
    reference_count: int | None,
    text_required: bool,
    dual_stream: bool,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "fireredtts3", **markers}), encoding="utf-8"
    )
    register_fireredtts3_hf_config()
    config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(config, FireRedTTS3Config)
    assert detect_fireredtts3_variant(config) == variant
    pipeline = FireRedTTS3PipelineConfig(model_path=str(tmp_path))
    assert pipeline.variant == variant
    assert [stage.name for stage in pipeline.stages] == stage_names
    assert {stage.process for stage in pipeline.stages} == {"pipeline"}
    assert pipeline.stage_named("preprocessing").next == stage_names[1]
    assert pipeline.terminal_stages == ["vocoder"]
    assert pipeline.required_speech_reference_count == reference_count
    assert pipeline.speech_reference_text_required is text_required
    assert pipeline.speech_dual_stream_supported is dual_stream


def test_unknown_checkpoint_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unrecognized FireRedTTS3 checkpoint"):
        detect_fireredtts3_variant(FireRedTTS3Config())


def test_firered_tts3_shares_generation_schedule_via_shared() -> None:
    """One shared entry feeds the schedule settings to both consumers."""

    config = FireRedTTS3PipelineConfig(model_path="model")
    patches = patches_from_shared_block(
        [
            {
                "select": {"stages": ["preprocessing", "talker"]},
                "factory": {"n_timesteps": 8, "max_patches": 256},
            }
        ],
        FireRedTTS3PipelineConfig,
        [stage.name for stage in config.stages],
    )
    resolved = ConfigResolver(config).resolve(patches).config

    for name in ("preprocessing", "talker"):
        kwargs = resolve_stage_typed_kwargs(resolved.stage_named(name))
        assert kwargs["n_timesteps"] == 8
        assert kwargs["max_patches"] == 256
    assert resolved.stage_factory_kwargs("preprocessing") == {
        "n_timesteps": 8,
        "max_patches": 256,
        "max_concurrency": 8,
    }


def test_firered_tts3_rejects_conflicting_preprocessing_schedule() -> None:

    config_values = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stages = {item["name"]: item for item in config_values["stages"]}
    stages["talker"]["factory"] = {"n_timesteps": 10}
    stages["preprocessing"]["factory"] = {"n_timesteps": 8}
    with pytest.raises(ValueError, match="n_timesteps"):
        FireRedTTS3PipelineConfig(**config_values)


def test_firered_tts3_rejects_mismatched_vocoder_stream_slots() -> None:

    config_values = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stages = {item["name"]: item for item in config_values["stages"]}
    stages["talker"]["engine"]["max_running_requests"] = 8
    stages["vocoder"]["factory"]["stream_slots"] = 16
    with pytest.raises(ValueError, match="stream_slots"):
        FireRedTTS3PipelineConfig(**config_values)


def test_firered_tts3_rejects_tp() -> None:

    config_values = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stage = next(item for item in config_values["stages"] if item["name"] == "talker")
    stage["tp_size"] = 2
    stage["gpu"] = [0, 1]
    stage["process"] = "talker"
    with pytest.raises(ValueError, match="tp_size=1"):
        FireRedTTS3PipelineConfig(**config_values)


def test_firered_tts3_model_registered_with_sglang() -> None:
    SGLModelRunner.register_omni_model(SGLModelRunner)

    assert (
        ModelRegistry.models["FireRedTTS3ForConditionalGeneration"]
        is FireRedTTS3SGLangModel
    )


def test_firered_tts3_state_wire_round_trip() -> None:

    state = FireRedTTS3State(
        variant="instruct",
        task_type="voice_design",
        text="hello",
        prompt_audio_path="ref.wav",
        prompt_text="reference",
        instructions="calm",
        language="zh",
        audio_in_path="edit.wav",
        input_ids=[1, 2, 3],
        n_timesteps=8,
        inference_cfg=1.2,
        stop_threshold=0.6,
        min_patches=4,
        max_patches=256,
        max_engine_tokens=788,
        dual_stream=True,
        text_stream_ids=[101, 102, 103],
        text_pad_id=151671,
        seed=7,
        stream=True,
        text_temperature=0.7,
        text_top_p=0.8,
        text_top_k=20,
        left_context_size=12,
        prompt_latents=torch.rand(1, 12, 64),
        audio_in_latents=torch.rand(1, 8, 64),
        speaker_embedding=torch.rand(1, 512),
        generated_latents=torch.rand(1, 8, 64),
        gen_text="generated",
        finish_reason="stop",
        prompt_tokens=3,
        completion_tokens=5,
        engine_time_s=0.25,
    )

    before = state.to_dict()
    restored = FireRedTTS3State.from_dict(before)
    after = restored.to_dict()

    assert set(after) == set(before)
    for state_field in dataclasses.fields(state):
        expected = getattr(state, state_field.name)
        actual = getattr(restored, state_field.name)
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor), state_field.name
            assert torch.equal(actual, expected), state_field.name
        else:
            assert actual == expected, state_field.name
