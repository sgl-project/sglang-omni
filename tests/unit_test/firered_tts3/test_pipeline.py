# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-driven pipeline topology, configuration validation, and state transport."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest import mock

import pytest
import torch
from sglang.srt.models.registry import ModelRegistry
from transformers import AutoConfig

from sglang_omni.cli.config import resolve_sources
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.sources import patches_from_shared_block
from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder
from sglang_omni.models.firered_tts3.hf_config import (
    FireRedTTS3Config,
    FireRedTTS3Variant,
    detect_fireredtts3_variant,
    register_fireredtts3_hf_config,
)
from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    engine_token_budget,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.utils.imports import import_string
from sglang_omni.utils.json import JsonValue


def test_firered_tts3_pipeline_is_registered() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FireRedTTS3ForConditionalGeneration")
        is FireRedTTS3PipelineConfig
    )


@pytest.mark.parametrize(
    "variant,reference_count,reference_text_required,has_reference_stage",
    [
        ("base", 1, True, True),
        ("instruct", None, False, True),
        ("speaker", None, False, False),
    ],
)
def test_example_config_selects_pipeline_without_local_checkpoint(
    variant: FireRedTTS3Variant,
    reference_count: int | None,
    reference_text_required: bool,
    has_reference_stage: bool,
) -> None:
    """A remote checkpoint must select validation and stages before download."""
    config_path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "configs"
        / f"firered_tts3_{variant}.yaml"
    )
    with mock.patch(
        "sglang_omni.models.firered_tts3.config.load_checkpoint_hf_config",
        return_value=None,
    ):
        pipeline = ConfigManager.from_file(str(config_path)).config
    assert isinstance(pipeline, FireRedTTS3PipelineConfig)
    assert pipeline.required_speech_reference_count == reference_count
    assert pipeline.speech_reference_text_required is reference_text_required
    assert any(stage.name == "reference_encode" for stage in pipeline.stages) is (
        has_reference_stage
    )
    validator = import_string(pipeline.speech_request_validator_factory)(
        default_model="tts"
    )
    assert validator.supports_dual_stream is (variant == "speaker")


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
            None,
            False,
            True,
        ),
    ],
)
def test_checkpoint_config_drives_pipeline_topology(
    tmp_path: Path,
    variant: FireRedTTS3Variant,
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
    config_path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "configs"
        / f"firered_tts3_{variant}.yaml"
    )
    resolved = resolve_sources(
        model_path=str(tmp_path),
        config_file=str(config_path),
        text_only=False,
        variant=None,
        mem_fraction_static=None,
        argv=[],
    ).resolved.config
    assert resolved.model_path == str(tmp_path)
    assert [stage.name for stage in resolved.stages] == stage_names
    service = import_string(pipeline.speech_request_validator_factory)(
        default_model="tts",
    )
    prepared = service.parse_generation_request(
        {
            "input": "hello",
            "extra_params": {"dual_stream": dual_stream, "style": "calm"},
            "initial_codec_chunk_frames": 8,
        }
    )
    generated = service.build_generate_request(
        prepared.request,
        validate=False,
        reference_descriptors=prepared.reference_descriptors,
    )
    assert generated.extra_params == {
        "dual_stream": dual_stream,
        "style": "calm",
        "initial_codec_chunk_frames": 8,
    }


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
        factory_parameters = resolve_stage_typed_kwargs(resolved.stage_named(name))
        assert factory_parameters["n_timesteps"] == 8
        assert factory_parameters["max_patches"] == 256
    assert resolved.stage_factory_kwargs("preprocessing") == {
        "n_timesteps": 8,
        "max_patches": 256,
        "max_concurrency": 8,
    }


def test_firered_tts3_rejects_conflicting_preprocessing_schedule() -> None:

    config_values = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stages = {stage["name"]: stage for stage in config_values["stages"]}
    stages["talker"]["factory"] = {"n_timesteps": 10}
    stages["preprocessing"]["factory"] = {"n_timesteps": 8}
    with pytest.raises(ValueError, match="n_timesteps"):
        FireRedTTS3PipelineConfig(**config_values)


def test_firered_tts3_rejects_mismatched_vocoder_stream_slots() -> None:

    config_values = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stages = {stage["name"]: stage for stage in config_values["stages"]}
    stages["talker"]["engine"]["max_running_requests"] = 8
    stages["vocoder"]["factory"]["stream_slots"] = 16
    with pytest.raises(ValueError, match="stream_slots"):
        FireRedTTS3PipelineConfig(**config_values)


def test_firered_tts3_rejects_tp() -> None:

    config_values = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stage = next(
        stage for stage in config_values["stages"] if stage["name"] == "talker"
    )
    stage["tp_size"] = 2
    stage["gpu"] = [0, 1]
    stage["process"] = "talker"
    with pytest.raises(ValueError, match="tp_size=1"):
        FireRedTTS3PipelineConfig(**config_values)


def test_firered_tts3_model_registered_with_sglang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(
        ModelRegistry.models, "FireRedTTS3ForConditionalGeneration", raising=False
    )
    pipeline = FireRedTTS3PipelineConfig(model_path="model")
    builder = FireRedTTS3EngineBuilder(
        max_running_requests=pipeline.generation_admission_defaults()[
            "max_running_requests"
        ],
        **pipeline.stage_factory_kwargs("talker"),
    )
    builder.pre_infra_setup("model")

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


@pytest.mark.parametrize(
    "wire_values, expected_fields, expected_budget",
    [
        ({}, (10, 400, 2.0, 0.5, 6), 420),
        ({"variant": "speaker"}, (10, 400, 2.0, 0.5, 6), 420),
        (
            {
                "n_timesteps": None,
                "max_patches": None,
                "inference_cfg": None,
                "stop_threshold": None,
                "min_patches": None,
            },
            (10, 400, 0.0, 0.0, 6),
            420,
        ),
        (
            {
                "n_timesteps": 0,
                "max_patches": 0,
                "inference_cfg": 0,
                "stop_threshold": 0,
                "min_patches": 0,
            },
            (10, 400, 0.0, 0.0, 6),
            420,
        ),
        (
            {
                "n_timesteps": 8,
                "max_patches": 256,
                "inference_cfg": 1.25,
                "stop_threshold": 0.75,
                "min_patches": 3,
            },
            (8, 256, 1.25, 0.75, 3),
            276,
        ),
    ],
)
def test_legacy_state_wire_defaults_and_overrides(
    wire_values: dict[str, JsonValue],
    expected_fields: tuple[int, int, float, float, int],
    expected_budget: int,
) -> None:
    """Preserve partial payloads and their derived engine budgets across upgrades."""
    state = FireRedTTS3State.from_dict(wire_values)
    assert (
        state.n_timesteps,
        state.max_patches,
        state.inference_cfg,
        state.stop_threshold,
        state.min_patches,
    ) == expected_fields
    assert engine_token_budget(state) == expected_budget
