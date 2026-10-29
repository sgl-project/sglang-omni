# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-driven pipeline topology, configuration validation, and state transport."""

from __future__ import annotations

import base64
import dataclasses
import json
from pathlib import Path
from typing import Literal
from unittest import mock

import pytest
import torch
from fastapi.testclient import TestClient
from sglang.srt.models.registry import ModelRegistry
from transformers import AutoConfig

import sglang_omni.serve.speech_service as speech_service
from sglang_omni.client.client import Client
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
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.utils.imports import import_string
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
            None,
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
    "variant,extra_params,delivery",
    [
        ("base", {"dual_stream": True}, "single"),
        ("speaker", {"dual_stream": "yes"}, "batch"),
        ("speaker", {"style": 3}, "single"),
        ("speaker", [1], "single"),
    ],
)
def test_speech_parameters_rejected_before_generation(
    variant: FireRedTTS3Variant,
    extra_params: dict[str, JsonValue] | list[int],
    delivery: Literal["single", "batch"],
) -> None:
    """Invalid model parameters must return 400 before any backend dispatch."""
    pipeline = FireRedTTS3PipelineConfig(model_path="model", variant=variant)
    backend = mock.create_autospec(Client, instance=True)
    client = TestClient(
        create_app(
            backend,
            model_name="tts",
            speech_request_validator_cls=import_string(
                pipeline.speech_request_validator_factory
            ),
        )
    )
    payload = {"input": "hello", "extra_params": extra_params}
    if delivery == "batch":
        response = client.post("/v1/audio/speech/batch", json={"items": [payload]})
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["status"] == "error"
        assert result["error"]["code"] == 400
        assert result["error"]["param"] == "items.0.extra_params"
    else:
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "extra_params"
    backend.speech.assert_not_called()
    backend.generate.assert_not_called()


@pytest.mark.parametrize("reference_state", ["missing", "oversized"])
def test_reference_audio_rejects_invalid_bare_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_state: Literal["missing", "oversized"],
) -> None:
    """Restoring local audio loading must preserve missing-file and size errors."""
    audio_path = tmp_path / "reference.wav"
    if reference_state == "oversized":
        audio_path.write_bytes(b"RIFF")
        monkeypatch.setattr(speech_service, "MAX_REFERENCE_AUDIO_BYTES", 3)
    else:
        pass
    service = SpeechRequestValidator(default_model="tts")
    with pytest.raises(SpeechAPIError) as failure:
        service.parse_request({"input": "hello", "ref_audio": str(audio_path)})
    assert failure.value.status_code == 400
    assert failure.value.param == "ref_audio"


def test_speaker_rejects_reference_audio() -> None:
    """Speaker-token checkpoints must reject references at the API boundary."""
    pipeline = FireRedTTS3PipelineConfig(model_path="model", variant="speaker")
    service = import_string(pipeline.speech_request_validator_factory)(
        default_model="tts"
    )
    reference = "data:audio/wav;base64," + base64.b64encode(b"RIFF").decode("ascii")
    for fields in (
        {"ref_audio": reference},
        {"references": [{"audio_path": reference}]},
    ):
        with pytest.raises(SpeechAPIError) as failure:
            service.parse_request({"input": "hello", **fields})
        assert failure.value.status_code == 400
        assert "does not accept speech references" in failure.value.message
    service.parse_request({"input": "hello"})


@pytest.mark.parametrize("task_type", ["semantic_edit", "AcousticEdit"])
def test_editing_policy_is_scoped_to_firered(task_type: str) -> None:
    """FireRed editing must not enable empty input or native speed for other models."""
    pipeline = FireRedTTS3PipelineConfig(model_path="model", variant="instruct")
    editing = import_string(pipeline.speech_request_validator_factory)(
        default_model="tts"
    )
    generic = SpeechRequestValidator(default_model="tts")
    payload = {"input": "", "task_type": task_type, "speed": 0.8}
    prepared = editing.parse_generation_request(payload)
    assert prepared.request.task_type in {"SemanticEdit", "AcousticEdit"}
    assert editing.resolve_encode_speed(prepared.request) == 1.0
    with pytest.raises(SpeechAPIError) as failure:
        generic.parse_generation_request(payload)
    assert failure.value.param == "input"
    with pytest.raises(SpeechAPIError) as failure:
        generic.parse_generation_request({**payload, "input": "hello"})
    assert failure.value.param == "task_type"
    prepared = generic.parse_generation_request({"input": "hello", "speed": 0.8})
    assert generic.resolve_encode_speed(prepared.request) == 0.8
