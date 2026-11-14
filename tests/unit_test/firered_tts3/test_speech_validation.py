# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 speech request validation across HTTP and batch delivery."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import sglang_omni.serve.speech_service as speech_service
from sglang_omni.client.client import Client
from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Variant
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.utils.imports import import_string
from sglang_omni.utils.json import JsonValue


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


@pytest.mark.parametrize(
    "overrides,parameter_path,delivery",
    [
        (
            {"stage_params": {"talker": {"n_timesteps": -1}}},
            "stage_params.talker.n_timesteps",
            "single",
        ),
        ({"extra_params": {"n_timesteps": 0}}, "extra_params.n_timesteps", "single"),
        (
            {"extra_params": {"stage_params": {"talker": {"n_timesteps": []}}}},
            "extra_params.stage_params.talker.n_timesteps",
            "batch",
        ),
        (
            {"extra_params": {"inference_cfg": {}}},
            "extra_params.inference_cfg",
            "single",
        ),
        ({"extra_params": {"max_patches": -1}}, "extra_params.max_patches", "single"),
        ({"extra_params": {"min_patches": -1}}, "extra_params.min_patches", "single"),
    ],
)
def test_invalid_generation_overrides_are_rejected_at_http_boundary(
    overrides: dict[str, JsonValue],
    parameter_path: str,
    delivery: Literal["single", "batch"],
) -> None:
    """Invalid Flow requests must fail before launching model inference."""
    pipeline = FireRedTTS3PipelineConfig(model_path="model", variant="speaker")
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
    payload = {"input": "hello", **overrides}
    if delivery == "batch":
        response = client.post("/v1/audio/speech/batch", json={"items": [payload]})
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["status"] == "error"
        assert result["error"]["code"] == 400
        assert result["error"]["param"] == f"items.0.{parameter_path}"
    else:
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["param"] == parameter_path
    backend.speech.assert_not_called()
    backend.generate.assert_not_called()


def test_http_generation_override_validation_uses_preprocessing_precedence() -> None:
    pipeline = FireRedTTS3PipelineConfig(model_path="model", variant="speaker")
    service = import_string(pipeline.speech_request_validator_factory)(
        default_model="tts"
    )
    prepared = service.parse_generation_request(
        {
            "input": "hello",
            "stage_params": {"talker": {"n_timesteps": -1}},
            "extra_params": {
                "n_timesteps": -1,
                "stage_params": {"talker": {"n_timesteps": "3"}},
                "seed": [],
            },
            "seed": 7,
        }
    )
    generated = service.build_generate_request(prepared.request, validate=False)
    assert generated.extra_params["stage_params"] == {"talker": {"n_timesteps": "3"}}
    assert generated.metadata["tts_params"]["seed"] == 7


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
