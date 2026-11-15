# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 speech request validation across HTTP and batch delivery."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import sglang_omni.serve.speech_service as speech_service
from sglang_omni.client.client import Client
from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.hf_config import (
    FireRedTTS3Config,
    FireRedTTS3Variant,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.utils.imports import import_string
from sglang_omni.utils.json import JsonValue
from tests.unit_test.fixtures.firered_tts3_preprocessing import preprocess_payload

REFERENCE_AUDIO = "data:audio/wav;base64," + base64.b64encode(b"RIFF").decode("ascii")


@pytest.mark.parametrize(
    "variant,overrides,parameter_path,delivery",
    [
        (
            "base",
            {"stage_params": {"preprocessing": {"dual_stream": True}}},
            "stage_params.preprocessing.dual_stream",
            "single",
        ),
        (
            "speaker",
            {"stage_params": {"preprocessing": {"dual_stream": "yes"}}},
            "stage_params.preprocessing.dual_stream",
            "batch",
        ),
        (
            "speaker",
            {"stage_params": {"preprocessing": {"style": 3}}},
            "stage_params.preprocessing.style",
            "websocket",
        ),
        (
            "speaker",
            {"stage_params": {"talker": {"n_timesteps": -1}}},
            "stage_params.talker.n_timesteps",
            "single",
        ),
        (
            "speaker",
            {"stage_params": {"talker": {"n_timesteps": []}}},
            "stage_params.talker.n_timesteps",
            "batch",
        ),
        (
            "speaker",
            {"stage_params": {"talker": {"inference_cfg": {}}}},
            "stage_params.talker.inference_cfg",
            "single",
        ),
        (
            "speaker",
            {"stage_params": {"talker": {"max_patches": 0}}},
            "stage_params.talker.max_patches",
            "single",
        ),
        (
            "speaker",
            {"stage_params": {"talker": {"min_patches": -1}}},
            "stage_params.talker.min_patches",
            "single",
        ),
        ("instruct", {"task_type": "CustomVoice"}, "task_type", "single"),
        ("base", {"task_type": "VoiceDesign"}, "task_type", "single"),
        ("speaker", {"task_type": "SemanticEdit"}, "task_type", "websocket"),
        (
            "instruct",
            {"task_type": "VoiceDesign", "instructions": " "},
            "instructions",
            "batch",
        ),
        (
            "instruct",
            {"input": "", "task_type": "SemanticEdit"},
            "instructions",
            "single",
        ),
        (
            "instruct",
            {
                "input": "",
                "task_type": "SemanticEdit",
                "instructions": "Replace a word.",
            },
            "ref_audio",
            "single",
        ),
        ("instruct", {"input": "", "task_type": "AcousticEdit"}, "ref_audio", "batch"),
        ("instruct", {}, "ref_audio", "single"),
        ("instruct", {"ref_audio": REFERENCE_AUDIO}, "ref_text", "single"),
        (
            "base",
            {"references": [{"vq_codes": [1], "text": "reference"}]},
            "references",
            "single",
        ),
        (
            "instruct",
            {"task_type": "AcousticEdit", "references": [{"vq_codes": [1]}]},
            "references",
            "single",
        ),
    ],
)
def test_speech_parameters_rejected_before_generation(
    tmp_path: Path,
    variant: FireRedTTS3Variant,
    overrides: dict[str, JsonValue],
    parameter_path: str,
    delivery: Literal["single", "batch", "websocket"],
) -> None:
    """Unsupported tasks, missing fields and invalid stage options must fail before dispatch."""
    pipeline = FireRedTTS3PipelineConfig(model_path=str(tmp_path), variant=variant)
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
    elif delivery == "websocket":
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_json({"type": "session.config", "session": payload})
            result = websocket.receive_json()
        assert result["type"] == "error"
        assert result["param"] == parameter_path
    else:
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["param"] == parameter_path
    backend.speech.assert_not_called()
    backend.generate.assert_not_called()


def test_stage_options_reach_preprocessing_without_overwriting_sampling(
    tmp_path: Path,
) -> None:
    """An unknown HTTP extra_params bag must not overwrite validated sampling or stage fields."""
    pipeline = FireRedTTS3PipelineConfig(model_path=str(tmp_path), variant="speaker")
    service = import_string(pipeline.speech_request_validator_factory)(
        default_model="tts"
    )
    prepared = service.parse_generation_request(
        {
            "input": "hello",
            "temperature": 0.5,
            "stage_params": {
                "talker": {"n_timesteps": "3", "min_patches": 0, "seed": -1},
                "preprocessing": {"dual_stream": True, "style": "calm"},
            },
            "extra_params": {
                "temperature": "not-a-number",
                "stage_params": {"talker": {"n_timesteps": -1}},
                "seed": [],
            },
            "seed": 7,
            "initial_codec_chunk_frames": 8,
        }
    )
    generated = service.build_generate_request(prepared.request, validate=False)
    assert generated.extra_params == {"initial_codec_chunk_frames": 8}
    payload = StagePayload(
        request_id="stage-options",
        request=Client.build_omni_request(generated),
        data={},
    )
    assert payload.request.params["temperature"] == 0.5
    state = preprocess_payload(
        payload,
        config=FireRedTTS3Config(text_latent_interleave_ratio=[2, 1], use_lm_head=True),
    )
    assert state.dual_stream is True
    assert state.n_timesteps == 3
    assert state.min_patches == 0
    assert state.seed == 7

    generated.extra_params["seed"] = 11
    assert Client.build_omni_request(generated).params["seed"] == 11


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


def test_speaker_rejects_reference_audio(tmp_path: Path) -> None:
    """Speaker-token checkpoints must reject references at the API boundary."""
    pipeline = FireRedTTS3PipelineConfig(model_path=str(tmp_path), variant="speaker")
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
def test_editing_policy_is_scoped_to_firered(tmp_path: Path, task_type: str) -> None:
    """FireRed editing must not enable empty input or native speed for other models."""
    pipeline = FireRedTTS3PipelineConfig(model_path=str(tmp_path), variant="instruct")
    editing = import_string(pipeline.speech_request_validator_factory)(
        default_model="tts"
    )
    generic = SpeechRequestValidator(default_model="tts")
    payload = {
        "input": "",
        "task_type": task_type,
        "speed": 0.8,
        "ref_audio": REFERENCE_AUDIO,
        "instructions": "adjust the speed to 0.8x",
    }
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


def test_remote_checkpoint_revision_selects_reference_validation(
    tmp_path: Path,
) -> None:
    """Remote Instruct metadata must permit VoiceDesign without Base references."""
    metadata_path = tmp_path / "config.json"
    metadata_path.write_text(json.dumps({"use_lm_head": True}), encoding="utf-8")
    model_path = "test/checkpoint@requested-revision"
    with mock.patch(
        "sglang_omni.models.firered_tts3.hf_config.hf_hub_download",
        return_value=str(metadata_path),
        create=True,
    ) as download_metadata:
        pipeline = FireRedTTS3PipelineConfig(model_path=model_path)
        assert pipeline.variant == "instruct"
        download_metadata.assert_called_once_with(
            repo_id="test/checkpoint",
            filename="config.json",
            revision="requested-revision",
        )
        with pytest.raises(ValueError, match="pinned variant"):
            FireRedTTS3PipelineConfig(model_path=model_path, variant="base")
    service = import_string(pipeline.speech_request_validator_factory)(
        default_model="tts",
        required_speech_reference_count=pipeline.required_speech_reference_count,
        speech_reference_text_required=pipeline.speech_reference_text_required,
    )
    request = service.parse_request(
        {
            "input": "Hello.",
            "task_type": "VoiceDesign",
            "instructions": "A warm voice.",
        }
    )
    assert request.task_type == "VoiceDesign"
