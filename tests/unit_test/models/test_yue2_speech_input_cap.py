# SPDX-License-Identifier: Apache-2.0
"""YuE2 speech input length contract: full-song lyrics exceed the generic cap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang_omni.config.schema import MAX_SPEECH_INPUT_CHARS, PipelineConfig
from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.yue2.config import Yue2PipelineConfig
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "wsb_prompt164_input.json"


def prompt164_input() -> str:
    payload = json.loads(FIXTURE.read_text())
    return json.dumps(payload["nested"], ensure_ascii=False)


def test_prompt164_fixture_exceeds_generic_cap() -> None:
    # official WildSongBench prompt 164: 4114 serialized chars vs 4096 cap
    assert len(prompt164_input()) == 4114
    assert len(prompt164_input()) > MAX_SPEECH_INPUT_CHARS


def test_generic_validator_rejects_prompt164_payload() -> None:
    service = SpeechRequestValidator(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": prompt164_input()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "input"


def test_yue2_limit_accepts_prompt164_payload() -> None:
    service = SpeechRequestValidator(
        default_model="yue2",
        max_speech_input_chars=Yue2PipelineConfig.max_speech_input_chars,
    )

    request = service.parse_request({"input": prompt164_input()})

    assert len(request.input) == 4114


def test_yue2_limit_accepts_synthetic_long_input() -> None:
    service = SpeechRequestValidator(
        default_model="yue2",
        max_speech_input_chars=Yue2PipelineConfig.max_speech_input_chars,
    )

    request = service.parse_request({"input": "la " * 4096})

    assert len(request.input) == 12288


def test_yue2_limit_is_none_like_moss_tts() -> None:
    assert Yue2PipelineConfig.max_speech_input_chars is None
    assert MossTTSPipelineConfig.max_speech_input_chars is None


def test_generic_pipeline_default_cap_unchanged() -> None:
    assert PipelineConfig.max_speech_input_chars == MAX_SPEECH_INPUT_CHARS
    assert MAX_SPEECH_INPUT_CHARS == 4096


def test_yue2_short_input_still_valid() -> None:
    service = SpeechRequestValidator(
        default_model="yue2",
        max_speech_input_chars=Yue2PipelineConfig.max_speech_input_chars,
    )

    request = service.parse_request({"input": "hello"})

    assert request.input == "hello"


def test_empty_input_still_rejected_under_yue2_limit() -> None:
    service = SpeechRequestValidator(
        default_model="yue2",
        max_speech_input_chars=Yue2PipelineConfig.max_speech_input_chars,
    )

    for bad in ("", "   "):
        with pytest.raises(SpeechAPIError) as exc_info:
            service.parse_request({"input": bad})

        assert exc_info.value.status_code == 400
        assert exc_info.value.param == "input"


def test_non_string_input_still_rejected() -> None:
    service = SpeechRequestValidator(default_model="yue2")

    with pytest.raises(SpeechAPIError):
        service.parse_request({"input": 12345})


def test_invalid_limit_values_still_rejected() -> None:
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer or None"):
            SpeechRequestValidator(
                default_model="yue2",
                max_speech_input_chars=bad,
            )


def test_yue2_config_contract_unchanged_beyond_cap() -> None:
    assert Yue2PipelineConfig.architecture == "YuE2ForCausalLM"
    assert Yue2PipelineConfig.requires_model_capabilities is True
    assert set(Yue2PipelineConfig.stage_config_types) == {"yue2_synth"}


def test_speech_endpoint_serves_prompt164_under_yue2_limit() -> None:
    from fastapi.testclient import TestClient

    from sglang_omni.serve import create_app
    from tests.unit_test.serve.test_openai_api import SuccessfulSpeechClient

    app = create_app(
        SuccessfulSpeechClient(),
        model_name="yue2",
        max_speech_input_chars=Yue2PipelineConfig.max_speech_input_chars,
    )

    response = TestClient(app).post(
        "/v1/audio/speech",
        json={
            "model": "yue2",
            "input": prompt164_input(),
            "voice": "default",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.content == b"RIFF"


def test_speech_endpoint_rejects_prompt164_under_generic_limit() -> None:
    from fastapi.testclient import TestClient

    from sglang_omni.serve import create_app
    from tests.unit_test.serve.test_openai_api import SuccessfulSpeechClient

    app = create_app(
        SuccessfulSpeechClient(),
        model_name="yue2",
        max_speech_input_chars=PipelineConfig.max_speech_input_chars,
    )

    response = TestClient(app).post(
        "/v1/audio/speech",
        json={
            "model": "yue2",
            "input": prompt164_input(),
            "voice": "default",
            "response_format": "wav",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "input"
