# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from sglang_omni.serve import speech_service
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechService


def test_speech_service_rejects_non_string_input() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": 123})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "input"


def test_speech_service_requires_pcm_for_http_streaming() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "stream": True, "response_format": "wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "response_format"


def test_speech_service_validates_stream_format_before_encoder_dependency() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "stream": True, "response_format": "mp3"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "response_format"
    assert "stream=true" in exc_info.value.message


def test_speech_service_rejects_boolean_seed() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "seed": True})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "seed"


@pytest.mark.parametrize(
    ("payload", "expected_param"),
    [
        ({"input": "hello", "response_format": "gif"}, "response_format"),
        ({"input": "hello", "speed": 0.24}, "speed"),
        ({"input": "hello", "speed": 4.01}, "speed"),
        ({"input": "hello", "language": "Klingon"}, "language"),
        ({"input": "hello", "task_type": "Narration"}, "task_type"),
        ({"input": "hello", "max_new_tokens": 0}, "max_new_tokens"),
        ({"input": "hello", "token_count": 0}, "token_count"),
        ({"input": "hello", "token_count": -1}, "token_count"),
        ({"input": "hello", "duration_tokens": 0}, "duration_tokens"),
        ({"input": "hello", "duration_tokens": -1}, "duration_tokens"),
        (
            {"input": "hello", "initial_codec_chunk_frames": 0},
            "initial_codec_chunk_frames",
        ),
    ],
)
def test_speech_service_rejects_invalid_boundary_values(
    payload: dict[str, object], expected_param: str
) -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == expected_param


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("token_count", "5"),
        ("token_count", True),
        ("duration_tokens", "5"),
        ("duration_tokens", True),
    ],
)
def test_speech_service_rejects_invalid_duration_field_types(
    field_name: str, value: object
) -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", field_name: value})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == field_name


def test_speech_service_normalizes_tts_extension_fields_into_tts_params() -> None:
    service = SpeechService(default_model="tts")
    request = CreateSpeechRequest.model_validate(
        {
            "input": "hello",
            "speaker": "alloy",
            "response_format": "WAV",
            "task_type": "voice-design",
            "language": "english",
            "instructions": "calm",
            "x_vector_only_mode": True,
            "initial_codec_chunk_frames": 8,
            "max_new_tokens": 128,
        }
    )

    gen_req = service.build_generate_request(request)
    tts_params = gen_req.metadata["tts_params"]

    assert gen_req.model == "tts"
    assert tts_params["voice"] == "alloy"
    assert tts_params["response_format"] == "wav"
    assert tts_params["task_type"] == "VoiceDesign"
    assert tts_params["language"] == "English"
    assert tts_params["instructions"] == "calm"
    assert tts_params["x_vector_only_mode"] is True
    assert tts_params["initial_codec_chunk_frames"] == 8
    assert gen_req.extra_params == {"initial_codec_chunk_frames": 8}
    assert gen_req.sampling.max_new_tokens == 128
    assert tts_params["explicit_generation_params"] == ["max_new_tokens"]


def test_file_reference_requires_allowlist() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "file:///tmp/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


@pytest.mark.parametrize(
    ("ref_audio", "expected_param"),
    [
        ("/tmp/reference.wav", "ref_audio"),
        ("relative/reference.wav", "ref_audio"),
        ("ftp://example.com/reference.wav", "ref_audio"),
        ("data:audio/wav;base64", "ref_audio"),
        ("data:audio/wav,AAAA", "ref_audio"),
        ("data:audio/wav;base64,not@@base64", "ref_audio"),
        ("https://example.com/reference.wav", "ref_audio"),
    ],
)
def test_reference_audio_rejects_unsupported_sources(
    ref_audio: str, expected_param: str
) -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": ref_audio})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == expected_param


def test_reference_audio_accepts_valid_base64_data_url() -> None:
    service = SpeechService(default_model="tts")
    encoded = base64.b64encode(b"RIFF").decode("ascii")

    request = service.parse_request(
        {"input": "hello", "ref_audio": f"data:audio/wav;base64,{encoded}"}
    )
    gen_req = service.build_generate_request(request, validate=False)

    assert request.ref_audio == f"data:audio/wav;base64,{encoded}"
    assert gen_req.prompt == {
        "text": "hello",
        "references": [{"data": encoded, "media_type": "audio/wav"}],
    }


def test_reference_audio_rejects_oversized_data_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechService(default_model="tts")
    monkeypatch.setattr(speech_service, "MAX_REFERENCE_AUDIO_BYTES", 3)
    encoded = base64.b64encode(b"RIFF").decode("ascii")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": f"data:audio/wav;base64,{encoded}"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_speech_service_rejects_oversized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechService(default_model="tts")
    monkeypatch.setattr(speech_service, "MAX_SPEECH_INPUT_CHARS", 5)

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello world"})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "input"


def test_reference_list_rejects_raw_local_path() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {
                "input": "hello",
                "references": [{"audio_path": "/tmp/reference.wav"}],
            }
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "references.audio_path"


def test_reference_list_rejects_invalid_base64_data_url() -> None:
    service = SpeechService(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {
                "input": "hello",
                "references": [{"audio_path": "data:audio/wav;base64,not@@base64"}],
            }
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "references.audio_path"


@pytest.mark.parametrize("field_name", ["audio_path", "ref_audio", "audio"])
def test_reference_list_canonicalizes_audio_aliases(
    tmp_path: Path, field_name: str
) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    service = SpeechService(
        default_model="tts",
        allowed_local_media_paths=[tmp_path],
    )

    request = service.parse_request(
        {
            "input": "hello",
            "references": [{field_name: audio_path.as_uri(), "text": "reference"}],
        }
    )
    gen_req = service.build_generate_request(request, validate=False)

    assert gen_req.prompt == {
        "text": "hello",
        "references": [{"audio_path": str(audio_path.resolve()), "text": "reference"}],
    }


def test_reference_list_canonicalizes_data_url() -> None:
    service = SpeechService(default_model="tts")
    encoded = base64.b64encode(b"RIFF").decode("ascii")

    request = service.parse_request(
        {
            "input": "hello",
            "references": [
                {
                    "audio": f"data:audio/wav;base64,{encoded}",
                    "text": "reference",
                }
            ],
        }
    )
    gen_req = service.build_generate_request(request, validate=False)

    assert gen_req.prompt == {
        "text": "hello",
        "references": [
            {"data": encoded, "media_type": "audio/wav", "text": "reference"}
        ],
    }


def test_allowed_local_media_path_must_be_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(
        ValueError, match="allowed local media path must be a directory"
    ):
        SpeechService(default_model="tts", allowed_local_media_paths=[missing])


def test_file_reference_resolves_inside_allowlist(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    service = SpeechService(
        default_model="tts",
        allowed_local_media_paths=[tmp_path],
    )

    request = service.parse_request(
        {"input": "hello", "ref_audio": audio_path.as_uri()}
    )
    gen_req = service.build_generate_request(request, validate=False)

    assert request.ref_audio == str(audio_path.resolve())
    assert gen_req.prompt == {
        "text": "hello",
        "references": [{"audio_path": str(audio_path.resolve())}],
    }

    prepared_again = service.prepare_request(request)
    assert prepared_again.ref_audio == str(audio_path.resolve())


def test_file_reference_rejects_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setattr(speech_service, "MAX_REFERENCE_AUDIO_BYTES", 3)
    service = SpeechService(
        default_model="tts",
        allowed_local_media_paths=[tmp_path],
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": audio_path.as_uri()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_file_reference_rejects_missing_file_inside_allowlist(tmp_path: Path) -> None:
    service = SpeechService(
        default_model="tts",
        allowed_local_media_paths=[tmp_path],
    )
    missing_path = tmp_path / "missing.wav"

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": missing_path.as_uri()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_file_reference_rejects_directory_inside_allowlist(tmp_path: Path) -> None:
    audio_dir = tmp_path / "reference-dir"
    audio_dir.mkdir()
    service = SpeechService(
        default_model="tts",
        allowed_local_media_paths=[tmp_path],
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": audio_dir.as_uri()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_file_reference_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside.wav"
    allowed.mkdir()
    outside.write_bytes(b"RIFF")
    link = allowed / "escape.wav"
    link.symlink_to(outside)
    service = SpeechService(
        default_model="tts",
        allowed_local_media_paths=[allowed],
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": link.as_uri()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"
