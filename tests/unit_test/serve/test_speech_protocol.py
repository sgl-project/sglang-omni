# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import ipaddress
from pathlib import Path

import httpx
import pytest

from sglang_omni.client import audio as client_audio
from sglang_omni.preprocessing import resource_connector
from sglang_omni.serve import speech_service
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator


class _MockHTTPConnection:
    def __init__(self, handler) -> None:
        self.client = httpx.Client(transport=httpx.MockTransport(handler))

    def get_sync_client(self) -> httpx.Client:
        return self.client


def _public_test_addresses(hostname: str) -> tuple[ipaddress.IPv4Address, ...]:
    del hostname
    return (ipaddress.ip_address("93.184.216.34"),)


def test_speech_service_rejects_non_string_input() -> None:
    service = SpeechRequestValidator(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": 123})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "input"


@pytest.mark.parametrize("response_format", ["wav", "mp3", "flac", "aac", "opus"])
def test_speech_service_requires_pcm_for_http_streaming(
    response_format: str,
) -> None:
    service = SpeechRequestValidator(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "stream": True, "response_format": response_format}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "response_format"
    assert "stream=true" in exc_info.value.message


def test_speech_service_reports_missing_encoder_dependency_as_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_reason(response_format: str) -> str | None:
        if response_format == "mp3":
            return "mp3 encoder is unavailable"
        return None

    monkeypatch.setattr(
        speech_service,
        "audio_encoding_unavailable_reason",
        unavailable_reason,
    )
    service = SpeechRequestValidator(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "response_format": "mp3"})

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_type == "server_error"
    assert exc_info.value.param == "response_format"


def test_speech_service_accepts_pyav_compressed_encoder_without_pydub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def find_spec(name: str):
        if name == "av":
            return object()
        if name == "pydub":
            return None
        if name == "soundfile":
            return object()
        return object()

    monkeypatch.setattr(client_audio.importlib.util, "find_spec", find_spec)
    client_audio.audio_encoding_unavailable_reason.cache_clear()

    try:
        service = SpeechRequestValidator(default_model="tts")
        request = service.parse_request({"input": "hello", "response_format": "mp3"})
    finally:
        client_audio.audio_encoding_unavailable_reason.cache_clear()

    assert request.response_format == "mp3"


def test_speech_service_rejects_boolean_seed() -> None:
    service = SpeechRequestValidator(default_model="tts")

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
    ],
)
def test_speech_service_rejects_invalid_boundary_values(
    payload: dict[str, object], expected_param: str
) -> None:
    service = SpeechRequestValidator(default_model="tts")

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
    service = SpeechRequestValidator(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", field_name: value})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == field_name


def test_speech_service_normalizes_tts_extension_fields_into_tts_params() -> None:
    service = SpeechRequestValidator(default_model="tts")
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
    service = SpeechRequestValidator(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "file:///tmp/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


@pytest.mark.parametrize(
    ("ref_audio", "expected_param"),
    [
        ("ftp://example.com/reference.wav", "ref_audio"),
        ("data:audio/wav;base64", "ref_audio"),
        ("data:audio/wav,AAAA", "ref_audio"),
        ("data:audio/wav;base64,not@@base64", "ref_audio"),
    ],
)
def test_reference_audio_rejects_unsupported_sources(
    ref_audio: str, expected_param: str
) -> None:
    service = SpeechRequestValidator(default_model="tts")

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": ref_audio})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == expected_param


def test_reference_audio_accepts_valid_base64_data_url() -> None:
    service = SpeechRequestValidator(default_model="tts")
    encoded = base64.b64encode(b"RIFF").decode("ascii")

    gen_req = service.build_generate_request(
        CreateSpeechRequest(
            input="hello",
            ref_audio=f"data:audio/wav;base64,{encoded}",
        )
    )

    assert gen_req.prompt == {
        "text": "hello",
        "references": [{"data": encoded, "media_type": "audio/wav"}],
    }


def test_reference_audio_accepts_allowed_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_media_domains=["example.com"],
    )
    monkeypatch.setattr(
        resource_connector,
        "_resolve_remote_addresses",
        _public_test_addresses,
    )
    service.reference_connector.connection = _MockHTTPConnection(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "audio/wav"},
            content=b"RIFF",
        )
    )

    prepared = service.parse_generation_request(
        {"input": "hello", "ref_audio": "https://example.com/reference.wav"}
    )
    gen_req = service.build_generate_request(
        prepared.request,
        validate=False,
        reference_descriptors=prepared.reference_descriptors,
    )
    encoded = base64.b64encode(b"RIFF").decode("ascii")

    assert prepared.request.ref_audio == f"data:audio/wav;base64,{encoded}"
    assert gen_req.prompt == {
        "text": "hello",
        "references": [{"data": encoded, "media_type": "audio/wav"}],
    }


def test_reference_audio_accepts_public_https_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechRequestValidator(default_model="tts")
    monkeypatch.setattr(
        resource_connector,
        "_resolve_remote_addresses",
        _public_test_addresses,
    )
    service.reference_connector.connection = _MockHTTPConnection(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "audio/wav"},
            content=b"RIFF",
        )
    )

    request = service.parse_request(
        {"input": "hello", "ref_audio": "https://example.com/reference.wav"}
    )

    assert request.ref_audio == "data:audio/wav;base64,UklGRg=="


def test_reference_audio_accepts_local_path_by_default(tmp_path: Path) -> None:
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"RIFF")
    service = SpeechRequestValidator(default_model="tts")

    prepared = service.parse_generation_request(
        {"input": "hello", "ref_audio": str(ref_audio), "ref_text": "reference text"}
    )
    gen_req = service.build_generate_request(
        prepared.request,
        validate=False,
        reference_descriptors=prepared.reference_descriptors,
    )

    assert prepared.request.ref_audio == str(ref_audio.resolve())
    assert gen_req.prompt == {
        "text": "hello",
        "references": [
            {"audio_path": str(ref_audio.resolve()), "text": "reference text"}
        ],
    }


def test_reference_audio_accepts_relative_local_path_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref_audio = tmp_path / "relative" / "reference.wav"
    ref_audio.parent.mkdir()
    ref_audio.write_bytes(b"RIFF")
    monkeypatch.chdir(tmp_path)
    service = SpeechRequestValidator(default_model="tts")

    prepared = service.parse_generation_request(
        {
            "input": "hello",
            "references": [
                {"audio_path": "relative/reference.wav", "text": "reference text"}
            ],
        }
    )
    gen_req = service.build_generate_request(
        prepared.request,
        validate=False,
        reference_descriptors=prepared.reference_descriptors,
    )

    assert prepared.request.references is not None
    assert prepared.request.references[0].audio_path == str(ref_audio.resolve())
    assert gen_req.prompt == {
        "text": "hello",
        "references": [
            {"audio_path": str(ref_audio.resolve()), "text": "reference text"}
        ],
    }


def test_reference_audio_rejects_http_status_with_speech_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_media_domains=["example.com"],
    )
    monkeypatch.setattr(
        resource_connector,
        "_resolve_remote_addresses",
        _public_test_addresses,
    )
    service.reference_connector.connection = _MockHTTPConnection(
        lambda request: httpx.Response(404)
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "https://example.com/missing.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "BadRequestError"
    assert exc_info.value.code == 400
    assert exc_info.value.param == "ref_audio"
    assert "HTTP 404" in exc_info.value.message


def test_reference_audio_honors_allowed_media_domains() -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_media_domains=["allowed.example"],
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "https://blocked.example/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_reference_audio_rejects_private_remote_addresses() -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_media_domains=["127.0.0.1"],
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "https://127.0.0.1/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"
    assert "loopback" in exc_info.value.message


def test_reference_audio_revalidates_redirect_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_media_domains=["allowed.example"],
    )
    monkeypatch.setattr(
        resource_connector,
        "_resolve_remote_addresses",
        _public_test_addresses,
    )
    service.reference_connector.connection = _MockHTTPConnection(
        lambda request: httpx.Response(
            302,
            headers={"location": "https://blocked.example/reference.wav"},
        )
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "https://allowed.example/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_reference_audio_allows_configured_domain_suffix_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_media_domains=["huggingface.co", ".hf.co"],
    )
    monkeypatch.setattr(
        resource_connector,
        "_resolve_remote_addresses",
        _public_test_addresses,
    )
    service.reference_connector.connection = _MockHTTPConnection(
        lambda request: (
            httpx.Response(
                302,
                headers={"location": "https://us.aws.cdn.hf.co/reference.wav"},
            )
            if request.url.host == "huggingface.co"
            else httpx.Response(
                200,
                content=b"RIFF",
                headers={"content-type": "audio/wav"},
            )
        )
    )

    request = service.parse_request(
        {"input": "hello", "ref_audio": "https://huggingface.co/reference.wav"}
    )

    assert request.ref_audio is not None
    assert request.ref_audio.startswith("data:audio/wav;base64,")


def test_reference_audio_revalidates_redirect_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve_addresses(
        hostname: str,
    ) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
        if hostname == "private.example":
            return (ipaddress.ip_address("10.0.0.1"),)
        return _public_test_addresses(hostname)

    service = SpeechRequestValidator(
        default_model="tts",
        allowed_media_domains=["allowed.example", "private.example"],
    )
    monkeypatch.setattr(
        resource_connector,
        "_resolve_remote_addresses",
        resolve_addresses,
    )
    service.reference_connector.connection = _MockHTTPConnection(
        lambda request: httpx.Response(
            302,
            headers={"location": "https://private.example/reference.wav"},
        )
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "https://allowed.example/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"
    assert "private" in exc_info.value.message


def test_reference_audio_rejects_oversized_https_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_media_domains=["example.com"],
    )
    monkeypatch.setattr(
        resource_connector,
        "_resolve_remote_addresses",
        _public_test_addresses,
    )
    service.reference_connector.connection = _MockHTTPConnection(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "audio/wav"},
            content=b"RIFF",
        )
    )
    monkeypatch.setattr(speech_service, "MAX_REFERENCE_AUDIO_BYTES", 3)

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "https://example.com/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_reference_audio_rejects_oversized_data_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SpeechRequestValidator(default_model="tts")
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
    service = SpeechRequestValidator(default_model="tts")
    monkeypatch.setattr(speech_service, "MAX_SPEECH_INPUT_CHARS", 5)

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello world"})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "input"


def test_reference_list_accepts_raw_local_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    service = SpeechRequestValidator(default_model="tts")

    request = service.parse_request(
        {
            "input": "hello",
            "references": [{"audio_path": str(audio_path)}],
        }
    )
    gen_req = service.build_generate_request(request, validate=False)

    assert gen_req.prompt == {
        "text": "hello",
        "references": [{"audio_path": str(audio_path.resolve())}],
    }


def test_reference_list_rejects_invalid_base64_data_url() -> None:
    service = SpeechRequestValidator(default_model="tts")

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
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=tmp_path,
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
    service = SpeechRequestValidator(default_model="tts")
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
        SpeechRequestValidator(default_model="tts", allowed_local_media_path=missing)


def test_file_reference_resolves_inside_allowlist(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=tmp_path,
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


def test_relative_file_reference_resolves_inside_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.chdir(tmp_path)
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=tmp_path,
    )

    request = service.parse_request({"input": "hello", "ref_audio": "reference.wav"})

    assert request.ref_audio == str(audio_path.resolve())


def test_file_reference_accepts_localhost_file_url(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=tmp_path,
    )

    request = service.parse_request(
        {"input": "hello", "ref_audio": f"file://localhost{audio_path}"}
    )

    assert request.ref_audio == str(audio_path.resolve())


def test_file_reference_rejects_remote_file_netloc(tmp_path: Path) -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=tmp_path,
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request(
            {"input": "hello", "ref_audio": "file://remotehost/reference.wav"}
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_file_reference_rejects_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setattr(speech_service, "MAX_REFERENCE_AUDIO_BYTES", 3)
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=tmp_path,
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": audio_path.as_uri()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_file_reference_rejects_missing_file_inside_allowlist(tmp_path: Path) -> None:
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=tmp_path,
    )
    missing_path = tmp_path / "missing.wav"

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": missing_path.as_uri()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"


def test_file_reference_rejects_directory_inside_allowlist(tmp_path: Path) -> None:
    audio_dir = tmp_path / "reference-dir"
    audio_dir.mkdir()
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=tmp_path,
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
    service = SpeechRequestValidator(
        default_model="tts",
        allowed_local_media_path=allowed,
    )

    with pytest.raises(SpeechAPIError) as exc_info:
        service.parse_request({"input": "hello", "ref_audio": link.as_uri()})

    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "ref_audio"
