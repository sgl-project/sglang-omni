# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.audio import DEFAULT_SAMPLE_RATE, encode_wav
from sglang_omni.scheduling.speaker_cache import SpeakerArtifactCache, SpeakerCacheKey
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import VoiceUploadBodyLimitMiddleware
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.serve.speech_voices import SpeakerSampleStore


class RecordingSpeechClient:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def speech(self, request: Any, **_: Any) -> Any:
        from sglang_omni.client.types import SpeechResult

        self.requests.append(request)
        return SpeechResult(
            audio_bytes=b"RIFF",
            mime_type="audio/wav",
            format="wav",
        )


@pytest.mark.asyncio
async def test_voice_upload_body_limit_rejects_before_endpoint() -> None:
    async def unreachable_app(scope, receive, send) -> None:
        raise AssertionError("oversized body reached downstream app")

    middleware = VoiceUploadBodyLimitMiddleware(unreachable_app, max_bytes=8)
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/voices",
            "headers": [(b"content-length", b"9")],
        },
        receive,
        send,
    )

    assert messages[0]["status"] == 413


@pytest.mark.asyncio
async def test_voice_upload_body_limit_rejects_chunked_body() -> None:
    async def downstream_app(scope, receive, send) -> None:
        while True:
            message = await receive()
            if not message.get("more_body", False):
                return

    middleware = VoiceUploadBodyLimitMiddleware(downstream_app, max_bytes=4)
    messages: list[dict[str, Any]] = []
    chunks = iter(
        (
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"de", "more_body": False},
        )
    )

    async def receive() -> dict[str, Any]:
        return next(chunks)

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/voices",
            "headers": [],
        },
        receive,
        send,
    )

    assert messages[0]["status"] == 413


def test_voice_routes_upload_list_use_and_delete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path))
    client_impl = RecordingSpeechClient()
    client = TestClient(create_app(client_impl, model_name="tts"))

    upload = client.post(
        "/v1/audio/voices",
        data={
            "name": "Narrator_01",
            "consent": "consent-123",
            "ref_text": "The narrator reference transcript.",
            "speaker_description": "clear narration voice",
        },
        files={
            "audio_sample": (
                "reference.wav",
                _reference_wav(),
                "audio/wav",
            )
        },
    )

    assert upload.status_code == 200
    assert upload.json()["name"] == "Narrator_01"
    assert (tmp_path / "narrator_01.safetensors").exists()

    listed = client.get("/v1/audio/voices")
    assert listed.status_code == 200
    assert listed.json()["voices"] == ["default", "Narrator_01"]
    assert listed.json()["uploaded_voices"][0]["ref_text"] == (
        "The narrator reference transcript."
    )

    speech = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "narrator_01", "response_format": "wav"},
    )
    assert speech.status_code == 200
    prompt = client_impl.requests[-1].prompt
    assert prompt["references"][0]["audio_path"].startswith("data:audio/wav;base64,")
    assert prompt["references"][0]["text"] == "The narrator reference transcript."

    deleted = client.delete("/v1/audio/voices/Narrator_01")
    assert deleted.status_code == 200
    assert deleted.json()["success"] is True
    assert not (tmp_path / "narrator_01.safetensors").exists()

    missing = client.delete("/v1/audio/voices/Narrator_01")
    assert missing.status_code == 404
    assert missing.json() == {
        "success": False,
        "error": "Voice 'Narrator_01' not found",
    }


def test_voice_store_restores_overwrites_and_invalidates_cache(tmp_path: Path) -> None:
    cache = SpeakerArtifactCache()
    store = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=2, cache=cache)
    first = store.upload(
        name="Guide",
        consent="consent-a",
        audio_bytes=_reference_wav(frequency=220),
        filename="guide.wav",
        content_type="audio/wav",
    )
    key = SpeakerCacheKey("higgs", "guide", first["created_at"], "ref_codes")
    cache.put(key, np.arange(8, dtype=np.float32))

    second = store.upload(
        name="guide",
        consent="consent-b",
        audio_bytes=_reference_wav(frequency=330),
        filename="guide.wav",
        content_type="audio/wav",
        ref_text="new transcript",
    )

    assert second["warning"] == "Voice 'guide' overwritten"
    assert cache.get(key) is None
    restored = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=2, cache=cache)
    voices = restored.list_response()["uploaded_voices"]
    assert voices == [
        {
            "name": "guide",
            "consent": "consent-b",
            "created_at": second["created_at"],
            "file_size": second["file_size"],
            "mime_type": "audio/wav",
            "ref_text": "new transcript",
        }
    ]


def test_voice_store_enforces_upload_contracts(tmp_path: Path) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=1)

    for name in ("../bad", ".", "..", "-"):
        with pytest.raises(SpeechAPIError, match="name must contain"):
            store.upload(
                name=name,
                consent="consent",
                audio_bytes=_reference_wav(),
                filename="bad.wav",
                content_type="audio/wav",
            )

    for name in ("default", "Default"):
        with pytest.raises(SpeechAPIError, match="reserved"):
            store.upload(
                name=name,
                consent="consent",
                audio_bytes=_reference_wav(),
                filename="default.wav",
                content_type="audio/wav",
            )

    with pytest.raises(SpeechAPIError, match="must not be empty"):
        store.upload(
            name="empty",
            consent="consent",
            audio_bytes=b"",
            filename="empty.wav",
            content_type="audio/wav",
        )

    with pytest.raises(SpeechAPIError, match="MIME type"):
        store.upload(
            name="text",
            consent="consent",
            audio_bytes=b"not audio",
            filename="text.txt",
            content_type="text/plain",
        )

    with pytest.raises(SpeechAPIError, match="non-silent"):
        store.upload(
            name="silent",
            consent="consent",
            audio_bytes=_reference_wav(amplitude=0.0),
            filename="silent.wav",
            content_type="audio/wav",
        )

    with pytest.raises(SpeechAPIError, match="at least 1.0s"):
        store.upload(
            name="short",
            consent="consent",
            audio_bytes=_reference_wav(duration_s=0.25),
            filename="short.wav",
            content_type="audio/wav",
        )

    with pytest.raises(SpeechAPIError, match="at most 30.0s"):
        store.upload(
            name="long",
            consent="consent",
            audio_bytes=_reference_wav(duration_s=30.1),
            filename="long.wav",
            content_type="audio/wav",
        )

    store.upload(
        name="one",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="one.wav",
        content_type="audio/wav",
    )
    with pytest.raises(SpeechAPIError, match="Uploaded voice limit reached"):
        store.upload(
            name="two",
            consent="consent",
            audio_bytes=_reference_wav(),
            filename="two.wav",
            content_type="audio/wav",
        )
    assert sorted(path.name for path in tmp_path.glob("*.safetensors")) == [
        "one.safetensors"
    ]
    assert list(tmp_path.glob("*.tmp")) == []


def test_voice_store_restore_preserves_max_uploaded_cap(tmp_path: Path) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=2)
    first = store.upload(
        name="older",
        consent="consent",
        audio_bytes=_reference_wav(frequency=220),
        filename="older.wav",
        content_type="audio/wav",
    )
    second = store.upload(
        name="newer",
        consent="consent",
        audio_bytes=_reference_wav(frequency=330),
        filename="newer.wav",
        content_type="audio/wav",
    )

    restored = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=1)
    voices = restored.list_response()["uploaded_voices"]

    assert [voice["name"] for voice in voices] == ["newer"]
    assert voices[0]["created_at"] == second["created_at"]
    assert second["created_at"] > first["created_at"]


def test_voice_store_restore_keeps_newest_duplicate_normalized_name(
    tmp_path: Path,
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=2)
    uploaded = store.upload(
        name="Guide",
        consent="consent-new",
        audio_bytes=_reference_wav(frequency=330),
        filename="guide.wav",
        content_type="audio/wav",
    )
    duplicate_path = tmp_path / "manual_duplicate.safetensors"

    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file

    voice_path = tmp_path / "guide.safetensors"
    with safe_open(str(voice_path), framework="np") as handle:
        metadata = dict(handle.metadata() or {})
    duplicate_metadata = {
        **metadata,
        "name": "Guide old",
        "created_at": str(uploaded["created_at"] - 1),
    }
    save_file(
        load_file(str(voice_path)),
        str(duplicate_path),
        metadata=duplicate_metadata,
    )

    restored = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=2)
    voices = restored.list_response()["uploaded_voices"]

    assert [voice["name"] for voice in voices] == ["Guide"]
    assert voices[0]["created_at"] == uploaded["created_at"]


def test_voice_store_restore_skips_malformed_metadata(tmp_path: Path) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=2)
    uploaded = store.upload(
        name="Guide",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="guide.wav",
        content_type="audio/wav",
    )

    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file

    voice_path = tmp_path / "guide.safetensors"
    bad_path = tmp_path / "bad.safetensors"
    with safe_open(str(voice_path), framework="np") as handle:
        metadata = dict(handle.metadata() or {})
    save_file(
        load_file(str(voice_path)),
        str(bad_path),
        metadata={
            **metadata,
            "name": "bad",
            "normalized_name": "bad",
            "created_at": "not-an-int",
        },
    )

    restored = SpeakerSampleStore(root_dir=tmp_path, max_uploaded=2)
    voices = restored.list_response()["uploaded_voices"]

    assert [voice["name"] for voice in voices] == ["Guide"]
    assert voices[0]["created_at"] == uploaded["created_at"]


def test_voice_store_decode_error_uses_stable_client_message(tmp_path: Path) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)

    with pytest.raises(SpeechAPIError) as exc_info:
        store.upload(
            name="broken",
            consent="consent",
            audio_bytes=b"not an audio file",
            filename="broken.wav",
            content_type="audio/wav",
        )

    assert exc_info.value.message == (
        "audio_sample could not be decoded as a supported audio format"
    )
    assert exc_info.value.param == "audio_sample"


def test_speech_service_resolves_uploaded_voice_to_reference(tmp_path: Path) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    uploaded = store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="anchor.wav",
        content_type="application/octet-stream",
    )
    service = SpeechRequestValidator(default_model="tts", voice_store=store)

    request = service.parse_request({"input": "hello", "voice": "ANCHOR"})
    gen_req = service.build_generate_request(request)
    tts_params = gen_req.metadata["tts_params"]

    assert gen_req.prompt["references"][0]["audio_path"].startswith(
        "data:audio/wav;base64,"
    )
    assert gen_req.prompt["references"][0]["uploaded_voice_name"] == "anchor"
    assert (
        gen_req.prompt["references"][0]["uploaded_voice_created_at"]
        == uploaded["created_at"]
    )
    assert tts_params["task_type"] == "Base"
    assert tts_params["uploaded_voice_name"] == "anchor"
    assert tts_params["uploaded_voice_created_at"] == uploaded["created_at"]


def test_speech_service_explicit_reference_overrides_uploaded_voice(
    tmp_path: Path,
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="anchor.wav",
        content_type="audio/wav",
    )
    service = SpeechRequestValidator(default_model="tts", voice_store=store)
    explicit_ref = "data:audio/wav;base64," + base64.b64encode(
        _reference_wav(frequency=880)
    ).decode("ascii")

    request = service.parse_request(
        {
            "input": "hello",
            "voice": "anchor",
            "ref_audio": explicit_ref,
            "response_format": "wav",
        }
    )
    gen_req = service.build_generate_request(request)

    ref = gen_req.prompt["references"][0]
    assert "uploaded_voice_name" not in ref
    assert gen_req.metadata["tts_params"]["voice"] == "anchor"
    assert "uploaded_voice_name" not in gen_req.metadata["tts_params"]


@pytest.mark.parametrize("task_type", ["CustomVoice", "VoiceDesign"])
def test_speech_service_rejects_uploaded_voice_with_non_base_task_type(
    tmp_path: Path,
    task_type: str,
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="anchor.wav",
        content_type="audio/wav",
    )
    service = SpeechRequestValidator(default_model="tts", voice_store=store)

    with pytest.raises(SpeechAPIError, match="task_type='Base'") as exc_info:
        service.parse_request(
            {"input": "hello", "voice": "anchor", "task_type": task_type}
        )
    assert exc_info.value.param == "task_type"


def test_speech_service_allows_uploaded_voice_with_explicit_base_task_type(
    tmp_path: Path,
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="anchor.wav",
        content_type="audio/wav",
    )
    service = SpeechRequestValidator(default_model="tts", voice_store=store)

    request = service.parse_request(
        {"input": "hello", "voice": "anchor", "task_type": "Base"}
    )
    gen_req = service.build_generate_request(request, validate=False)

    assert gen_req.metadata["tts_params"]["task_type"] == "Base"


def test_speech_service_uses_same_uploaded_voice_resolution_for_prompt_and_params(
    tmp_path: Path,
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    first = store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=_reference_wav(frequency=220),
        filename="anchor.wav",
        content_type="audio/wav",
    )
    service = SpeechRequestValidator(default_model="tts", voice_store=store)

    prepared = service.parse_generation_request({"input": "hello", "voice": "anchor"})
    store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=_reference_wav(frequency=330),
        filename="anchor.wav",
        content_type="audio/wav",
    )
    gen_req = service.build_generate_request(
        prepared.request,
        validate=False,
        reference_descriptors=prepared.reference_descriptors,
        uploaded_voice=prepared.uploaded_voice,
    )

    ref = gen_req.prompt["references"][0]
    tts_params = gen_req.metadata["tts_params"]
    assert ref["uploaded_voice_created_at"] == first["created_at"]
    assert tts_params["uploaded_voice_created_at"] == first["created_at"]


def test_speech_service_rejects_unknown_required_uploaded_voice(
    tmp_path: Path,
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    service = SpeechRequestValidator(
        default_model="public-tts-name",
        requires_uploaded_voice_for_named_voice=True,
        voice_store=store,
    )

    with pytest.raises(SpeechAPIError, match="Unknown voice"):
        service.parse_request({"input": "hello", "voice": "missing"})


def test_speech_service_rejects_batch_default_uploaded_voice_task_type(
    tmp_path: Path,
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="anchor.wav",
        content_type="audio/wav",
    )
    service = SpeechRequestValidator(
        default_model="public-tts-name",
        voice_store=store,
    )

    with pytest.raises(SpeechAPIError, match="uploaded voice requests require"):
        service.parse_batch_request(
            {
                "voice": "Anchor",
                "task_type": "VoiceDesign",
                "items": [{"input": "hello"}],
            }
        )


def test_speech_service_preserves_preset_voice_names(tmp_path: Path) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    service = SpeechRequestValidator(
        default_model="preset-voice-model",
        voice_store=store,
    )

    request = service.parse_request({"input": "hello", "voice": "Vivian"})
    gen_req = service.build_generate_request(request, validate=False)

    assert gen_req.prompt == "hello"
    assert gen_req.metadata["tts_params"]["voice"] == "Vivian"


def test_speech_service_can_disable_uploaded_voice_resolution(
    tmp_path: Path,
) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    store.upload(
        name="Vivian",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="vivian.wav",
        content_type="audio/wav",
    )
    service = SpeechRequestValidator(
        default_model="qwen3-customvoice",
        supports_uploaded_voice_references=False,
        voice_store=store,
    )

    request = service.parse_request({"input": "hello", "voice": "Vivian"})
    gen_req = service.build_generate_request(request)

    assert gen_req.prompt == "hello"
    assert gen_req.metadata["tts_params"]["voice"] == "Vivian"
    assert "uploaded_voice_name" not in gen_req.metadata["tts_params"]


def _reference_wav(
    *,
    duration_s: float = 1.2,
    frequency: float = 440.0,
    amplitude: float = 0.2,
) -> bytes:
    sample_count = int(DEFAULT_SAMPLE_RATE * duration_s)
    t = np.arange(sample_count, dtype=np.float32) / DEFAULT_SAMPLE_RATE
    audio = amplitude * np.sin(2.0 * np.pi * frequency * t)
    return encode_wav(audio.astype(np.float32), DEFAULT_SAMPLE_RATE)
