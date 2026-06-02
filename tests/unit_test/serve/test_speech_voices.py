# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.audio import DEFAULT_SAMPLE_RATE, encode_wav
from sglang_omni.serve import create_app
from sglang_omni.serve.speaker_cache import SpeakerArtifactCache, SpeakerCacheKey
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechService
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

    with pytest.raises(SpeechAPIError, match="name must contain"):
        store.upload(
            name="../bad",
            consent="consent",
            audio_bytes=_reference_wav(),
            filename="bad.wav",
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


def test_speech_service_resolves_uploaded_voice_to_reference(tmp_path: Path) -> None:
    store = SpeakerSampleStore(root_dir=tmp_path)
    uploaded = store.upload(
        name="Anchor",
        consent="consent",
        audio_bytes=_reference_wav(),
        filename="anchor.wav",
        content_type="application/octet-stream",
    )
    service = SpeechService(default_model="tts", voice_store=store)

    request = service.parse_request({"input": "hello", "voice": "ANCHOR"})
    gen_req = service.build_generate_request(request, validate=False)
    tts_params = gen_req.metadata["tts_params"]

    assert gen_req.prompt["references"][0]["audio_path"].startswith(
        "data:audio/wav;base64,"
    )
    assert tts_params["task_type"] == "Base"
    assert tts_params["uploaded_voice_name"] == "anchor"
    assert tts_params["uploaded_voice_created_at"] == uploaded["created_at"]


def _reference_wav(
    *,
    duration_s: float = 1.2,
    frequency: float = 440.0,
) -> bytes:
    sample_count = int(DEFAULT_SAMPLE_RATE * duration_s)
    t = np.arange(sample_count, dtype=np.float32) / DEFAULT_SAMPLE_RATE
    audio = 0.2 * np.sin(2.0 * np.pi * frequency * t)
    return encode_wav(audio.astype(np.float32), DEFAULT_SAMPLE_RATE)
