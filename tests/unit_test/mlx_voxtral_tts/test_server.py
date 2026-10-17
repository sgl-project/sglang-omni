# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from starlette.testclient import TestClient  # noqa: E402

from sglang_omni_mlx.voxtral_tts.audio import encode_wav  # noqa: E402
from sglang_omni_mlx.voxtral_tts.server import build_app  # noqa: E402
from sglang_omni_mlx.voxtral_tts.worker import (  # noqa: E402
    SpeechGenerationTimeoutError,
    SpeechQueueFullError,
    SpeechResult,
)

MODEL_NAME = "voxtral-test"


@dataclass
class FakeWorker:
    voice_embeddings: dict[str, np.ndarray] = field(
        default_factory=lambda: {"cheerful_female": np.zeros((1, 4), dtype=np.float32)}
    )
    calls: list[tuple[str, str, int]] = field(default_factory=list)
    failure: Exception | None = None

    def request_states(self) -> dict[str, int]:
        return {"queued": 0, "running": 0}

    async def synthesize(
        self, text: str, voice: str, max_new_tokens: int
    ) -> SpeechResult:
        self.calls.append((text, voice, max_new_tokens))
        if self.failure is not None:
            raise self.failure
        else:
            audio = encode_wav(np.zeros(240, dtype=np.float32), 24000)
            return SpeechResult(audio=audio)


def test_speech_route_returns_wav_and_forwards_options() -> None:
    worker = FakeWorker()
    client = TestClient(build_app(worker, MODEL_NAME))
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": MODEL_NAME,
            "input": "Hello there.",
            "voice": "cheerful_female",
            "response_format": "wav",
            "max_new_tokens": 24,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert response.headers["x-speech-finish-reason"] == "stop"
    assert worker.calls == [("Hello there.", "cheerful_female", 24)]


def test_health_and_models_routes_return_server_state() -> None:
    client = TestClient(build_app(FakeWorker(), MODEL_NAME))

    health_response = client.get("/health")
    models_response = client.get("/v1/models")

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "healthy",
        "running": True,
        "request_states": {"queued": 0, "running": 0},
    }
    assert models_response.status_code == 200
    assert models_response.json() == {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model"}],
    }


@pytest.mark.parametrize(
    ("fields", "status_code"),
    [
        ({"input": "Hello", "voice": "unknown"}, 400),
        ({"input": "Hello", "response_format": "mp3"}, 400),
        ({"input": "Hello", "model": "another-model"}, 404),
        ({"input": "", "voice": "cheerful_female"}, 400),
        ({"input": "   ", "voice": "cheerful_female"}, 400),
    ],
)
def test_invalid_speech_requests_do_not_reach_worker(
    fields: dict[str, str], status_code: int
) -> None:
    worker = FakeWorker()
    response = TestClient(build_app(worker, MODEL_NAME)).post(
        "/v1/audio/speech", json=fields
    )
    assert response.status_code == status_code
    assert worker.calls == []


@pytest.mark.parametrize(
    ("failure", "status_code", "detail"),
    [
        (
            SpeechQueueFullError("Voxtral speech request queue is full"),
            429,
            "Voxtral speech request queue is full",
        ),
        (
            SpeechGenerationTimeoutError("Voxtral speech generation timed out"),
            504,
            "Voxtral speech generation timed out",
        ),
        (RuntimeError("internal failure"), 500, "Voxtral speech generation failed"),
    ],
)
def test_speech_route_returns_json_for_worker_errors(
    failure: Exception, status_code: int, detail: str
) -> None:
    worker = FakeWorker(failure=failure)
    response = TestClient(build_app(worker, MODEL_NAME)).post(
        "/v1/audio/speech",
        json={"input": "Hello there.", "voice": "cheerful_female"},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert worker.calls == [("Hello there.", "cheerful_female", 4096)]
    if status_code == 429:
        assert response.headers["retry-after"] == "5"
    else:
        pass
