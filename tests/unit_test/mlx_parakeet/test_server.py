# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import threading
import wave
from dataclasses import dataclass, field

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from starlette.testclient import TestClient  # noqa: E402

from sglang_omni_mlx.parakeet.server import build_app  # noqa: E402
from sglang_omni_mlx.parakeet.transcriber import (  # noqa: E402
    TranscriptionOptions,
    TranscriptionResult,
)

MODEL_NAME = "parakeet-test"


@dataclass
class FakeWorker:
    """Text is 'heard <sample count>'."""

    calls: list[int] = field(default_factory=list)

    def request_states(self) -> dict[str, int]:
        return {}

    async def transcribe(
        self,
        samples: np.ndarray,
        options: TranscriptionOptions,
        cancel: threading.Event,
    ) -> TranscriptionResult:
        self.calls.append(len(samples))
        return TranscriptionResult(
            text=f"heard {len(samples)}", generated_token_count=2, chunk_count=1
        )


def wav_bytes(seconds: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.full(int(seconds * 16000), 3000, dtype="<i2").tobytes())
    return buffer.getvalue()


def post(worker: FakeWorker, wav: bytes | None = None, **fields: str):
    files = {"file": ("audio.wav", wav, "audio/wav")} if wav is not None else None
    return TestClient(build_app(worker, MODEL_NAME)).post(
        "/v1/audio/transcriptions", data={"model": MODEL_NAME, **fields}, files=files
    )


def test_health_and_models() -> None:
    client = TestClient(build_app(FakeWorker(), MODEL_NAME))
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/v1/models").json()["data"] == [
        {"id": MODEL_NAME, "object": "model"}
    ]


def test_transcription_returns_json_text_and_ignores_language() -> None:
    worker = FakeWorker()
    response = post(worker, wav_bytes(0.5), language="fr", temperature="0")
    assert response.status_code == 200
    assert response.json() == {"text": "heard 8000"}


def test_stream_sends_done_event_then_done_marker() -> None:
    response = post(FakeWorker(), wav_bytes(0.5), stream="true")
    lines = [line for line in response.text.splitlines() if line]
    assert json.loads(lines[0].removeprefix("data: ")) == {
        "type": "transcript.text.done",
        "text": "heard 8000",
    }
    assert lines[1] == "data: [DONE]"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"temperature": "0.5"}, "temperature must be 0"),
        ({"temperature": "warm"}, "temperature must be a number"),
        ({"prompt": "SGLang"}, "text prompt"),
        ({"max_new_tokens": "10"}, "max_new_tokens"),
    ],
)
def test_fields_greedy_decoding_cannot_honor_are_bad_requests(
    fields: dict[str, str], message: str
) -> None:
    worker = FakeWorker()
    response = post(worker, wav_bytes(0.5), **fields)
    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert worker.calls == []


def test_missing_file_and_non_wav_are_bad_requests() -> None:
    assert post(FakeWorker()).status_code == 400
    assert post(FakeWorker(), b"not a wav").status_code == 400
