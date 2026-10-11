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

from sglang_omni_mlx.moss_transcribe_diarize.server import build_app  # noqa: E402
from sglang_omni_mlx.moss_transcribe_diarize.transcriber import (  # noqa: E402
    TranscriptionOptions,
    TranscriptionResult,
)
from sglang_omni_mlx.transcription import FinishReason  # noqa: E402

MODEL_NAME = "moss-td-test"


@dataclass
class FakeWorker:
    failure: Exception | None = None
    calls: list[TranscriptionOptions] = field(default_factory=list)

    def request_states(self) -> dict[str, int]:
        return {}

    async def transcribe(
        self,
        samples: np.ndarray,
        options: TranscriptionOptions,
        cancel: threading.Event,
    ) -> TranscriptionResult:
        self.calls.append(options)
        if self.failure is not None:
            raise self.failure
        else:
            pass
        return TranscriptionResult(
            text=f"heard {len(samples)}",
            generated_token_count=2,
            finish_reason=FinishReason.STOP,
        )


def wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(
            np.full(int(seconds * sample_rate), 3000, dtype="<i2").tobytes()
        )
    return buffer.getvalue()


def post(worker: FakeWorker, wav: bytes | None = None, **fields: str):
    files = {"file": ("audio.wav", wav, "audio/wav")} if wav is not None else None
    return TestClient(build_app(worker, MODEL_NAME)).post(
        "/v1/audio/transcriptions", data={"model": MODEL_NAME, **fields}, files=files
    )


def test_health_models_and_plain_transcription() -> None:
    worker = FakeWorker()
    client = TestClient(build_app(worker, MODEL_NAME))
    assert client.get("/health").json()["request_states"] == {}
    assert client.get("/v1/models").json()["data"] == [
        {"id": MODEL_NAME, "object": "model"}
    ]
    response = post(worker, wav_bytes(0.5))
    assert response.status_code == 200
    assert response.json() == {"text": "heard 8000"}
    assert worker.calls == [TranscriptionOptions()]


def test_request_fields_become_options() -> None:
    worker = FakeWorker()
    response = post(
        worker,
        wav_bytes(0.5),
        prompt="Transcribe plain text.",
        max_new_tokens="40",
        temperature="0",
        repetition_penalty="1",
    )
    assert response.status_code == 200
    assert worker.calls == [
        TranscriptionOptions(prompt="Transcribe plain text.", max_new_tokens=40)
    ]


def test_request_audio_is_resampled_to_16_khz() -> None:
    worker = FakeWorker()
    response = post(worker, wav_bytes(0.5, sample_rate=48000))
    assert response.status_code == 200
    assert response.json() == {"text": "heard 8000"}


def test_stream_sends_done_event_then_marker() -> None:
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
        ({"max_new_tokens": "0"}, "at least 1"),
        ({"temperature": "0.5"}, "temperature=0"),
        ({"repetition_penalty": "1.1"}, "repetition_penalty=1"),
    ],
)
def test_unsupported_options_are_bad_requests(
    fields: dict[str, str], message: str
) -> None:
    response = post(FakeWorker(), wav_bytes(0.5), **fields)
    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_missing_file_and_non_wav_are_bad_requests() -> None:
    assert post(FakeWorker()).status_code == 400
    assert post(FakeWorker(), b"not a wav").status_code == 400


def test_failed_stream_hides_internal_error() -> None:
    response = post(
        FakeWorker(failure=RuntimeError("secret transcript")),
        wav_bytes(0.5),
        stream="true",
    )
    lines = [line for line in response.text.splitlines() if line]
    assert json.loads(lines[0].removeprefix("data: "))["error"]["code"] == (
        "transcription_failed"
    )
    assert "secret" not in response.text
    assert lines[1] == "data: [DONE]"
