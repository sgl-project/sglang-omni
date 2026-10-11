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

from sglang_omni_mlx.fun_asr.server import build_app  # noqa: E402
from sglang_omni_mlx.fun_asr.transcriber import (  # noqa: E402
    TranscriptionOptions,
    TranscriptionResult,
)
from sglang_omni_mlx.transcription import FinishReason  # noqa: E402

MODEL_NAME = "fun-asr-test"


@dataclass
class FakeWorker:
    """Text is 'heard <sample count>'; records the options each request asked for."""

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
    assert client.get("/health").json() == {
        "status": "healthy",
        "running": True,
        "request_states": {},
    }
    assert client.get("/v1/models").json()["data"] == [
        {"id": MODEL_NAME, "object": "model"}
    ]


def test_plain_transcription_returns_json_text() -> None:
    worker = FakeWorker()
    response = post(worker, wav_bytes(0.5))
    assert response.status_code == 200
    assert response.json() == {"text": "heard 8000"}
    assert worker.calls == [TranscriptionOptions()]


def test_request_fields_become_options() -> None:
    worker = FakeWorker()
    response = post(
        worker,
        wav_bytes(0.5),
        language="en",
        itn="false",
        prompt="SGLang, MLX ,",
        max_new_tokens="40",
    )
    assert response.status_code == 200
    assert worker.calls == [
        TranscriptionOptions(
            language="英文", itn=False, hotwords=("SGLang", "MLX"), max_new_tokens=40
        )
    ]


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
        ({"max_new_tokens": "0"}, "max_new_tokens must be between"),
        ({"max_new_tokens": "513"}, "max_new_tokens must be between"),
    ],
)
def test_invalid_fields_are_bad_requests(fields: dict[str, str], message: str) -> None:
    response = post(FakeWorker(), wav_bytes(0.5), **fields)
    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_missing_file_and_non_wav_are_bad_requests() -> None:
    assert post(FakeWorker()).status_code == 400
    assert post(FakeWorker(), b"not a wav").status_code == 400


def test_audio_over_thirty_seconds_is_a_bad_request() -> None:
    worker = FakeWorker()
    response = post(worker, wav_bytes(31.0))
    assert response.status_code == 400
    assert "30 seconds" in response.json()["detail"]
    assert worker.calls == []


def test_itn_is_on_unless_the_request_turns_it_off() -> None:
    worker = FakeWorker()
    post(worker, wav_bytes(0.5), itn="")
    post(worker, wav_bytes(0.5), itn="no")
    assert [options.itn for options in worker.calls] == [True, False]


def test_a_failed_stream_sends_an_error_event_then_done() -> None:
    response = post(
        FakeWorker(failure=RuntimeError("boom")), wav_bytes(0.5), stream="true"
    )
    lines = [line for line in response.text.splitlines() if line]
    assert json.loads(lines[0].removeprefix("data: "))["error"]["code"] == (
        "transcription_failed"
    )
    assert lines[1] == "data: [DONE]"
