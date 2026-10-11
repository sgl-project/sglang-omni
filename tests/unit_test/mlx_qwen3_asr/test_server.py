# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from starlette.testclient import TestClient  # noqa: E402

from sglang_omni_mlx.qwen3_asr.audio import AudioLayout  # noqa: E402
from sglang_omni_mlx.qwen3_asr.realtime import realtime_settings  # noqa: E402
from sglang_omni_mlx.qwen3_asr.server import build_app  # noqa: E402
from tests.unit_test.mlx_qwen3_asr.fakes import FakeWorker  # noqa: E402

MODEL_NAME = "voxt-qwen3_asr-test"


def wav_bytes(seconds: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.full(int(seconds * 16000), 3000, dtype="<i2").tobytes())
    return buffer.getvalue()


def client_for(worker: FakeWorker) -> TestClient:
    return TestClient(build_app(worker, MODEL_NAME, realtime_settings(1000, 100, 30.0)))


def post(client: TestClient, wav: bytes | None = None, **fields: str):
    files = {"file": ("audio.wav", wav, "audio/wav")} if wav is not None else None
    return client.post(
        "/v1/audio/transcriptions", data={"model": MODEL_NAME, **fields}, files=files
    )


def sse_events(body: str) -> list[object]:
    return [
        (
            line[len("data: ") :]
            if line == "data: [DONE]"
            else json.loads(line[len("data: ") :])
        )
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_health_and_models() -> None:
    client = client_for(FakeWorker())
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
    response = post(client_for(worker), wav_bytes(0.5))
    assert response.status_code == 200
    assert response.json() == {"text": "heard 8000"}
    assert worker.calls[0].options.layout is AudioLayout.REFERENCE
    assert worker.calls[0].options.language is None


def test_voxt_final_request_streams_text_and_generation_metadata() -> None:
    worker = FakeWorker()
    response = post(
        client_for(worker),
        wav_bytes(0.5),
        stream="true",
        language="en",
        prompt="Kubernetes",
        max_new_tokens="1024",
        stop_at_end_of_text="true",
        stop_on_token_loop="true",
        include_generation_metadata="true",
        audio_layout="voxt_swift",
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    assert sse_events(response.text) == [
        {
            "type": "transcript.text.done",
            "text": "heard 8000",
            "generation_metadata": {
                "generated_token_count": 2,
                "language": "English",
                "finish_reason": "stop",
            },
        },
        "[DONE]",
    ]
    options = worker.calls[0].options
    assert (options.language, options.context, options.max_new_tokens) == (
        "English",
        "Kubernetes",
        1024,
    )
    assert options.stop_at_end_of_text and options.stop_on_token_loop
    assert options.layout is AudioLayout.VOXT_SWIFT


def test_streaming_without_metadata_leaves_it_out() -> None:
    response = post(client_for(FakeWorker()), wav_bytes(0.2), stream="true")
    assert sse_events(response.text) == [
        {"type": "transcript.text.done", "text": "heard 3200"},
        "[DONE]",
    ]


@pytest.mark.parametrize(
    ("wav", "fields"),
    [
        (None, {}),
        (b"not audio", {}),
        (wav_bytes(0.1), {"include_generation_metadata": "true"}),
        (wav_bytes(0.1), {"audio_layout": "sideways"}),
        (wav_bytes(0.1), {"max_new_tokens": "many"}),
    ],
)
def test_invalid_requests_are_rejected_before_decoding(
    wav: bytes | None, fields: dict[str, str]
) -> None:
    worker = FakeWorker()
    response = post(client_for(worker), wav, **fields)
    assert response.status_code == 400
    assert worker.calls == []


def test_an_unknown_language_is_passed_on_as_given() -> None:
    worker = FakeWorker()
    response = post(client_for(worker), wav_bytes(0.1), language=" Klingon ")
    assert response.status_code == 200
    assert worker.calls[0].options.language == "Klingon"


def test_a_failed_decode_streams_an_error_without_content() -> None:
    response = post(
        client_for(FakeWorker(failure=RuntimeError("secret transcript"))),
        wav_bytes(0.2),
        stream="true",
    )
    events = sse_events(response.text)
    assert events[0]["error"]["code"] == "transcription_failed"
    assert "secret" not in response.text
    assert events[-1] == "[DONE]"


def test_realtime_socket_runs_a_manual_session() -> None:
    client = client_for(FakeWorker())
    pcm = np.full(3200, 3000, dtype="<i2").tobytes()
    with client.websocket_connect("/v1/realtime") as socket:
        socket.send_text("{not json")
        assert json.loads(socket.receive_text())["error"]["code"] == "invalid_json"
        socket.send_json(
            {
                "type": "session.update",
                "session": {"turn_detection": None, "language": "en"},
            }
        )
        assert (
            json.loads(socket.receive_text())["type"] == "transcription_session.updated"
        )
        socket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }
        )
        socket.send_json({"type": "transcription.done"})
        events = []
        while not events or events[-1]["type"] != "transcription.completed":
            events.append(json.loads(socket.receive_text()))
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["text"] for event in finals] == ["heard 3200"]
    assert events[-1]["text"] == "heard 3200"


def test_a_failed_live_decode_is_reported_on_the_socket() -> None:
    client = client_for(FakeWorker(failure=RuntimeError("secret transcript")))
    pcm = np.full(3200, 3000, dtype="<i2").tobytes()
    with client.websocket_connect("/v1/realtime") as socket:
        socket.send_json(
            {"type": "session.update", "session": {"turn_detection": None}}
        )
        assert (
            json.loads(socket.receive_text())["type"] == "transcription_session.updated"
        )
        socket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        events = []
        while not any(event["type"] == "error" for event in events):
            events.append(json.loads(socket.receive_text()))
    assert events[-1]["error"]["code"] == "transcription_failed"
    assert "secret" not in json.dumps(events)
