# SPDX-License-Identifier: Apache-2.0
"""POST /v1/audio/speech, the speech API every native TTS server shares."""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

import pytest
from native_server import SupervisedServer

# stub_server's speech sample rate and request body limit.
STUB_SAMPLE_RATE = 24000
STUB_MAX_REQUEST_BODY_BYTES = 64 * 1024
CHUNK_COUNT = 3
CHUNK_SAMPLE_COUNT = 800
SPEECH_REQUEST = {
    "input": "hello",
    "chunks": CHUNK_COUNT,
    "chunk_samples": CHUNK_SAMPLE_COUNT,
}
BLOCKING_REQUEST = {"input": "hello", "outcome": "block"}


@pytest.mark.parametrize("response_format", ["wav", "pcm"])
def test_a_request_answers_one_audio_file(
    speech_server: SupervisedServer, response_format: str
) -> None:
    response = speech_server.post_speech(
        {**SPEECH_REQUEST, "response_format": response_format}
    )
    assert response.status == 200
    assert response.headers["Content-Type"] == f"audio/{response_format}"
    assert int(response.headers["Content-Length"]) == len(response.body)
    assert response.headers["X-Request-Id"].startswith("speech-")
    assert response.headers["X-Finish-Reason"] == "stop"
    assert response.headers["X-Completion-Tokens"] == str(CHUNK_COUNT)
    if response_format == "wav":
        with wave.open(io.BytesIO(response.body)) as reader:
            assert reader.getframerate() == STUB_SAMPLE_RATE
            assert reader.getnchannels() == 1
            assert reader.getsampwidth() == 2
            assert reader.getnframes() == CHUNK_COUNT * CHUNK_SAMPLE_COUNT
    else:
        assert len(response.body) == 2 * CHUNK_COUNT * CHUNK_SAMPLE_COUNT


def test_a_generate_request_answers_json_around_the_base64_wav(
    speech_server: SupervisedServer,
) -> None:
    response = speech_server.request(
        "POST",
        "/generate",
        json.dumps(SPEECH_REQUEST).encode(),
        {"Content-Type": "application/json"},
    )
    assert response.status == 200
    result = response.json()
    assert result["text"] == ""
    assert result["meta_info"] == {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": 0,
        "completion_tokens": CHUNK_COUNT,
    }
    assert result["audio"]["format"] == "wav"
    assert result["audio"]["sample_rate"] == STUB_SAMPLE_RATE
    audio = base64.b64decode(result["audio"]["data"], validate=True)
    with wave.open(io.BytesIO(audio)) as reader:
        assert reader.getframerate() == STUB_SAMPLE_RATE
        assert reader.getnframes() == CHUNK_COUNT * CHUNK_SAMPLE_COUNT


def test_a_streamed_request_sends_every_chunk_as_pcm(
    speech_server: SupervisedServer,
) -> None:
    response = speech_server.post_speech(
        {**SPEECH_REQUEST, "response_format": "pcm", "stream": True}
    )
    assert response.status == 200
    assert response.headers["Content-Type"] == "audio/pcm"
    assert response.headers["X-Sample-Rate"] == str(STUB_SAMPLE_RATE)
    assert response.headers["X-Channels"] == "1"
    assert response.headers["X-Bit-Depth"] == "16"
    assert len(response.body) == 2 * CHUNK_COUNT * CHUNK_SAMPLE_COUNT


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        (b"{", "speech request body must be valid JSON"),
        (b"[]", "speech request body must be a JSON object"),
        (json.dumps({"input": "hello", "model": 5}).encode(), "model must be a string"),
        (json.dumps({"input": "hello", "response_format": "mp3"}).encode(), "response_format must be one of: pcm, wav"),
        (json.dumps({"input": "hello", "stream": "yes"}).encode(), "stream must be a boolean"),
        (json.dumps({"input": "hello", "stream_format": "sse"}).encode(), "stream_format must be audio"),
        (json.dumps({"input": "hello", "stream": True}).encode(), "stream=true requires response_format='pcm'"),
    ],
)  # fmt: skip
def test_invalid_requests_get_400(
    speech_server: SupervisedServer, body: bytes, detail: str
) -> None:
    response = speech_server.request(
        "POST", "/v1/audio/speech", body, {"Content-Type": "application/json"}
    )
    assert response.status == 400
    assert response.json()["detail"] == detail


def test_a_body_over_the_limit_gets_413(speech_server: SupervisedServer) -> None:
    response = speech_server.post_speech({"input": "x" * STUB_MAX_REQUEST_BODY_BYTES})
    assert response.status == 413


def test_a_failed_generation_gets_500(speech_server: SupervisedServer) -> None:
    response = speech_server.post_speech({"input": "hello", "outcome": "fail"})
    assert response.status == 500
    assert response.json() == {"detail": "Speech generation failed."}


def test_a_chosen_model_name_rejects_another_model(
    stub_server_binary: Path, model_directory: Path
) -> None:
    server = SupervisedServer(
        stub_server_binary, "stub_speech", model_directory, ("--model-name", "voxt-tts")
    )
    assert server.post_speech({**SPEECH_REQUEST, "model": "voxt-tts"}).status == 200
    rejected = server.post_speech({**SPEECH_REQUEST, "model": "another"})
    assert rejected.status == 404
    assert rejected.json() == {"detail": "Unknown model: another"}
    server.stop()


def test_a_full_queue_gets_503_until_a_client_leaves(
    stub_server_binary: Path, model_directory: Path
) -> None:
    server = SupervisedServer(
        stub_server_binary,
        "stub_speech",
        model_directory,
        ("--max-queued-requests", "0"),
    )
    client = server.open_request(
        "/v1/audio/speech", "application/json", json.dumps(BLOCKING_REQUEST).encode()
    )
    server.wait_for_request_states({"running": 1})
    refused = server.post_speech(SPEECH_REQUEST)
    assert refused.status == 503
    assert refused.json() == {"detail": "The request queue is full."}
    client.close()
    server.wait_for_request_states({})
    assert server.post_speech(SPEECH_REQUEST).status == 200
    server.stop()


def test_a_request_past_its_deadline_gets_504(
    stub_server_binary: Path, model_directory: Path
) -> None:
    server = SupervisedServer(
        stub_server_binary,
        "stub_speech",
        model_directory,
        ("--request-timeout-s", "0.5"),
    )
    response = server.post_speech(BLOCKING_REQUEST)
    assert response.status == 504
    assert response.json() == {"detail": "Speech generation timed out."}
    server.wait_for_request_states({})
    server.stop()
