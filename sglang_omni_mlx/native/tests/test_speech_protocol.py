# SPDX-License-Identifier: Apache-2.0
"""POST /v1/audio/speech, the speech API every native TTS server shares."""

from __future__ import annotations

import base64
import http.client
import io
import json
import socket
import wave
from pathlib import Path

import pytest
from native_server import REQUEST_TIMEOUT_SECONDS, SupervisedServer

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
BLOCKING_REQUEST = {"input": "hello", "outcome": "block", "chunks": 0}
# Far more audio than a socket's buffers hold, for a client that never reads it.
UNREAD_REQUEST = {
    **BLOCKING_REQUEST,
    "response_format": "pcm",
    "stream": True,
    "chunks": 64,
    "chunk_samples": 240000,
}


def open_running_request(
    server: SupervisedServer, request: dict[str, object]
) -> socket.socket:
    """A speech request the server is running, its client not reading the answer."""
    client = server.open_request(
        "/v1/audio/speech", "application/json", json.dumps(request).encode()
    )
    server.wait_for_request_states({"running": 1})
    return client


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
        (json.dumps({"input": "hello", "response_format": "p c m"}).encode(), "response_format must be one of: pcm, wav"),
        (json.dumps({"input": "hello", "response_format": None}).encode(), "response_format must be a string"),
        (json.dumps({"input": "hello", "stream": None}).encode(), "stream must be a boolean"),
        (json.dumps({"input": "hello", "stream_format": None}).encode(), "stream_format must be audio"),
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


def test_a_declared_body_over_the_limit_is_refused_without_reading_it(
    speech_server: SupervisedServer,
) -> None:
    # Note (Minyi Tang): the body is never sent, so a server that waits to read it
    # never answers.
    connection = http.client.HTTPConnection(
        "127.0.0.1", speech_server.port, timeout=REQUEST_TIMEOUT_SECONDS
    )
    connection.putrequest("POST", "/v1/audio/speech")
    connection.putheader("Content-Length", str(STUB_MAX_REQUEST_BODY_BYTES + 1))
    connection.endheaders()
    response = connection.getresponse()
    assert (response.status, json.loads(response.read())) == (
        413,
        {"detail": f"request body must be at most {STUB_MAX_REQUEST_BODY_BYTES} bytes"},
    )
    connection.close()


@pytest.mark.parametrize(
    ("body_byte_count", "status"),
    [(STUB_MAX_REQUEST_BODY_BYTES, 200), (STUB_MAX_REQUEST_BODY_BYTES + 1, 413)],
)
def test_a_chunked_body_is_capped_as_it_is_read(
    speech_server: SupervisedServer, body_byte_count: int, status: int
) -> None:
    prefix = '{"input": "'
    body = (prefix + "x" * (body_byte_count - len(prefix) - 2) + '"}').encode()
    assert len(body) == body_byte_count
    connection = http.client.HTTPConnection(
        "127.0.0.1", speech_server.port, timeout=REQUEST_TIMEOUT_SECONDS
    )
    # Note (Minyi Tang): a body of one chunk is read to its end before the answer,
    # so the socket closes with nothing unread and is not reset.
    connection.request("POST", "/v1/audio/speech", body=[body])
    assert connection.getresponse().status == status
    connection.close()


def test_a_response_format_is_trimmed_and_lowercased(
    speech_server: SupervisedServer,
) -> None:
    response = speech_server.post_speech(
        {**SPEECH_REQUEST, "response_format": "\u3000PCM\u00a0"}
    )
    assert (response.status, response.headers["Content-Type"]) == (200, "audio/pcm")


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_generation_gets_500(
    speech_server: SupervisedServer, stream: bool
) -> None:
    """A failure before the first audio chunk returned an empty successful stream."""
    response = speech_server.post_speech(
        {
            "input": "hello",
            "outcome": "fail",
            "response_format": "pcm",
            "stream": stream,
        }
    )
    assert response.status == 500
    assert response.json() == {"detail": "Speech generation failed."}


@pytest.mark.parametrize("stream", [False, True])
def test_a_request_error_during_generation_gets_400(
    speech_server: SupervisedServer, stream: bool
) -> None:
    """Input errors discovered during generation returned 500 or an empty successful stream."""
    response = speech_server.post_speech(
        {
            "input": "hello",
            "outcome": "invalid",
            "response_format": "pcm",
            "stream": stream,
        }
    )
    assert response.status == 400
    assert response.json() == {"detail": "stub invalid speech request"}


def test_a_stream_without_audio_gets_500(speech_server: SupervisedServer) -> None:
    """An empty generation returned a successful stream without any audio."""
    response = speech_server.post_speech(
        {"input": "hello", "chunks": 0, "response_format": "pcm", "stream": True}
    )
    assert response.status == 500
    assert response.json() == {"detail": "Speech generation failed."}


def test_a_failure_after_streamed_audio_does_not_append_json(
    speech_server: SupervisedServer,
) -> None:
    """An error after committed PCM must end the stream without a JSON suffix."""
    response = speech_server.post_speech(
        {
            **SPEECH_REQUEST,
            "outcome": "fail_after_audio",
            "response_format": "pcm",
            "stream": True,
        }
    )
    assert response.status == 200
    assert response.body == b"\0\0" * CHUNK_COUNT * CHUNK_SAMPLE_COUNT


def test_a_server_without_a_chosen_name_takes_any_model(
    speech_server: SupervisedServer,
) -> None:
    assert speech_server.post_speech({**SPEECH_REQUEST, "model": "tts-1"}).status == 200


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
    client = open_running_request(server, BLOCKING_REQUEST)
    refused = server.post_speech(SPEECH_REQUEST)
    assert refused.status == 503
    assert refused.json() == {"detail": "The request queue is full."}
    client.close()
    server.wait_for_request_states({})
    assert server.post_speech(SPEECH_REQUEST).status == 200
    server.stop()


@pytest.mark.parametrize("stream", [False, True])
def test_a_request_past_its_deadline_gets_504(
    stub_server_binary: Path, model_directory: Path, stream: bool
) -> None:
    """A stream that timed out before its first audio chunk returned an empty 200."""
    server = SupervisedServer(
        stub_server_binary,
        "stub_speech",
        model_directory,
        ("--request-timeout-s", "0.5"),
    )
    response = server.post_speech(
        {**BLOCKING_REQUEST, "response_format": "pcm", "stream": stream}
    )
    assert response.status == 504
    assert response.json() == {"detail": "Speech generation timed out."}
    server.wait_for_request_states({})
    server.stop()


def test_a_client_that_leaves_cancels_its_stream(
    speech_server: SupervisedServer,
) -> None:
    client = open_running_request(
        speech_server, {**BLOCKING_REQUEST, "response_format": "pcm", "stream": True}
    )
    client.close()
    speech_server.wait_for_request_states({})


def test_a_disconnect_is_seen_past_bytes_sent_after_the_request(
    speech_server: SupervisedServer,
) -> None:
    """Bytes a client sent after its request hid that it then disconnected."""
    client = open_running_request(speech_server, BLOCKING_REQUEST)
    client.sendall(b"GET /health HTTP/1.1\r\nHost: test\r\n\r\n")
    client.close()
    speech_server.wait_for_request_states({})


def test_a_half_closed_client_counts_as_gone(speech_server: SupervisedServer) -> None:
    """uvicorn closes a connection whose client shuts down its sending side, so
    CUDA cancels that request and sends nothing; this server must match it."""
    client = open_running_request(speech_server, BLOCKING_REQUEST)
    client.shutdown(socket.SHUT_WR)
    assert client.recv(65536) == b""
    client.close()
    speech_server.wait_for_request_states({})


def test_a_stream_its_client_stops_reading_stops_at_its_deadline(
    stub_server_binary: Path, model_directory: Path
) -> None:
    """The request's thread is blocked writing, so the generation checks the
    deadline itself."""
    server = SupervisedServer(
        stub_server_binary,
        "stub_speech",
        model_directory,
        ("--request-timeout-s", "0.5"),
    )
    client = open_running_request(server, UNREAD_REQUEST)
    server.wait_for_request_states({})
    client.close()
    server.stop()


def test_a_speech_server_serves_no_transcriptions(
    speech_server: SupervisedServer,
) -> None:
    # Note (Minyi Tang): civetweb answers a path it does not serve without reading
    # the request's body, and a socket closed with bytes unread is reset, which can
    # drop the answer; so the request has no body.
    assert speech_server.request("POST", "/v1/audio/transcriptions").status == 404
