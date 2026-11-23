# SPDX-License-Identifier: Apache-2.0
"""API tests for voxtral_tts_server, run against the real model; the speech it
returns is checked by transcribing it with qwen3_asr_server.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_voxtral_tts_api.py
"""

from __future__ import annotations

import http.client
import io
import json
import socket
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from speech_api import (
    open_speech_stream,
    request_states,
    send_speech_request,
    speak,
    transcript_words,
    wait_until_idle,
    wait_until_running,
    word_errors,
    words,
)
from test_server_api import DATA_ROOT, RUNTIME_BIN, Server

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
VOXTRAL_REPO = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
CHOSEN_MODEL_NAME = "mistralai/Voxtral-4B-TTS-2603"
SAMPLE_RATE = 24000
# One audio frame is 80 ms of 16-bit samples.
FRAME_BYTES = 1920 * 2
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."
LONG_SENTENCE = " ".join([SENTENCE] * 8)
# Minutes of speech: no Mac generates it within REQUEST_TIMEOUT_SECONDS.
PASSAGE = " ".join([SENTENCE] * 50)
# Note (Minyi Tang): generation samples noise, so the audio differs on every
# request; the transcript may miss one word without the speech being wrong.
MAX_WORD_ERRORS = 1
MAX_QUEUED_REQUESTS = 1
REQUEST_TIMEOUT_SECONDS = 20
MAX_REQUEST_BODY_BYTES = 1024 * 1024
# A cancelled generation stops within two 25-frame chunks.
CANCEL_SECONDS = 8.0


def voxtral_directory() -> Path:
    return Path(DATA_ROOT) / "models" / VOXTRAL_REPO.replace("/", "_")


@pytest.fixture(scope="module")
def voxtral_server() -> Iterator[Server]:
    running = Server(
        "voxtral_tts_server",
        "voxtral_tts",
        voxtral_directory(),
        (
            "--max-queued-requests",
            str(MAX_QUEUED_REQUESTS),
            "--request-timeout-s",
            str(REQUEST_TIMEOUT_SECONDS),
        ),
    )
    yield running
    running.stop()


@pytest.fixture(scope="module")
def asr_server() -> Iterator[Server]:
    running = Server()
    yield running
    running.stop()


def test_voxtral_wav_speaks_its_input(
    voxtral_server: Server, asr_server: Server
) -> None:
    status, headers, audio = speak(
        voxtral_server,
        {
            "input": SENTENCE,
            "voice": "neutral_female",
            "model": voxtral_server.ready["model_name"],
        },
    )
    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Content-Disposition"] == 'attachment; filename="speech.wav"'
    assert headers["X-Request-Id"].startswith("speech-")
    assert headers["X-Finish-Reason"] == "stop"
    frame_count = int(headers["X-Completion-Tokens"])
    prompt_token_count = int(headers["X-Prompt-Tokens"])
    assert int(headers["X-SGLang-Omni-Total-Tokens"]) == (
        prompt_token_count + frame_count
    )
    with wave.open(io.BytesIO(audio)) as reader:
        assert (reader.getnchannels(), reader.getsampwidth()) == (1, 2)
        assert reader.getframerate() == SAMPLE_RATE
        pcm = reader.readframes(reader.getnframes())
    assert len(pcm) == frame_count * FRAME_BYTES
    heard = transcript_words(asr_server, pcm, SAMPLE_RATE)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_voxtral_streams_pcm_while_it_speaks(
    voxtral_server: Server, asr_server: Server
) -> None:
    connection = http.client.HTTPConnection(
        "127.0.0.1", voxtral_server.port, timeout=300
    )
    started_at = time.monotonic()
    connection.request(
        "POST",
        "/v1/audio/speech",
        body=json.dumps({"input": SENTENCE, "response_format": "pcm", "stream": True}),
    )
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type") == "audio/pcm"
    assert response.getheader("X-Sample-Rate") == str(SAMPLE_RATE)
    assert (response.getheader("X-Channels"), response.getheader("X-Bit-Depth")) == (
        "1",
        "16",
    )
    chunks = []
    first_audio_at = None
    while chunk := response.read1(65536):
        first_audio_at = first_audio_at or time.monotonic()
        chunks.append(chunk)
    finished_at = time.monotonic()
    pcm = b"".join(chunks)
    assert pcm and len(pcm) % FRAME_BYTES == 0
    # The first chunk is five frames: it arrives well before the last.
    assert first_audio_at - started_at < (finished_at - started_at) / 2
    heard = transcript_words(asr_server, pcm, SAMPLE_RATE)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_voxtral_stops_at_max_new_tokens(voxtral_server: Server) -> None:
    status, headers, pcm = speak(
        voxtral_server,
        {"input": SENTENCE, "response_format": "pcm", "max_new_tokens": 10},
    )
    assert status == 200
    assert (headers["X-Finish-Reason"], headers["X-Completion-Tokens"]) == (
        "length",
        "10",
    )
    assert len(pcm) == 10 * FRAME_BYTES


def test_voxtral_disconnect_stops_its_generation(voxtral_server: Server) -> None:
    with open_speech_stream(voxtral_server, {"input": LONG_SENTENCE}) as client:
        assert client.recv(65536)
        assert request_states(voxtral_server) == {"running": 1}
    wait_until_idle(voxtral_server, CANCEL_SECONDS)


def test_voxtral_disconnect_stops_a_generation_it_has_not_answered(
    voxtral_server: Server,
) -> None:
    """A WAV request kept generating after its client disconnected."""
    with send_speech_request(voxtral_server, {"input": PASSAGE}):
        wait_until_running(voxtral_server, CANCEL_SECONDS)
    wait_until_idle(voxtral_server, CANCEL_SECONDS)


def test_voxtral_disconnect_is_seen_past_bytes_sent_after_the_request(
    voxtral_server: Server,
) -> None:
    """Bytes a client sent after its request hid that it then disconnected."""
    with send_speech_request(voxtral_server, {"input": PASSAGE}) as client:
        wait_until_running(voxtral_server, CANCEL_SECONDS)
        client.sendall(b"GET /health HTTP/1.1\r\nHost: test\r\n\r\n")
    wait_until_idle(voxtral_server, CANCEL_SECONDS)


def test_voxtral_half_closed_client_counts_as_gone(voxtral_server: Server) -> None:
    """uvicorn closes a connection whose client shuts down its sending side, so
    CUDA cancels that request and sends nothing; this server must match it."""
    with send_speech_request(voxtral_server, {"input": PASSAGE}) as client:
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        assert received == b""
    wait_until_idle(voxtral_server, CANCEL_SECONDS)


def test_voxtral_refuses_requests_beyond_its_queue(voxtral_server: Server) -> None:
    with (
        open_speech_stream(voxtral_server, {"input": LONG_SENTENCE}) as running,
        open_speech_stream(voxtral_server, {"input": LONG_SENTENCE}),
    ):
        assert running.recv(65536)
        assert request_states(voxtral_server) == {
            "running": 1,
            "queued": MAX_QUEUED_REQUESTS,
        }
        status, _, body = speak(voxtral_server, {"input": "hi"})
        assert (status, json.loads(body)) == (
            503,
            {"detail": "The request queue is full."},
        )
    # Each close cancels its request, whether it runs or waits in the queue.
    wait_until_idle(voxtral_server, 2 * CANCEL_SECONDS)


def test_voxtral_times_out_a_request_past_its_deadline(
    voxtral_server: Server,
) -> None:
    started_at = time.monotonic()
    status, _, body = speak(
        voxtral_server, {"input": PASSAGE, "response_format": "pcm"}
    )
    assert (status, json.loads(body)) == (
        504,
        {"detail": "Speech generation timed out."},
    )
    assert time.monotonic() - started_at >= REQUEST_TIMEOUT_SECONDS
    wait_until_idle(voxtral_server, CANCEL_SECONDS)


def test_voxtral_stops_a_stream_its_client_stops_reading_at_its_deadline(
    voxtral_server: Server,
) -> None:
    # Note (Minyi Tang): seconds of audio fill a 4 KB receive window and the
    # server's send buffer, so the server's write blocks while generation runs.
    with open_speech_stream(
        voxtral_server, {"input": PASSAGE}, receive_buffer_bytes=4096
    ):
        assert request_states(voxtral_server) == {"running": 1}
        wait_until_idle(voxtral_server, REQUEST_TIMEOUT_SECONDS + CANCEL_SECONDS)


def test_voxtral_accepts_any_model_without_a_chosen_name(
    voxtral_server: Server,
) -> None:
    status, _, _ = speak(
        voxtral_server,
        {
            "input": "hi",
            "model": "tts-1",
            "response_format": "pcm",
            "max_new_tokens": 1,
        },
    )
    assert status == 200


def test_voxtral_ignores_unsupported_fields_left_empty(
    voxtral_server: Server,
) -> None:
    status, _, _ = speak(
        voxtral_server,
        {
            "input": "hi",
            "response_format": "pcm",
            "max_new_tokens": 1,
            "seed": None,
            "language": "",
            "references": [],
            "stage_params": {},
        },
    )
    assert status == 200


def test_voxtral_trims_and_lowercases_its_response_format(
    voxtral_server: Server,
) -> None:
    status, headers, _ = speak(
        voxtral_server,
        {"input": "hi", "response_format": "\u3000PCM\u00a0", "max_new_tokens": 1},
    )
    assert (status, headers["Content-Type"]) == (200, "audio/pcm")


def test_voxtral_rejects_another_model_than_its_chosen_name() -> None:
    named_server = Server(
        "voxtral_tts_server",
        "voxtral_tts",
        voxtral_directory(),
        ("--model-name", CHOSEN_MODEL_NAME),
    )
    try:
        status, _, _ = speak(
            named_server,
            {
                "input": "hi",
                "model": CHOSEN_MODEL_NAME,
                "response_format": "pcm",
                "max_new_tokens": 1,
            },
        )
        assert status == 200
        status, _, body = speak(named_server, {"input": "hi", "model": "tts-1"})
        assert (status, json.loads(body)) == (
            404,
            {"detail": "Unknown model: tts-1"},
        )
    finally:
        named_server.stop()


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ([1], "speech request body must be a JSON object"),
        ({"input": "  "}, "input must be a non-empty string"),
        ({"input": "\u3000\u00a0\u2028"}, "input must be a non-empty string"),
        ({"input": "x" * 4097}, "input must be at most 4096 characters"),
        ({"input": "hi", "model": 5}, "model must be a string"),
        ({"input": "hi", "voice": "nobody"}, "Unknown voice 'nobody'. "),
        ({"input": "hi", "response_format": "mp3"}, "response_format must be one of"),
        ({"input": "hi", "response_format": "p c m"}, "response_format must be one of"),
        ({"input": "hi", "stream": True}, "stream=true requires response_format='pcm'"),
        ({"input": "hi", "max_new_tokens": 0}, "max_new_tokens must be greater than 0"),
        ({"input": "hi", "speed": 2.0}, "speed must be 1.0 on this model"),
        ({"input": "hi", "voice": None}, "voice must be a string"),
        ({"input": "hi", "speaker": None}, "voice must be a string"),
        (
            {"input": "hi", "voice": None, "speaker": "casual_male"},
            "voice must be a string",
        ),
        ({"input": "hi", "response_format": None}, "response_format must be a string"),
        ({"input": "hi", "stream": None}, "stream must be a boolean"),
        ({"input": "hi", "stream_format": None}, "stream_format must be audio"),
        ({"input": "hi", "speed": None}, "speed must be 1.0 on this model"),
        ({"input": "hi", "references": False}, "references must be a list"),
        ({"input": "hi", "references": {}}, "references must be a list"),
        ({"input": "hi", "stage_params": []}, "stage_params must be an object"),
        (
            {"input": "hi", "seed": 1, "language": "en", "ref_text": ""},
            "Voxtral TTS does not support these /v1/audio/speech fields: "
            "language, seed. ",
        ),
    ],
)
def test_voxtral_rejects_invalid_requests(
    voxtral_server: Server, body: object, detail: str
) -> None:
    status, response = voxtral_server.request(
        "POST", "/v1/audio/speech", json.dumps(body).encode()
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(detail)


def test_voxtral_refuses_a_declared_body_too_large_without_reading_it(
    voxtral_server: Server,
) -> None:
    # Note (Minyi Tang): the body is never sent, so a server that waits to
    # read it never answers.
    connection = http.client.HTTPConnection(
        "127.0.0.1", voxtral_server.port, timeout=10
    )
    connection.putrequest("POST", "/v1/audio/speech")
    connection.putheader("Content-Length", str(MAX_REQUEST_BODY_BYTES + 1))
    connection.endheaders()
    response = connection.getresponse()
    assert (response.status, json.loads(response.read())) == (
        413,
        {"detail": f"request body must be at most {MAX_REQUEST_BODY_BYTES} bytes"},
    )
    connection.close()


@pytest.mark.parametrize(
    ("body_bytes", "status", "detail"),
    [
        (MAX_REQUEST_BODY_BYTES, 400, "input must be at most 4096 characters"),
        (
            MAX_REQUEST_BODY_BYTES + 1,
            413,
            f"request body must be at most {MAX_REQUEST_BODY_BYTES} bytes",
        ),
    ],
)
def test_voxtral_caps_a_chunked_body_it_cannot_size_in_advance(
    voxtral_server: Server, body_bytes: int, status: int, detail: str
) -> None:
    body = json.dumps({"input": "x" * (body_bytes - len('{"input": ""}'))}).encode()
    assert len(body) == body_bytes
    connection = http.client.HTTPConnection(
        "127.0.0.1", voxtral_server.port, timeout=60
    )
    # Note (Minyi Tang): a body of one chunk is read to its end before the
    # answer, so the socket closes with nothing unread and is not reset.
    connection.request("POST", "/v1/audio/speech", body=[body])
    response = connection.getresponse()
    assert response.status == status
    assert json.loads(response.read())["detail"].startswith(detail)
    connection.close()


def test_voxtral_server_serves_no_transcriptions(voxtral_server: Server) -> None:
    # Note (Minyi Tang): civetweb answers a path it does not serve without
    # reading the request's body, and a socket closed with bytes unread is
    # reset, which can drop the answer; so the request has no body.
    status, _ = voxtral_server.request("POST", "/v1/audio/transcriptions")
    assert status == 404
