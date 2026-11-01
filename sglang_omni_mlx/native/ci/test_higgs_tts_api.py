# SPDX-License-Identifier: Apache-2.0
"""API tests for higgs_tts_server, run against the real model; the speech it
returns is checked by transcribing it with qwen3_asr_server.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_higgs_tts_api.py
"""

from __future__ import annotations

import http.client
import io
import json
import re
import socket
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from test_server_api import DATA_ROOT, RUNTIME_BIN, Server

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
HIGGS_REPO = "bosonai/higgs-audio-v3-tts-4b"
CHOSEN_MODEL_NAME = "bosonai/higgs-audio-v3-tts-4b"
SAMPLE_RATE = 24000
ASR_SAMPLE_RATE = 16000
# One audio frame is 40 ms of 16-bit samples.
FRAME_BYTES = 960 * 2
# Rows before the first whole frame: every codebook opens one row after the last.
DELAY_ROWS = 7
# The fewest rows a request may ask for, one frame's worth.
MIN_ROWS = 8
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."
LONG_SENTENCE = " ".join([SENTENCE] * 8)
# Note (Jiaxin Deng): the model ends this passage early, after about 340 rows
# (13 s on an M5), which still outlasts SHORT_REQUEST_TIMEOUT_SECONDS.
PASSAGE = " ".join([SENTENCE] * 50)
# Note (Minyi Tang): generation samples noise, so the audio differs on every
# request; the transcript may miss one word without the speech being wrong.
MAX_WORD_ERRORS = 1
MAX_QUEUED_REQUESTS = 1
REQUEST_TIMEOUT_SECONDS = 20
# The deadline of a second server, which the deadline tests reach.
SHORT_REQUEST_TIMEOUT_SECONDS = 3
MAX_REQUEST_BODY_BYTES = 1024 * 1024
# A cancelled generation stops within a 75-row window.
CANCEL_SECONDS = 8.0


def higgs_directory() -> Path:
    return Path(DATA_ROOT) / "models" / HIGGS_REPO.replace("/", "_")


@pytest.fixture(scope="module")
def higgs_server() -> Iterator[Server]:
    running = Server(
        "higgs_tts_server",
        "higgs_tts",
        higgs_directory(),
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
def short_deadline_server() -> Iterator[Server]:
    running = Server(
        "higgs_tts_server",
        "higgs_tts",
        higgs_directory(),
        ("--request-timeout-s", str(SHORT_REQUEST_TIMEOUT_SECONDS)),
    )
    yield running
    running.stop()


@pytest.fixture(scope="module")
def asr_server() -> Iterator[Server]:
    running = Server()
    yield running
    running.stop()


def speak(server: Server, body: dict) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=300)
    connection.request("POST", "/v1/audio/speech", body=json.dumps(body))
    response = connection.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def send_speech_request(
    server: Server, body: dict, receive_buffer_bytes: int | None = None
) -> socket.socket:
    """Sends a speech request on a socket of its own, without reading the answer."""
    encoded_body = json.dumps(body)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # The receive buffer is sized before connecting, so the window follows it.
    if receive_buffer_bytes is not None:
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer_bytes)
    else:
        pass
    client.settimeout(300)
    client.connect(("127.0.0.1", server.port))
    client.sendall(
        b"POST /v1/audio/speech HTTP/1.1\r\nHost: test\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(encoded_body)}\r\n\r\n{encoded_body}".encode()
    )
    return client


def open_speech_stream(
    server: Server, text: str, receive_buffer_bytes: int | None = None
) -> socket.socket:
    """Sends a streamed speech request; returns once its 200 headers arrive."""
    client = send_speech_request(
        server,
        {"input": text, "response_format": "pcm", "stream": True},
        receive_buffer_bytes,
    )
    received = b""
    while b"\r\n\r\n" not in received:
        chunk = client.recv(65536)
        if not chunk:
            client.close()
            raise ConnectionError(f"stream closed before its headers: {received!r}")
        else:
            received += chunk
    if not received.startswith(b"HTTP/1.1 200 "):
        client.close()
        raise AssertionError(f"stream was refused: {received!r}")
    else:
        pass
    return client


def request_states(server: Server) -> dict[str, int]:
    return json.loads(server.request("GET", "/health")[1])["request_states"]


def wait_until_running(server: Server) -> None:
    started_at = time.monotonic()
    while request_states(server) != {"running": 1}:
        assert time.monotonic() - started_at < CANCEL_SECONDS
        time.sleep(0.05)


def wait_until_idle(server: Server, timeout_seconds: float = CANCEL_SECONDS) -> None:
    started_at = time.monotonic()
    while request_states(server):
        assert (
            time.monotonic() - started_at < timeout_seconds
        ), "generation kept running after it was cancelled"
        time.sleep(0.05)


def transcript_words(asr_server: Server, pcm: bytes) -> list[str]:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    resampled_count = len(samples) * ASR_SAMPLE_RATE // SAMPLE_RATE
    # Band-limited resampling: keep the spectrum below the new Nyquist rate.
    spectrum = np.fft.rfft(samples)[: resampled_count // 2 + 1]
    resampled = np.fft.irfft(spectrum, resampled_count) * resampled_count / len(samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(ASR_SAMPLE_RATE)
        writer.writeframes(np.clip(resampled, -32768, 32767).astype("<i2").tobytes())
    status, body = asr_server.post_form({"language": "en"}, buffer.getvalue())
    assert status == 200
    return words(json.loads(body)["text"])


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    distances = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        diagonal, distances[0] = distances[0], row
        for column, heard in enumerate(hypothesis, 1):
            diagonal, distances[column] = distances[column], min(
                distances[column] + 1,
                distances[column - 1] + 1,
                diagonal + (expected != heard),
            )
    return distances[-1]


def test_higgs_wav_speaks_its_input(higgs_server: Server, asr_server: Server) -> None:
    status, headers, audio = speak(
        higgs_server,
        {
            "input": SENTENCE,
            "voice": "default",
            "model": higgs_server.ready["model_name"],
        },
    )
    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Content-Disposition"] == 'attachment; filename="speech.wav"'
    assert headers["X-Request-Id"].startswith("speech-")
    assert headers["X-Finish-Reason"] == "stop"
    row_count = int(headers["X-Completion-Tokens"])
    prompt_token_count = int(headers["X-Prompt-Tokens"])
    assert int(headers["X-SGLang-Omni-Total-Tokens"]) == (
        prompt_token_count + row_count
    )
    with wave.open(io.BytesIO(audio)) as reader:
        assert (reader.getnchannels(), reader.getsampwidth()) == (1, 2)
        assert reader.getframerate() == SAMPLE_RATE
        pcm = reader.readframes(reader.getnframes())
    assert len(pcm) == (row_count - DELAY_ROWS) * FRAME_BYTES
    heard = transcript_words(asr_server, pcm)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_higgs_streams_pcm_while_it_speaks(
    higgs_server: Server, asr_server: Server
) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", higgs_server.port, timeout=300)
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
    # The first chunk is twenty frames: it arrives well before the last.
    assert first_audio_at - started_at < (finished_at - started_at) / 2
    heard = transcript_words(asr_server, pcm)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_higgs_stops_at_max_new_tokens(higgs_server: Server) -> None:
    status, headers, pcm = speak(
        higgs_server,
        {"input": SENTENCE, "response_format": "pcm", "max_new_tokens": 30},
    )
    assert status == 200
    assert (headers["X-Finish-Reason"], headers["X-Completion-Tokens"]) == (
        "length",
        "30",
    )
    assert len(pcm) == (30 - DELAY_ROWS) * FRAME_BYTES


def test_higgs_streamed_and_whole_responses_share_a_seed(
    higgs_server: Server,
) -> None:
    """A seed draws the same rows whether the response streams or not, and the
    stream's windows add up to the whole response's frames."""
    body = {"input": SENTENCE, "response_format": "pcm", "seed": 7}
    status, headers, whole = speak(higgs_server, body)
    assert status == 200
    status, _, again = speak(higgs_server, body)
    assert (status, again) == (200, whole)
    status, _, streamed = speak(higgs_server, body | {"stream": True})
    assert status == 200
    assert len(streamed) == len(whole)
    assert (
        len(whole) == (int(headers["X-Completion-Tokens"]) - DELAY_ROWS) * FRAME_BYTES
    )


def test_higgs_disconnect_stops_its_generation(higgs_server: Server) -> None:
    with open_speech_stream(higgs_server, LONG_SENTENCE) as client:
        assert client.recv(65536)
        assert request_states(higgs_server) == {"running": 1}
    wait_until_idle(higgs_server)


def test_higgs_disconnect_stops_a_generation_it_has_not_answered(
    higgs_server: Server,
) -> None:
    """A WAV request kept generating after its client disconnected."""
    with send_speech_request(higgs_server, {"input": PASSAGE}):
        wait_until_running(higgs_server)
    wait_until_idle(higgs_server)


def test_higgs_disconnect_is_seen_past_bytes_sent_after_the_request(
    higgs_server: Server,
) -> None:
    """Bytes a client sent after its request hid that it then disconnected."""
    with send_speech_request(higgs_server, {"input": PASSAGE}) as client:
        wait_until_running(higgs_server)
        client.sendall(b"GET /health HTTP/1.1\r\nHost: test\r\n\r\n")
    wait_until_idle(higgs_server)


def test_higgs_half_closed_client_counts_as_gone(higgs_server: Server) -> None:
    """uvicorn closes a connection whose client shuts down its sending side, so
    CUDA cancels that request and sends nothing; this server must match it."""
    with send_speech_request(higgs_server, {"input": PASSAGE}) as client:
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        assert received == b""
    wait_until_idle(higgs_server)


def test_higgs_refuses_requests_beyond_its_queue(higgs_server: Server) -> None:
    with (
        open_speech_stream(higgs_server, LONG_SENTENCE) as running,
        open_speech_stream(higgs_server, LONG_SENTENCE),
    ):
        assert running.recv(65536)
        assert request_states(higgs_server) == {
            "running": 1,
            "queued": MAX_QUEUED_REQUESTS,
        }
        status, _, body = speak(higgs_server, {"input": "hi"})
        assert (status, json.loads(body)) == (
            503,
            {"detail": "The request queue is full."},
        )
    # Each close cancels its request, whether it runs or waits in the queue.
    wait_until_idle(higgs_server, 2 * CANCEL_SECONDS)


def test_higgs_times_out_a_request_past_its_deadline(
    short_deadline_server: Server,
) -> None:
    started_at = time.monotonic()
    status, _, body = speak(
        short_deadline_server, {"input": PASSAGE, "response_format": "pcm"}
    )
    assert (status, json.loads(body)) == (
        504,
        {"detail": "Speech generation timed out."},
    )
    assert time.monotonic() - started_at >= SHORT_REQUEST_TIMEOUT_SECONDS
    wait_until_idle(short_deadline_server)


def test_higgs_stops_a_stream_its_client_stops_reading_at_its_deadline(
    short_deadline_server: Server,
) -> None:
    # Note (Minyi Tang): seconds of audio fill a 4 KB receive window and the
    # server's send buffer, so the server's write blocks while generation runs.
    with open_speech_stream(short_deadline_server, PASSAGE, receive_buffer_bytes=4096):
        assert request_states(short_deadline_server) == {"running": 1}
        wait_until_idle(
            short_deadline_server, SHORT_REQUEST_TIMEOUT_SECONDS + CANCEL_SECONDS
        )


def test_higgs_accepts_any_model_without_a_chosen_name(
    higgs_server: Server,
) -> None:
    status, _, _ = speak(
        higgs_server,
        {
            "input": "hi",
            "model": "tts-1",
            "response_format": "pcm",
            "max_new_tokens": MIN_ROWS,
        },
    )
    assert status == 200


def test_higgs_ignores_unsupported_fields_left_empty(
    higgs_server: Server,
) -> None:
    status, _, _ = speak(
        higgs_server,
        {
            "input": "hi",
            "response_format": "pcm",
            "max_new_tokens": MIN_ROWS,
            "seed": None,
            "language": "",
            "references": [],
            "stage_params": {},
        },
    )
    assert status == 200


def test_higgs_trims_and_lowercases_its_response_format(
    higgs_server: Server,
) -> None:
    status, headers, _ = speak(
        higgs_server,
        {
            "input": "hi",
            "response_format": "\u3000PCM\u00a0",
            "max_new_tokens": MIN_ROWS,
        },
    )
    assert (status, headers["Content-Type"]) == (200, "audio/pcm")


def test_higgs_rejects_another_model_than_its_chosen_name() -> None:
    named_server = Server(
        "higgs_tts_server",
        "higgs_tts",
        higgs_directory(),
        ("--model-name", CHOSEN_MODEL_NAME),
    )
    try:
        status, _, _ = speak(
            named_server,
            {
                "input": "hi",
                "model": CHOSEN_MODEL_NAME,
                "response_format": "pcm",
                "max_new_tokens": MIN_ROWS,
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
        ({"input": "hi", "voice": "alloy"}, "Unknown voice 'alloy'. "),
        ({"input": "hi", "response_format": "mp3"}, "response_format must be one of"),
        ({"input": "hi", "response_format": "p c m"}, "response_format must be one of"),
        ({"input": "hi", "stream": True}, "stream=true requires response_format='pcm'"),
        ({"input": "hi", "max_new_tokens": 7}, "max_new_tokens must be at least 8"),
        ({"input": "hi", "max_new_tokens": 8.5}, "max_new_tokens must be an integer"),
        ({"input": "hi", "temperature": -0.1}, "temperature must be at least 0"),
        ({"input": "hi", "temperature": "hot"}, "temperature must be a number"),
        ({"input": "hi", "top_p": 0}, "top_p must be above 0 and at most 1"),
        ({"input": "hi", "top_p": 1.5}, "top_p must be above 0 and at most 1"),
        ({"input": "hi", "top_k": 0}, "top_k must be -1 (off) or at least 1"),
        ({"input": "hi", "top_k": 2.0}, "top_k must be an integer"),
        ({"input": "hi", "top_k": 2**64 - 1}, "top_k must be an integer"),
        ({"input": "hi", "seed": -1}, "seed must be a non-negative integer"),
        ({"input": "hi", "seed": "1"}, "seed must be a non-negative integer"),
        ({"input": "hi", "speed": 2.0}, "speed must be 1.0 on this model"),
        ({"input": "hi", "voice": None}, "voice must be a string"),
        ({"input": "hi", "speaker": None}, "voice must be a string"),
        (
            {"input": "hi", "voice": None, "speaker": "default"},
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
            {
                "input": "hi",
                "repetition_penalty": 1.1,
                "language": "en",
                "ref_text": "",
                "ref_audio": "https://example.com/a.wav",
            },
            "Higgs Audio v3 TTS does not support these /v1/audio/speech fields: "
            "language, ref_audio, repetition_penalty. ",
        ),
    ],
)
def test_higgs_rejects_invalid_requests(
    higgs_server: Server, body: object, detail: str
) -> None:
    status, response = higgs_server.request(
        "POST", "/v1/audio/speech", json.dumps(body).encode()
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(detail)


def test_higgs_refuses_a_declared_body_too_large_without_reading_it(
    higgs_server: Server,
) -> None:
    # Note (Minyi Tang): the body is never sent, so a server that waits to
    # read it never answers.
    connection = http.client.HTTPConnection("127.0.0.1", higgs_server.port, timeout=10)
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
def test_higgs_caps_a_chunked_body_it_cannot_size_in_advance(
    higgs_server: Server, body_bytes: int, status: int, detail: str
) -> None:
    body = json.dumps({"input": "x" * (body_bytes - len('{"input": ""}'))}).encode()
    assert len(body) == body_bytes
    connection = http.client.HTTPConnection("127.0.0.1", higgs_server.port, timeout=60)
    # Note (Minyi Tang): a body of one chunk is read to its end before the
    # answer, so the socket closes with nothing unread and is not reset.
    connection.request("POST", "/v1/audio/speech", body=[body])
    response = connection.getresponse()
    assert response.status == status
    assert json.loads(response.read())["detail"].startswith(detail)
    connection.close()


def test_higgs_server_serves_no_transcriptions(higgs_server: Server) -> None:
    # Note (Minyi Tang): civetweb answers a path it does not serve without
    # reading the request's body, and a socket closed with bytes unread is
    # reset, which can drop the answer; so the request has no body.
    status, _ = higgs_server.request("POST", "/v1/audio/transcriptions")
    assert status == 404
