# SPDX-License-Identifier: Apache-2.0
"""API tests for qwen3_tts_server, run against the real model; the speech it
returns is checked by transcribing it with qwen3_asr_server.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_qwen3_tts_tts_api.py
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
QWEN3_TTS_REPO = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
CHOSEN_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SAMPLE_RATE = 24000
ASR_SAMPLE_RATE = 16000
# One audio frame is 80 ms of 16-bit samples.
FRAME_BYTES = 1920 * 2
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."
LONG_SENTENCE = " ".join([SENTENCE] * 8)
# Minutes of speech: no Mac generates it within REQUEST_TIMEOUT_SECONDS.
PASSAGE = " ".join([SENTENCE] * 50)
# Note (Minyi Tang): generation samples its codes, so the audio differs on
# every unseeded request; the transcript may miss one word without the speech
# being wrong.
MAX_WORD_ERRORS = 1
MAX_QUEUED_REQUESTS = 1
REQUEST_TIMEOUT_SECONDS = 20
MAX_REQUEST_BODY_BYTES = 1024 * 1024
# A cancelled generation stops within two 25-frame streamed chunks.
CANCEL_SECONDS = 8.0


def qwen3_tts_directory() -> Path:
    return Path(DATA_ROOT) / "models" / QWEN3_TTS_REPO.replace("/", "_")


@pytest.fixture(scope="module")
def qwen3_tts_server() -> Iterator[Server]:
    running = Server(
        "qwen3_tts_server",
        "qwen3_tts",
        qwen3_tts_directory(),
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


def test_qwen3_tts_wav_speaks_its_input(
    qwen3_tts_server: Server, asr_server: Server
) -> None:
    status, headers, audio = speak(
        qwen3_tts_server,
        {
            "input": SENTENCE,
            "voice": "Ryan",
            "language": "English",
            "model": qwen3_tts_server.ready["model_name"],
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
    # A whole request keeps one frame of audio per frame whose first code is
    # not 0, as CUDA's decode does.
    assert 0 < len(pcm) <= frame_count * FRAME_BYTES
    heard = transcript_words(asr_server, pcm)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_qwen3_tts_streams_pcm_while_it_speaks(
    qwen3_tts_server: Server, asr_server: Server
) -> None:
    connection = http.client.HTTPConnection(
        "127.0.0.1", qwen3_tts_server.port, timeout=300
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
    chunks = [response.read1(65536)]
    first_audio_at = time.monotonic()
    # The default first chunk is five frames, decoded before the next 25.
    assert len(chunks[0]) == 5 * FRAME_BYTES
    while chunk := response.read1(65536):
        first_audio_at = first_audio_at or time.monotonic()
        chunks.append(chunk)
    finished_at = time.monotonic()
    pcm = b"".join(chunks)
    assert pcm and len(pcm) % FRAME_BYTES == 0
    # The first chunk arrives well before the last.
    assert first_audio_at - started_at < (finished_at - started_at) / 2
    heard = transcript_words(asr_server, pcm)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_qwen3_tts_stops_at_max_new_tokens(qwen3_tts_server: Server) -> None:
    status, headers, pcm = speak(
        qwen3_tts_server,
        {"input": SENTENCE, "response_format": "pcm", "max_new_tokens": 10},
    )
    assert status == 200
    assert (headers["X-Finish-Reason"], headers["X-Completion-Tokens"]) == (
        "length",
        "10",
    )
    assert 0 < len(pcm) <= 10 * FRAME_BYTES


def test_qwen3_tts_disconnect_stops_its_generation(qwen3_tts_server: Server) -> None:
    with open_speech_stream(qwen3_tts_server, LONG_SENTENCE) as client:
        assert client.recv(65536)
        assert request_states(qwen3_tts_server) == {"running": 1}
    wait_until_idle(qwen3_tts_server)


def test_qwen3_tts_disconnect_stops_a_generation_it_has_not_answered(
    qwen3_tts_server: Server,
) -> None:
    """A WAV request kept generating after its client disconnected."""
    with send_speech_request(qwen3_tts_server, {"input": PASSAGE}):
        wait_until_running(qwen3_tts_server)
    wait_until_idle(qwen3_tts_server)


def test_qwen3_tts_disconnect_is_seen_past_bytes_sent_after_the_request(
    qwen3_tts_server: Server,
) -> None:
    """Bytes a client sent after its request hid that it then disconnected."""
    with send_speech_request(qwen3_tts_server, {"input": PASSAGE}) as client:
        wait_until_running(qwen3_tts_server)
        client.sendall(b"GET /health HTTP/1.1\r\nHost: test\r\n\r\n")
    wait_until_idle(qwen3_tts_server)


def test_qwen3_tts_half_closed_client_counts_as_gone(qwen3_tts_server: Server) -> None:
    """uvicorn closes a connection whose client shuts down its sending side, so
    CUDA cancels that request and sends nothing; this server must match it."""
    with send_speech_request(qwen3_tts_server, {"input": PASSAGE}) as client:
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        assert received == b""
    wait_until_idle(qwen3_tts_server)


def test_qwen3_tts_refuses_requests_beyond_its_queue(qwen3_tts_server: Server) -> None:
    with (
        open_speech_stream(qwen3_tts_server, LONG_SENTENCE) as running,
        open_speech_stream(qwen3_tts_server, LONG_SENTENCE),
    ):
        assert running.recv(65536)
        assert request_states(qwen3_tts_server) == {
            "running": 1,
            "queued": MAX_QUEUED_REQUESTS,
        }
        status, _, body = speak(qwen3_tts_server, {"input": "hi"})
        assert (status, json.loads(body)) == (
            503,
            {"detail": "The request queue is full."},
        )
    # Each close cancels its request, whether it runs or waits in the queue.
    wait_until_idle(qwen3_tts_server, 2 * CANCEL_SECONDS)


def test_qwen3_tts_times_out_a_request_past_its_deadline(
    qwen3_tts_server: Server,
) -> None:
    started_at = time.monotonic()
    status, _, body = speak(
        qwen3_tts_server, {"input": PASSAGE, "response_format": "pcm"}
    )
    assert (status, json.loads(body)) == (
        504,
        {"detail": "Speech generation timed out."},
    )
    assert time.monotonic() - started_at >= REQUEST_TIMEOUT_SECONDS
    wait_until_idle(qwen3_tts_server)


def test_qwen3_tts_stops_a_stream_its_client_stops_reading_at_its_deadline(
    qwen3_tts_server: Server,
) -> None:
    # Note (Minyi Tang): seconds of audio fill a 4 KB receive window and the
    # server's send buffer, so the server's write blocks while generation runs.
    with open_speech_stream(qwen3_tts_server, PASSAGE, receive_buffer_bytes=4096):
        assert request_states(qwen3_tts_server) == {"running": 1}
        wait_until_idle(qwen3_tts_server, REQUEST_TIMEOUT_SECONDS + CANCEL_SECONDS)


def test_qwen3_tts_accepts_any_model_without_a_chosen_name(
    qwen3_tts_server: Server,
) -> None:
    status, _, _ = speak(
        qwen3_tts_server,
        {
            "input": "hi",
            "model": "tts-1",
            "response_format": "pcm",
            "max_new_tokens": 1,
        },
    )
    assert status == 200


def test_qwen3_tts_takes_null_for_its_optional_fields(
    qwen3_tts_server: Server,
) -> None:
    status, _, _ = speak(
        qwen3_tts_server,
        {
            "input": "hi",
            "response_format": "pcm",
            "max_new_tokens": 1,
            "seed": None,
            "language": None,
            "instructions": None,
            "task_type": None,
            "ref_audio": None,
            "references": [],
            "stage_params": {},
            "temperature": None,
        },
    )
    assert status == 200


@pytest.mark.parametrize(
    ("initial_codec_chunk_frames", "first_chunk_frames"),
    # 0 starts at the steady 25-frame chunk, as on CUDA.
    [(2, 2), (0, 25)],
)
def test_qwen3_tts_first_streamed_chunk_follows_initial_codec_chunk_frames(
    qwen3_tts_server: Server, initial_codec_chunk_frames: int, first_chunk_frames: int
) -> None:
    body = {
        "input": SENTENCE,
        "response_format": "pcm",
        "stream": True,
        "seed": 5,
        "initial_codec_chunk_frames": initial_codec_chunk_frames,
    }
    connection = http.client.HTTPConnection(
        "127.0.0.1", qwen3_tts_server.port, timeout=300
    )
    connection.request("POST", "/v1/audio/speech", body=json.dumps(body))
    response = connection.getresponse()
    assert response.status == 200
    first = b""
    while len(first) < first_chunk_frames * FRAME_BYTES:
        first += response.read1(first_chunk_frames * FRAME_BYTES - len(first))
    started_waiting = time.monotonic()
    # Nothing more arrives until the next chunk is decoded.
    assert response.read1(65536)
    assert time.monotonic() - started_waiting > 0.2
    response.read()


def test_qwen3_tts_tiny_top_p_keeps_the_likeliest_code(
    qwen3_tts_server: Server,
) -> None:
    """A top_p whose 1 - top_p rounds to 1 still keeps the likeliest code, so
    it samples as greedy decoding chooses."""
    body = {"input": "hello", "response_format": "pcm", "max_new_tokens": 8}
    engine = {"subtalker_dosample": False}
    status, _, nucleus = speak(
        qwen3_tts_server,
        {**body, "top_p": 1e-12, "stage_params": {"tts_engine": engine}},
    )
    status_greedy, _, greedy = speak(
        qwen3_tts_server,
        {**body, "stage_params": {"tts_engine": {**engine, "do_sample": False}}},
    )
    assert status == status_greedy == 200
    assert nucleus == greedy


def test_qwen3_tts_seed_repeats_its_audio(qwen3_tts_server: Server) -> None:
    body = {"input": SENTENCE, "response_format": "pcm", "seed": 7}
    first = speak(qwen3_tts_server, body)
    second = speak(qwen3_tts_server, body)
    assert first[0] == second[0] == 200
    assert first[2] == second[2]
    other = speak(qwen3_tts_server, {**body, "seed": 8})
    assert other[2] != first[2]


def test_qwen3_tts_speed_resamples_its_audio(qwen3_tts_server: Server) -> None:
    body = {
        "input": SENTENCE,
        "response_format": "pcm",
        "seed": 3,
        "max_new_tokens": 20,
    }
    status, _, normal = speak(qwen3_tts_server, body)
    assert status == 200
    status, _, faster = speak(qwen3_tts_server, {**body, "speed": 2.0})
    assert status == 200
    assert len(faster) == 2 * round(len(normal) / 2 / 2.0)


def test_qwen3_tts_greedy_stage_params_repeat_without_a_seed(
    qwen3_tts_server: Server,
) -> None:
    body = {
        "input": "hello there",
        "response_format": "pcm",
        "max_new_tokens": 30,
        "stage_params": {
            "tts_engine": {"do_sample": False, "subtalker_dosample": False}
        },
    }
    first = speak(qwen3_tts_server, body)
    second = speak(qwen3_tts_server, body)
    assert first[0] == second[0] == 200
    assert first[2] == second[2]


def test_qwen3_tts_follows_instructions_and_dialects(
    qwen3_tts_server: Server,
) -> None:
    """With one seed, audio differs only through what the prompt embeds, so a
    dropped instruction or dialect id would repeat the plain request's audio."""

    def pcm(body: dict) -> bytes:
        status, _, audio = speak(
            qwen3_tts_server,
            {
                "input": "今天天气真好。",
                "response_format": "pcm",
                "max_new_tokens": 12,
                "seed": 11,
                **body,
            },
        )
        assert status == 200 and audio, body
        return audio

    plain = pcm({"voice": "vivian"})
    assert pcm({"voice": "vivian"}) == plain
    assert pcm({"voice": "vivian", "instructions": "Speak slowly."}) != plain
    # Dylan's dialect id is used for Auto, and an explicit Chinese replaces it.
    assert pcm({"voice": "dylan"}) != pcm({"voice": "dylan", "language": "Chinese"})
    assert pcm({"voice": "eric"}) != pcm({"voice": "eric", "language": "Chinese"})


def test_qwen3_tts_trims_and_lowercases_its_response_format(
    qwen3_tts_server: Server,
) -> None:
    status, headers, _ = speak(
        qwen3_tts_server,
        {"input": "hi", "response_format": "\u3000PCM\u00a0", "max_new_tokens": 1},
    )
    assert (status, headers["Content-Type"]) == (200, "audio/pcm")


def test_qwen3_tts_rejects_another_model_than_its_chosen_name() -> None:
    named_server = Server(
        "qwen3_tts_server",
        "qwen3_tts",
        qwen3_tts_directory(),
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
        (
            {"input": "hi", "voice": "nobody"},
            "Unknown voice 'nobody'. Supported voices: default, serena, vivian, "
            "uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan",
        ),
        ({"input": "hi", "response_format": "mp3"}, "response_format must be one of"),
        ({"input": "hi", "response_format": "p c m"}, "response_format must be one of"),
        ({"input": "hi", "stream": True}, "stream=true requires response_format='pcm'"),
        ({"input": "hi", "max_new_tokens": 0}, "max_new_tokens must be greater than 0"),
        ({"input": "hi", "speed": 4.5}, "speed must be between 0.25 and 4.0"),
        ({"input": "hi", "speed": "fast"}, "speed must be a number"),
        ({"input": "hi", "voice": None}, "voice must be a string"),
        ({"input": "hi", "speaker": None}, "voice must be a string"),
        (
            {"input": "hi", "voice": None, "speaker": "ryan"},
            "voice must be a string",
        ),
        ({"input": "hi", "response_format": None}, "response_format must be a string"),
        ({"input": "hi", "stream": None}, "stream must be a boolean"),
        ({"input": "hi", "stream_format": None}, "stream_format must be audio"),
        ({"input": "hi", "speed": None}, "speed must be a number"),
        ({"input": "hi", "stream_format": "sse"}, "stream_format must be audio"),
        ({"input": "hi", "references": False}, "references must be a list"),
        ({"input": "hi", "references": {}}, "references must be a list"),
        ({"input": "hi", "stage_params": []}, "stage_params must be an object"),
        ({"input": "hi", "ref_text": ""}, "ref_text is not supported by this model"),
        ({"input": "hi", "ref_audio": "a"}, "ref_audio is not supported by this model"),
        (
            {"input": "hi", "x_vector_only_mode": False},
            "x_vector_only_mode is not supported by this model",
        ),
        ({"input": "hi", "references": [{}]}, "references are not supported"),
        ({"input": "hi", "task_type": "Base"}, "task_type must be one of: CustomVoice"),
        (
            {"input": "hi", "task_type": "clone"},
            "task_type must be one of: Base, CustomVoice, VoiceDesign",
        ),
        (
            {"input": "hi", "language": "Klingon"},
            "language must be one of: Auto, Chinese, English, French, German, "
            "Italian, Japanese, Korean, Portuguese, Russian, Spanish",
        ),
        ({"input": "hi", "language": 3}, "language must be a string"),
        ({"input": "hi", "max_new_tokens": 1.5}, "max_new_tokens must be an integer"),
        ({"input": "hi", "temperature": -1}, "temperature must be non-negative"),
        ({"input": "hi", "top_p": 0}, "top_p must be in (0, 1]"),
        ({"input": "hi", "top_k": 0}, "top_k must be -1 (disable) or at least 1"),
        ({"input": "hi", "top_k": True}, "top_k must be an integer"),
        ({"input": "hi", "repetition_penalty": 3}, "repetition_penalty must be in"),
        (
            {"input": "hi", "repetition_penalty": 0},
            "repetition_penalty must be in (0, 2]",
        ),
        (
            {"input": "hi", "instructions": "x" * 4097},
            "instructions must be at most 4096",
        ),
        (
            {"input": "hi", "max_new_tokens": 8190},
            "max_new_tokens must be at most",
        ),
        ({"input": "hi", "seed": -1}, "seed must be greater than or equal to 0"),
        ({"input": "hi", "token_count": 0}, "token_count must be greater than 0"),
        ({"input": "hi", "stream_codec_output": 1}, "stream_codec_output must be"),
        (
            {"input": "hi", "stage_params": {"tts_engine": []}},
            "stage_params.tts_engine must be an object",
        ),
        (
            {"input": "hi", "stage_params": {"tts_engine": {"subtalker_top_k": 0}}},
            "subtalker_top_k must be -1 (disable) or at least 1",
        ),
    ],
)
def test_qwen3_tts_rejects_invalid_requests(
    qwen3_tts_server: Server, body: object, detail: str
) -> None:
    status, response = qwen3_tts_server.request(
        "POST", "/v1/audio/speech", json.dumps(body).encode()
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(detail)


def test_qwen3_tts_refuses_a_declared_body_too_large_without_reading_it(
    qwen3_tts_server: Server,
) -> None:
    # Note (Minyi Tang): the body is never sent, so a server that waits to
    # read it never answers.
    connection = http.client.HTTPConnection(
        "127.0.0.1", qwen3_tts_server.port, timeout=10
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
def test_qwen3_tts_caps_a_chunked_body_it_cannot_size_in_advance(
    qwen3_tts_server: Server, body_bytes: int, status: int, detail: str
) -> None:
    body = json.dumps({"input": "x" * (body_bytes - len('{"input": ""}'))}).encode()
    assert len(body) == body_bytes
    connection = http.client.HTTPConnection(
        "127.0.0.1", qwen3_tts_server.port, timeout=60
    )
    # Note (Minyi Tang): a body of one chunk is read to its end before the
    # answer, so the socket closes with nothing unread and is not reset.
    connection.request("POST", "/v1/audio/speech", body=[body])
    response = connection.getresponse()
    assert response.status == status
    assert json.loads(response.read())["detail"].startswith(detail)
    connection.close()


def test_qwen3_tts_server_serves_no_transcriptions(qwen3_tts_server: Server) -> None:
    # Note (Minyi Tang): civetweb answers a path it does not serve without
    # reading the request's body, and a socket closed with bytes unread is
    # reset, which can drop the answer; so the request has no body.
    status, _ = qwen3_tts_server.request("POST", "/v1/audio/transcriptions")
    assert status == 404
