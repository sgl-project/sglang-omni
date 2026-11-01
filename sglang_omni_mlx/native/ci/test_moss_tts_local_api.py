# SPDX-License-Identifier: Apache-2.0
"""API tests for moss_tts_local_server, run against the real model; the speech it
returns is checked by transcribing it with qwen3_asr_server.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_moss_tts_local_api.py
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

import numpy as np
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
MOSS_REPO = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
CHOSEN_MODEL_NAME = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
SAMPLE_RATE = 48000
# One audio frame is 80 ms: 3840 samples a channel, 16-bit.
FRAME_SAMPLES = 3840
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."
LONG_SENTENCE = " ".join([SENTENCE] * 8)
# Minutes of speech: no Mac generates it within REQUEST_TIMEOUT_SECONDS.
PASSAGE = " ".join([SENTENCE] * 50)
# Note (Jiaxin Deng): seeded, so each request is deterministic on one chip; the
# transcript may still miss one word without the speech being wrong.
MAX_WORD_ERRORS = 1
SEED = 1
MAX_QUEUED_REQUESTS = 1
REQUEST_TIMEOUT_SECONDS = 20
MAX_REQUEST_BODY_BYTES = 1024 * 1024
# A cancelled generation stops within two 25-frame codec chunks.
CANCEL_SECONDS = 8.0


def model_directory(repo: str) -> Path:
    return Path(DATA_ROOT) / "models" / repo.replace("/", "_")


def moss_flags(*extra: str) -> tuple[str, ...]:
    return ("--codec-path", str(model_directory(CODEC_REPO)), *extra)


@pytest.fixture(scope="module")
def moss_server() -> Iterator[Server]:
    running = Server(
        "moss_tts_local_server",
        "moss_tts_local",
        model_directory(MOSS_REPO),
        moss_flags(
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


def test_moss_wav_speaks_its_input_in_stereo(
    moss_server: Server, asr_server: Server
) -> None:
    status, headers, audio = speak(
        moss_server,
        {
            "input": SENTENCE,
            "seed": SEED,
            "language": "english",
            "model": moss_server.ready["model_name"],
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
        assert (reader.getnchannels(), reader.getsampwidth()) == (2, 2)
        assert reader.getframerate() == SAMPLE_RATE
        stereo = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    assert len(stereo) == frame_count * FRAME_SAMPLES * 2
    mono = stereo.reshape(-1, 2).mean(axis=1).astype("<i2").tobytes()
    heard = transcript_words(asr_server, mono, SAMPLE_RATE)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_moss_streams_the_mono_pcm_of_its_wav(moss_server: Server) -> None:
    """A streamed request decodes in chunks, a WAV one in the same chunks; the same
    seed gives the same speech, downmixed to mono for PCM."""
    connection = http.client.HTTPConnection("127.0.0.1", moss_server.port, timeout=300)
    started_at = time.monotonic()
    connection.request(
        "POST",
        "/v1/audio/speech",
        body=json.dumps(
            {"input": SENTENCE, "response_format": "pcm", "stream": True, "seed": SEED}
        ),
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
    streamed = b"".join(chunks)
    # The first chunk is five frames: it arrives well before the last.
    assert first_audio_at - started_at < (finished_at - started_at) / 2
    status, _, pcm = speak(
        moss_server, {"input": SENTENCE, "response_format": "pcm", "seed": SEED}
    )
    assert status == 200
    assert streamed == pcm
    _, _, audio = speak(moss_server, {"input": SENTENCE, "seed": SEED})
    with wave.open(io.BytesIO(audio)) as reader:
        stereo = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    assert len(stereo) == len(pcm)


def test_moss_stops_at_max_new_tokens(moss_server: Server) -> None:
    status, headers, pcm = speak(
        moss_server,
        {
            "input": SENTENCE,
            "response_format": "pcm",
            "max_new_tokens": 10,
            "seed": SEED,
        },
    )
    assert status == 200
    assert (headers["X-Finish-Reason"], headers["X-Completion-Tokens"]) == (
        "length",
        "10",
    )
    assert len(pcm) == 10 * FRAME_SAMPLES * 2


def test_moss_speed_resamples_to_the_cuda_length(moss_server: Server) -> None:
    status, _, pcm = speak(
        moss_server,
        {
            "input": SENTENCE,
            "response_format": "pcm",
            "max_new_tokens": 9,
            "speed": 2.0,
            "seed": SEED,
        },
    )
    assert status == 200
    assert len(pcm) == round(9 * FRAME_SAMPLES / 2.0) * 2


def test_moss_disconnect_stops_its_generation(moss_server: Server) -> None:
    with open_speech_stream(moss_server, {"input": LONG_SENTENCE}) as client:
        assert client.recv(65536)
        assert request_states(moss_server) == {"running": 1}
    wait_until_idle(moss_server, CANCEL_SECONDS)


def test_moss_disconnect_stops_a_generation_it_has_not_answered(
    moss_server: Server,
) -> None:
    with send_speech_request(moss_server, {"input": PASSAGE}):
        wait_until_running(moss_server, CANCEL_SECONDS)
    wait_until_idle(moss_server, CANCEL_SECONDS)


def test_moss_half_closed_client_counts_as_gone(moss_server: Server) -> None:
    with send_speech_request(moss_server, {"input": PASSAGE}) as client:
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        assert received == b""
    wait_until_idle(moss_server, CANCEL_SECONDS)


def test_moss_refuses_requests_beyond_its_queue(moss_server: Server) -> None:
    # A queued stream gets its 200 only once it starts, as on CUDA.
    with (
        open_speech_stream(moss_server, {"input": LONG_SENTENCE}) as running,
        send_speech_request(
            moss_server,
            {"input": LONG_SENTENCE, "response_format": "pcm", "stream": True},
        ),
    ):
        assert running.recv(65536)
        started_at = time.monotonic()
        while request_states(moss_server) != {
            "running": 1,
            "queued": MAX_QUEUED_REQUESTS,
        }:
            assert time.monotonic() - started_at < CANCEL_SECONDS
            time.sleep(0.05)
        status, _, body = speak(moss_server, {"input": "hi"})
        assert (status, json.loads(body)) == (
            503,
            {"detail": "The request queue is full."},
        )
    wait_until_idle(moss_server, 2 * CANCEL_SECONDS)


def test_moss_times_out_a_request_past_its_deadline(moss_server: Server) -> None:
    started_at = time.monotonic()
    status, _, body = speak(moss_server, {"input": PASSAGE, "response_format": "pcm"})
    assert (status, json.loads(body)) == (
        504,
        {"detail": "Speech generation timed out."},
    )
    assert time.monotonic() - started_at >= REQUEST_TIMEOUT_SECONDS
    wait_until_idle(moss_server, CANCEL_SECONDS)


def test_moss_stops_a_stream_its_client_stops_reading_at_its_deadline(
    moss_server: Server,
) -> None:
    with open_speech_stream(moss_server, {"input": PASSAGE}, receive_buffer_bytes=4096):
        assert request_states(moss_server) == {"running": 1}
        wait_until_idle(moss_server, REQUEST_TIMEOUT_SECONDS + CANCEL_SECONDS)


def test_moss_accepts_the_cuda_fields_it_passes_over(moss_server: Server) -> None:
    status, _, _ = speak(
        moss_server,
        {
            "input": "hi",
            "model": "tts-1",
            "voice": "alloy",
            "task_type": "voice_design",
            "ref_text": "unused without reference audio",
            "references": [],
            "stage_params": {},
            "x_vector_only_mode": True,
            "response_format": "　PCM ",
            "max_new_tokens": 1,
            "seed": SEED,
        },
    )
    assert status == 200


def test_moss_rejects_another_model_than_its_chosen_name() -> None:
    named_server = Server(
        "moss_tts_local_server",
        "moss_tts_local",
        model_directory(MOSS_REPO),
        moss_flags("--model-name", CHOSEN_MODEL_NAME),
    )
    try:
        status, _, _ = speak(
            named_server,
            {
                "input": "hi",
                "model": CHOSEN_MODEL_NAME,
                "response_format": "pcm",
                "max_new_tokens": 1,
                "seed": SEED,
            },
        )
        assert status == 200
        status, _, body = speak(named_server, {"input": "hi", "model": "tts-1"})
        assert (status, json.loads(body)) == (404, {"detail": "Unknown model: tts-1"})
    finally:
        named_server.stop()


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ([1], "speech request body must be a JSON object"),
        ({"input": "  "}, "input must be a non-empty string"),
        ({"input": "　  "}, "input must be a non-empty string"),
        ({"input": "x" * 4097}, "input must be at most 4096 characters"),
        ({"input": "hi", "model": 5}, "model must be a string"),
        ({"input": "hi", "language": 5}, "language must be a string"),
        ({"input": "hi", "language": "Klingon"}, "language must be one of: Arabic, "),
        ({"input": "hi", "task_type": "Clone"}, "task_type must be one of: Base, "),
        ({"input": "hi", "response_format": "mp3"}, "response_format must be one of"),
        ({"input": "hi", "stream": True}, "stream=true requires response_format='pcm'"),
        ({"input": "hi", "stream_format": "sse"}, "stream_format must be audio"),
        ({"input": "hi", "max_new_tokens": 0}, "max_new_tokens must be greater than 0"),
        (
            {"input": "hi", "max_new_tokens": 4097},
            "max_new_tokens must be at most 4096",
        ),
        (
            {
                "input": "hi",
                "response_format": "pcm",
                "stream": True,
                "max_new_tokens": 22501,
            },
            "max_new_tokens must be at most 22500",
        ),
        (
            {"input": "hi", "instructions": "x" * 4097},
            "instructions must be at most 4096 characters",
        ),
        ({"input": "hi", "max_new_tokens": 1.5}, "max_new_tokens must be an integer"),
        ({"input": "hi", "max_new_tokens": True}, "max_new_tokens must be an integer"),
        ({"input": "hi", "token_count": -1}, "token_count must be greater than 0"),
        ({"input": "${token:0} hi"}, "the ${token:N} count must be greater than 0"),
        ({"input": "${token:x} hi"}, "the ${token:N} prefix must hold"),
        ({"input": "${token:22501} hi"}, "the ${token:N} count must be at most"),
        ({"input": "hi", "seed": 2**64}, "seed must be an integer"),
        ({"input": "hi", "x_vector_only_mode": "yes"}, "x_vector_only_mode must be"),
        ({"input": "hi", "seed": -1}, "seed must be greater than or equal to 0"),
        ({"input": "hi", "temperature": "hot"}, "temperature must be a number"),
        ({"input": "hi", "temperature": -0.1}, "temperature must be >= 0"),
        ({"input": "hi", "top_p": 0}, "top_p must be in (0, 1]"),
        ({"input": "hi", "top_k": -2}, "top_k must be >= -1"),
        ({"input": "hi", "repetition_penalty": 0}, "repetition_penalty must be > 0"),
        ({"input": "hi", "speed": 5}, "speed must be between 0.25 and 4.0"),
        ({"input": "hi", "speed": None}, "speed must be a number"),
        ({"input": "hi", "voice": None}, "voice must be a string"),
        ({"input": "hi", "response_format": None}, "response_format must be a string"),
        ({"input": "hi", "stream": None}, "stream must be a boolean"),
        ({"input": "hi", "ref_audio": "voice.wav"}, "reference audio (ref_audio, "),
        (
            {"input": "hi", "references": [{"audio_path": "voice.wav"}]},
            "reference audio (ref_audio, ",
        ),
    ],
)
def test_moss_rejects_invalid_requests(
    moss_server: Server, body: object, detail: str
) -> None:
    status, response = moss_server.request(
        "POST", "/v1/audio/speech", json.dumps(body).encode()
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(detail)


def test_moss_refuses_a_declared_body_too_large_without_reading_it(
    moss_server: Server,
) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", moss_server.port, timeout=10)
    connection.putrequest("POST", "/v1/audio/speech")
    connection.putheader("Content-Length", str(MAX_REQUEST_BODY_BYTES + 1))
    connection.endheaders()
    response = connection.getresponse()
    assert (response.status, json.loads(response.read())) == (
        413,
        {"detail": f"request body must be at most {MAX_REQUEST_BODY_BYTES} bytes"},
    )
    connection.close()


def test_moss_server_serves_no_transcriptions(moss_server: Server) -> None:
    status, _ = moss_server.request("POST", "/v1/audio/transcriptions")
    assert status == 404
