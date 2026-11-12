# SPDX-License-Identifier: Apache-2.0
"""API tests for zonos3_server, run against the real model; the speech it returns
is checked by transcribing it with qwen3_asr_server.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_zonos3_api.py
"""

from __future__ import annotations

import base64
import http.client
import io
import json
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
    word_errors,
    words,
)
from test_server_api import DATA_ROOT, RUNTIME_BIN, Server

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
ZONOS3_REPO = "mlx-community/ZONOS3-Mini"
SAMPLE_RATE = 44100
# One codec frame is 2940 16-bit samples.
FRAME_BYTES = 2940 * 2
CODEBOOK_COUNT = 10
SILENCE_PREFILL_FRAMES = 3
SPEAKER_EMBEDDING_SIZE = 192
REFERENCE_CLIP = "0066_en_mid"
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."
TWO_SENTENCES = SENTENCE + " The museum opens at nine."
LONG_SENTENCE = " ".join([SENTENCE] * 8)
# Minutes of speech: no Mac generates it within REQUEST_TIMEOUT_SECONDS.
PASSAGE = " ".join([SENTENCE] * 50)
# Generation samples, so a transcript may miss one word without the speech being
# wrong; the requests checked by ear are seeded so the check is repeatable.
MAX_WORD_ERRORS = 1
REQUEST_TIMEOUT_SECONDS = 10
MAX_QUEUED_REQUESTS = 1
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
# A cancelled generation stops at its next row.
CANCEL_SECONDS = 8.0


def zonos3_directory() -> Path:
    return Path(DATA_ROOT) / "models" / ZONOS3_REPO.replace("/", "_")


def wav_bytes(sample_rate: int, seconds: float = 0.5, channels: int = 1) -> bytes:
    """A tone as a 16-bit WAV file."""
    times = np.arange(int(sample_rate * seconds)) / sample_rate
    tone = (np.sin(2 * np.pi * 220 * times) * 8000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(np.repeat(tone, channels).tobytes())
    return buffer.getvalue()


def data_url(wav: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav).decode()


def reference_audio() -> str:
    """The reference clip as a WAV data: URL."""
    clip = Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{REFERENCE_CLIP}.wav"
    return data_url(clip.read_bytes())


@pytest.fixture(scope="module")
def zonos3_server() -> Iterator[Server]:
    running = Server(
        "zonos3_server",
        "zonos3",
        zonos3_directory(),
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


def test_zonos3_wav_speaks_its_input(zonos3_server: Server, asr_server: Server) -> None:
    status, headers, audio = speak(
        zonos3_server,
        {"input": SENTENCE, "model": zonos3_server.ready["model_name"], "seed": 1},
    )
    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert headers["X-Finish-Reason"] == "stop"
    row_count = int(headers["X-Completion-Tokens"])
    assert int(headers["X-SGLang-Omni-Total-Tokens"]) == (
        int(headers["X-Prompt-Tokens"]) + row_count
    )
    with wave.open(io.BytesIO(audio)) as reader:
        assert (reader.getnchannels(), reader.getsampwidth()) == (1, 2)
        assert reader.getframerate() == SAMPLE_RATE
        pcm = reader.readframes(reader.getnframes())
    # Frame f is complete once its last codebook's row is generated.
    frame_count = len(pcm) // FRAME_BYTES
    assert len(pcm) == frame_count * FRAME_BYTES
    assert 0 < frame_count <= SILENCE_PREFILL_FRAMES + row_count - CODEBOOK_COUNT + 1
    heard = transcript_words(asr_server, pcm, SAMPLE_RATE)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_zonos3_streams_pcm_while_it_speaks(
    zonos3_server: Server, asr_server: Server
) -> None:
    connection = http.client.HTTPConnection(
        "127.0.0.1", zonos3_server.port, timeout=300
    )
    started_at = time.monotonic()
    connection.request(
        "POST",
        "/v1/audio/speech",
        body=json.dumps(
            {
                "input": TWO_SENTENCES,
                "response_format": "pcm",
                "stream": True,
                "seed": 3,
            }
        ),
    )
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Type") == "audio/pcm"
    assert response.getheader("X-Sample-Rate") == str(SAMPLE_RATE)
    chunks = []
    first_audio_at = None
    while chunk := response.read1(65536):
        first_audio_at = first_audio_at or time.monotonic()
        chunks.append(chunk)
    finished_at = time.monotonic()
    pcm = b"".join(chunks)
    assert pcm and len(pcm) % FRAME_BYTES == 0
    # Audio comes every 8 frames, so the first chunk is far ahead of the last.
    assert first_audio_at - started_at < (finished_at - started_at) / 2
    heard = transcript_words(asr_server, pcm, SAMPLE_RATE)
    assert word_errors(words(TWO_SENTENCES), heard) <= MAX_WORD_ERRORS, heard


def test_zonos3_stream_and_wav_give_the_same_speech(zonos3_server: Server) -> None:
    """The streamed decode carries each layer's state, so it gives the whole
    decode's samples."""
    body = {"input": "Good morning.", "response_format": "pcm", "seed": 5}
    whole = speak(zonos3_server, body)
    streamed = speak(zonos3_server, {**body, "stream": True})
    assert whole[0] == streamed[0] == 200
    expected = np.frombuffer(whole[2], dtype="<i2").astype(np.int32)
    actual = np.frombuffer(streamed[2], dtype="<i2").astype(np.int32)
    assert len(actual) == len(expected)
    # The streamed iSTFT runs in float64 on the host; the whole decode in float32.
    assert np.abs(actual - expected).max() <= 2


def test_zonos3_stops_at_max_new_tokens(zonos3_server: Server) -> None:
    status, headers, pcm = speak(
        zonos3_server,
        {"input": SENTENCE, "response_format": "pcm", "max_new_tokens": 10},
    )
    assert status == 200
    assert (headers["X-Finish-Reason"], headers["X-Completion-Tokens"]) == (
        "length",
        "10",
    )
    # Three opening silence rows and ten generated rows complete four frames.
    frame_count = SILENCE_PREFILL_FRAMES + 10 - CODEBOOK_COUNT + 1
    assert len(pcm) == frame_count * FRAME_BYTES


def test_zonos3_answers_a_budget_shorter_than_a_frame_with_empty_audio(
    zonos3_server: Server,
) -> None:
    """Codebook 9 lags its frame by nine rows, so one generated row completes no
    frame; that request got a 500."""
    status, headers, pcm = speak(
        zonos3_server, {"input": "hi", "response_format": "pcm", "max_new_tokens": 1}
    )
    assert (status, headers["X-Finish-Reason"], pcm) == (200, "length", b"")


def test_zonos3_seed_repeats_its_speech(zonos3_server: Server) -> None:
    body = {"input": "Good morning.", "response_format": "pcm", "seed": 7}
    first = speak(zonos3_server, body)
    second = speak(zonos3_server, body)
    other = speak(zonos3_server, {**body, "seed": 8})
    assert first[0] == second[0] == other[0] == 200
    assert first[2] == second[2]
    assert first[2] != other[2]


@pytest.mark.parametrize(
    "fields",
    [
        {"voice": "MEILIN"},
        {"voice": "none"},
        {"speaker": "imogen"},
        {"speed": 1.4},
        {"condition_values": {"loudness": -20, "fmax": 20000, "speaking_rate": 16}},
        {"condition_buckets": {"loudness": 3, "speaking_rate": 30}},
        {"prepend_silence": False},
        {"language": "zh"},
        {"temperature": 0, "top_k": 1, "top_p": 1, "repetition_penalty": 1},
    ],
    ids=[
        "voice-any-case",
        "no-voice",
        "speaker-alias",
        "speed",
        "condition-values",
        "condition-buckets",
        "no-silence-prefill",
        "language",
        "greedy",
    ],
)
def test_zonos3_takes_its_model_fields(
    zonos3_server: Server, fields: dict[str, object]
) -> None:
    status, headers, pcm = speak(
        zonos3_server,
        {
            "input": "Good morning.",
            "response_format": "pcm",
            "max_new_tokens": 12,
            "seed": 1,
            **fields,
        },
    )
    assert status == 200
    assert pcm and len(pcm) % FRAME_BYTES == 0
    assert int(headers["X-Completion-Tokens"]) <= 12


def test_zonos3_prompt_rows_follow_the_voice(zonos3_server: Server) -> None:
    """A speaker takes one prompt row; no voice and no opening silence take none."""
    body = {"input": "Good morning.", "response_format": "pcm", "max_new_tokens": 1}
    rows = {
        name: int(speak(zonos3_server, {**body, **fields})[1]["X-Prompt-Tokens"])
        for name, fields in {
            "speaker": {},
            "none": {"voice": "none"},
            "no-silence": {"prepend_silence": False},
        }.items()
    }
    assert rows["speaker"] == rows["none"] + 1
    assert rows["speaker"] == rows["no-silence"] + SILENCE_PREFILL_FRAMES


def test_zonos3_clones_a_reference_voice(
    zonos3_server: Server, asr_server: Server
) -> None:
    status, headers, pcm = speak(
        zonos3_server,
        {
            "input": SENTENCE,
            "ref_audio": reference_audio(),
            "response_format": "pcm",
            "seed": 2,
        },
    )
    assert (status, headers["X-Finish-Reason"]) == (200, "stop")
    heard = transcript_words(asr_server, pcm, SAMPLE_RATE)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_zonos3_reference_audio_and_its_embedding_give_one_voice(
    zonos3_server: Server,
) -> None:
    """references[0].audio and ref_audio are the same voice; a speaker_embedding
    takes the place of audio."""
    body = {"input": "Good morning.", "response_format": "pcm", "seed": 4}
    audio = reference_audio()
    flat = speak(zonos3_server, {**body, "ref_audio": audio})
    listed = speak(zonos3_server, {**body, "references": [{"audio": audio}]})
    assert flat[0] == listed[0] == 200
    assert flat[2] == listed[2]
    embedding = [0.0] * SPEAKER_EMBEDDING_SIZE
    status, _, _ = speak(
        zonos3_server,
        {**body, "references": [{"speaker_embedding": embedding}], "max_new_tokens": 1},
    )
    assert status == 200


@pytest.mark.parametrize(
    "wav",
    [
        wav_bytes(16000)[:40] + b"\xff\xff\xff\xff" + wav_bytes(16000)[44:],
        wav_bytes(22050, channels=2),
    ],
    ids=["oversized-data-chunk", "stereo"],
)
def test_zonos3_reads_reference_wavs_cuda_reads(
    zonos3_server: Server, wav: bytes
) -> None:
    status, _, _ = speak(
        zonos3_server,
        {
            "input": "hi",
            "ref_audio": data_url(wav),
            "response_format": "pcm",
            "max_new_tokens": 1,
        },
    )
    assert status == 200


def test_zonos3_refuses_a_rate_it_cannot_resample(zonos3_server: Server) -> None:
    status, response = zonos3_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps({"input": "hi", "ref_audio": data_url(wav_bytes(44101))}).encode(),
    )
    assert status == 400
    assert json.loads(response)["detail"] == (
        "reference WAV sample rate 44101 Hz is not supported"
    )


def test_zonos3_answers_a_stream_with_unusable_audio_with_400(
    zonos3_server: Server,
) -> None:
    """A stream answers 200 once it is admitted, so its reference audio is
    decoded first."""
    status, response = zonos3_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(
            {
                "input": "hi",
                "ref_audio": "aGVsbG8=",
                "stream": True,
                "response_format": "pcm",
            }
        ).encode(),
    )
    assert (status, json.loads(response)) == (
        400,
        {"detail": "ref_audio: reference audio must be a RIFF/WAVE file"},
    )


@pytest.mark.parametrize("max_new_tokens", [100000, 2**64 - 1])
def test_zonos3_takes_max_new_tokens_past_its_cap(
    zonos3_server: Server, max_new_tokens: int
) -> None:
    """The context caps the budget without an error, even past int64."""
    status, headers, _ = speak(
        zonos3_server,
        {
            "input": "hi",
            "response_format": "pcm",
            "max_new_tokens": max_new_tokens,
            "seed": 1,
        },
    )
    assert (status, headers["X-Finish-Reason"]) == (200, "stop")


def test_zonos3_ignores_unsupported_fields_left_empty(zonos3_server: Server) -> None:
    status, _, _ = speak(
        zonos3_server,
        {
            "input": "hi",
            "response_format": "pcm",
            "max_new_tokens": 1,
            "seed": None,
            "temperature": None,
            "instructions": "",
            "task_type": "",
            "references": [],
            "stage_params": {},
        },
    )
    assert status == 200


def test_zonos3_disconnect_stops_its_generation(zonos3_server: Server) -> None:
    with open_speech_stream(zonos3_server, {"input": LONG_SENTENCE}) as client:
        assert client.recv(65536)
        assert request_states(zonos3_server) == {"running": 1}
    wait_until_idle(zonos3_server, CANCEL_SECONDS)


def test_zonos3_refuses_requests_beyond_its_queue(zonos3_server: Server) -> None:
    """A queued stream gets its 200 only with its first audio, so the queued
    request is sent without waiting for an answer."""
    queued_body = {"input": PASSAGE, "response_format": "pcm", "stream": True}
    with (
        open_speech_stream(zonos3_server, {"input": PASSAGE}) as running,
        send_speech_request(zonos3_server, queued_body),
    ):
        assert running.recv(65536)
        started_at = time.monotonic()
        while request_states(zonos3_server) != {
            "running": 1,
            "queued": MAX_QUEUED_REQUESTS,
        }:
            assert time.monotonic() - started_at < CANCEL_SECONDS
            time.sleep(0.05)
        status, _, body = speak(zonos3_server, {"input": "hi"})
        assert (status, json.loads(body)) == (
            503,
            {"detail": "The request queue is full."},
        )
    wait_until_idle(zonos3_server, 2 * CANCEL_SECONDS)


def test_zonos3_times_out_a_request_past_its_deadline(zonos3_server: Server) -> None:
    started_at = time.monotonic()
    status, _, body = speak(
        zonos3_server,
        {"input": PASSAGE, "response_format": "pcm", "max_new_tokens": 30000},
    )
    assert (status, json.loads(body)) == (
        504,
        {"detail": "Speech generation timed out."},
    )
    assert time.monotonic() - started_at >= REQUEST_TIMEOUT_SECONDS
    wait_until_idle(zonos3_server, CANCEL_SECONDS)


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"input": "  "}, "input must be a non-empty string"),
        ({"input": "　  "}, "input must be a non-empty string"),
        ({"input": "x" * 4097}, "input must be at most 4096 characters"),
        ({"input": "hi", "voice": "nobody"}, "Unknown voice 'nobody'. "),
        ({"input": "hi", "voice": None}, "voice must be a string"),
        ({"input": "hi", "speaker": None}, "voice must be a string"),
        ({"input": "hi", "temperature": -1}, "temperature must be >= 0"),
        ({"input": "hi", "temperature": "hot"}, "temperature must be a number"),
        (
            {"input": "hi", "temperature": 1e300},
            "temperature must fit in a finite float32 number",
        ),
        ({"input": "hi", "top_p": 0}, "top_p must be in (0, 1]"),
        ({"input": "hi", "top_p": 1.5}, "top_p must be in (0, 1]"),
        ({"input": "hi", "top_k": 0}, "top_k must be in [1, 4098]"),
        ({"input": "hi", "top_k": 4099}, "top_k must be in [1, 4098]"),
        ({"input": "hi", "top_k": 1.5}, "top_k must be in [1, 4098]"),
        ({"input": "hi", "top_k": 2**64 - 1}, "top_k must be in [1, 4098]"),
        ({"input": "hi", "repetition_penalty": 0.5}, "repetition_penalty must be >= 1"),
        ({"input": "hi", "seed": "1"}, "seed must be an integer in [0, 2**63)"),
        ({"input": "hi", "seed": -1}, "seed must be an integer in [0, 2**63)"),
        ({"input": "hi", "seed": 2**63}, "seed must be an integer in [0, 2**63)"),
        ({"input": "hi", "max_new_tokens": 0}, "max_new_tokens must be greater than 0"),
        ({"input": "hi", "max_new_tokens": "9"}, "max_new_tokens must be an integer"),
        ({"input": "hi", "speed": 0}, "speed must be positive"),
        ({"input": "hi", "speed": "fast"}, "speed must be a number"),
        (
            {"input": "hi", "speed": 1.5, "condition_values": {"loudness": -20}},
            "speed cannot be combined with condition_buckets or condition_values",
        ),
        (
            {"input": "hi", "condition_buckets": {}, "condition_values": {}},
            "condition_buckets and condition_values are mutually exclusive",
        ),
        (
            {"input": "hi", "condition_buckets": []},
            "condition_buckets must be an object",
        ),
        ({"input": "hi", "condition_buckets": {"pitch": 1}}, "unknown condition pitch"),
        (
            {"input": "hi", "condition_buckets": {"loudness": 11}},
            "loudness bucket must be in [0, 10]",
        ),
        (
            {"input": "hi", "condition_buckets": {"loudness": 1.5}},
            "loudness bucket must be in [0, 10]",
        ),
        (
            {"input": "hi", "condition_values": {"fmax": "high"}},
            "condition fmax must be a number",
        ),
        (
            {"input": "hi", "prepend_silence": "yes"},
            "prepend_silence must be a boolean",
        ),
        ({"input": "hi", "language": 5}, "language must be a string"),
        ({"input": "hi", "references": {}}, "references must be a list"),
        ({"input": "hi", "references": [5]}, "references[0] must be an object"),
        (
            {"input": "hi", "references": [{}]},
            "references[0] has no audio or speaker_embedding",
        ),
        (
            {"input": "hi", "references": [{"data": "a"}, {"data": "b"}]},
            "at most one reference is supported",
        ),
        (
            {"input": "hi", "ref_audio": "aGVsbG8=", "references": [{"data": "a"}]},
            "at most one reference is supported",
        ),
        (
            {"input": "hi", "references": [{"vq_codes": [[1]] * 10}]},
            "references[0].vq_codes is not supported by this model",
        ),
        (
            {"input": "hi", "references": [{"audio_path": "/tmp/a.wav"}]},
            "references[0].audio_path is not supported",
        ),
        (
            {"input": "hi", "references": [{"speaker_embedding": [0.5]}]},
            "references[0].speaker_embedding must be 192 numbers",
        ),
        (
            {"input": "hi", "references": [{"speaker_embedding": ["x"] * 192}]},
            "references[0].speaker_embedding must be finite numbers",
        ),
        ({"input": "hi", "references": [{"audio": 5}]}, "references[0].audio must be"),
        ({"input": "hi", "ref_audio": 5}, "ref_audio must be a string"),
        ({"input": "hi", "ref_audio": "/tmp/a.wav"}, "ref_audio must be a WAV file"),
        ({"input": "hi", "ref_audio": "https://a/b.wav"}, "ref_audio must be a WAV"),
        ({"input": "hi", "ref_audio": "@@@"}, "ref_audio is not valid base64"),
        ({"input": "hi", "stage_params": []}, "stage_params must be an object"),
        (
            {"input": "hi", "x_vector_only_mode": False, "instructions": "loud"},
            "ZONOS3 does not support these /v1/audio/speech fields: instructions, "
            "x_vector_only_mode",
        ),
    ],
)
def test_zonos3_rejects_invalid_requests(
    zonos3_server: Server, body: dict[str, object], detail: str
) -> None:
    status, response = zonos3_server.request(
        "POST", "/v1/audio/speech", json.dumps(body).encode()
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(detail)


def test_zonos3_refuses_a_declared_body_too_large_without_reading_it(
    zonos3_server: Server,
) -> None:
    # The body is never sent, so a server that waits to read it never answers.
    connection = http.client.HTTPConnection("127.0.0.1", zonos3_server.port, timeout=10)
    connection.putrequest("POST", "/v1/audio/speech")
    connection.putheader("Content-Length", str(MAX_REQUEST_BODY_BYTES + 1))
    connection.endheaders()
    response = connection.getresponse()
    assert (response.status, json.loads(response.read())) == (
        413,
        {"detail": f"request body must be at most {MAX_REQUEST_BODY_BYTES} bytes"},
    )
    connection.close()
