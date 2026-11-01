# SPDX-License-Identifier: Apache-2.0
"""API tests for fish_s2_pro_server, run against the real model; the speech it
returns is checked by transcribing it with qwen3_asr_server.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_fish_s2_pro_api.py
"""

from __future__ import annotations

import base64
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
FISH_REPO = "mlx-community/fish-audio-s2-pro-8bit"
CHOSEN_MODEL_NAME = "fishaudio/s2-pro"
SAMPLE_RATE = 44100
ASR_SAMPLE_RATE = 16000
# One audio frame is 2048 16-bit samples.
FRAME_BYTES = 2048 * 2
# A corpus clip to clone and its transcript.
REFERENCE_CLIP = "0066_en_mid"
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."
# Note (Jiaxin Deng): with seed 3 this is 129 frames, about 11 s to generate
# on an M5, well inside REQUEST_TIMEOUT_SECONDS.
TWO_SENTENCES = SENTENCE + " The museum opens at nine."
LONG_SENTENCE = " ".join([SENTENCE] * 8)
# Minutes of speech: no Mac generates it within REQUEST_TIMEOUT_SECONDS.
PASSAGE = " ".join([SENTENCE] * 50)
# Note (Minyi Tang): generation samples, so the audio differs on every unseeded
# request; the transcript may miss one word without the speech being wrong.
MAX_WORD_ERRORS = 1
MAX_QUEUED_REQUESTS = 1
REQUEST_TIMEOUT_SECONDS = 20
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
PLAIN_BODY_BYTES = 1024 * 1024
# A cancelled generation stops at its next frame.
CANCEL_SECONDS = 8.0


def fish_directory() -> Path:
    return Path(DATA_ROOT) / "models" / FISH_REPO.replace("/", "_")


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


def wav_data_url(sample_rate: int, seconds: float = 0.5) -> str:
    return data_url(wav_bytes(sample_rate, seconds))


def reference_audio() -> tuple[str, str]:
    """The reference clip as a WAV data: URL, and its transcript."""
    clip = Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{REFERENCE_CLIP}.wav"
    manifest = Path(__file__).resolve().parent / "corpus" / "manifest.jsonl"
    rows = map(json.loads, manifest.read_text().splitlines())
    transcript = next(row["reference"] for row in rows if row["id"] == REFERENCE_CLIP)
    encoded = base64.b64encode(clip.read_bytes()).decode()
    return f"data:audio/wav;base64,{encoded}", transcript.lower()


@pytest.fixture(scope="module")
def fish_server() -> Iterator[Server]:
    running = Server(
        "fish_s2_pro_server",
        "fish_s2_pro",
        fish_directory(),
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


def test_fish_wav_speaks_its_input(fish_server: Server, asr_server: Server) -> None:
    status, headers, audio = speak(
        fish_server,
        {
            "input": SENTENCE,
            "model": fish_server.ready["model_name"],
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
    heard = transcript_words(asr_server, pcm)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_fish_streams_pcm_while_it_speaks(
    fish_server: Server, asr_server: Server
) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", fish_server.port, timeout=300)
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
    # The first chunk is 40 frames of 129: it arrives well before the last.
    assert first_audio_at - started_at < (finished_at - started_at) / 2
    heard = transcript_words(asr_server, pcm)
    assert word_errors(words(TWO_SENTENCES), heard) <= MAX_WORD_ERRORS, heard


def test_fish_stops_at_max_new_tokens(fish_server: Server) -> None:
    status, headers, pcm = speak(
        fish_server,
        {"input": SENTENCE, "response_format": "pcm", "max_new_tokens": 10},
    )
    assert status == 200
    assert (headers["X-Finish-Reason"], headers["X-Completion-Tokens"]) == (
        "length",
        "10",
    )
    assert len(pcm) == 10 * FRAME_BYTES


def test_fish_disconnect_stops_its_generation(fish_server: Server) -> None:
    with open_speech_stream(fish_server, LONG_SENTENCE) as client:
        assert client.recv(65536)
        assert request_states(fish_server) == {"running": 1}
    wait_until_idle(fish_server)


def test_fish_disconnect_stops_a_generation_it_has_not_answered(
    fish_server: Server,
) -> None:
    """A WAV request kept generating after its client disconnected."""
    with send_speech_request(fish_server, {"input": PASSAGE}):
        wait_until_running(fish_server)
    wait_until_idle(fish_server)


def test_fish_disconnect_is_seen_past_bytes_sent_after_the_request(
    fish_server: Server,
) -> None:
    """Bytes a client sent after its request hid that it then disconnected."""
    with send_speech_request(fish_server, {"input": PASSAGE}) as client:
        wait_until_running(fish_server)
        client.sendall(b"GET /health HTTP/1.1\r\nHost: test\r\n\r\n")
    wait_until_idle(fish_server)


def test_fish_half_closed_client_counts_as_gone(fish_server: Server) -> None:
    """uvicorn closes a connection whose client shuts down its sending side, so
    CUDA cancels that request and sends nothing; this server must match it."""
    with send_speech_request(fish_server, {"input": PASSAGE}) as client:
        client.shutdown(socket.SHUT_WR)
        received = b""
        while chunk := client.recv(65536):
            received += chunk
        assert received == b""
    wait_until_idle(fish_server)


def test_fish_refuses_requests_beyond_its_queue(fish_server: Server) -> None:
    # A stream's 200 waits for its first audio, so the queued request is
    # sent without waiting for its answer.
    queued_body = {"input": LONG_SENTENCE, "response_format": "pcm", "stream": True}
    with (
        open_speech_stream(fish_server, LONG_SENTENCE) as running,
        send_speech_request(fish_server, queued_body),
    ):
        assert running.recv(65536)
        assert request_states(fish_server) == {
            "running": 1,
            "queued": MAX_QUEUED_REQUESTS,
        }
        status, _, body = speak(fish_server, {"input": "hi"})
        assert (status, json.loads(body)) == (
            503,
            {"detail": "The request queue is full."},
        )
    # Each close cancels its request, whether it runs or waits in the queue.
    wait_until_idle(fish_server, 2 * CANCEL_SECONDS)


def test_fish_times_out_a_request_past_its_deadline(
    fish_server: Server,
) -> None:
    started_at = time.monotonic()
    status, _, body = speak(fish_server, {"input": PASSAGE, "response_format": "pcm"})
    assert (status, json.loads(body)) == (
        504,
        {"detail": "Speech generation timed out."},
    )
    assert time.monotonic() - started_at >= REQUEST_TIMEOUT_SECONDS
    wait_until_idle(fish_server)


def test_fish_stops_a_stream_its_client_stops_reading_at_its_deadline(
    fish_server: Server,
) -> None:
    # Note (Minyi Tang): seconds of audio fill a 4 KB receive window and the
    # server's send buffer, so the server's write blocks while generation runs.
    with open_speech_stream(fish_server, PASSAGE, receive_buffer_bytes=4096):
        assert request_states(fish_server) == {"running": 1}
        wait_until_idle(fish_server, REQUEST_TIMEOUT_SECONDS + CANCEL_SECONDS)


def test_fish_accepts_any_model_without_a_chosen_name(
    fish_server: Server,
) -> None:
    status, _, _ = speak(
        fish_server,
        {
            "input": "hi",
            "model": "tts-1",
            "response_format": "pcm",
            "max_new_tokens": 1,
        },
    )
    assert status == 200


def test_fish_ignores_unsupported_fields_left_empty(
    fish_server: Server,
) -> None:
    status, _, _ = speak(
        fish_server,
        {
            "input": "hi",
            "response_format": "pcm",
            "max_new_tokens": 1,
            "seed": None,
            "temperature": None,
            "language": "",
            "references": [],
            "stage_params": {},
        },
    )
    assert status == 200


def test_fish_clones_a_reference_voice(fish_server: Server, asr_server: Server) -> None:
    audio, transcript = reference_audio()
    status, headers, pcm = speak(
        fish_server,
        {
            "input": SENTENCE,
            "ref_audio": audio,
            "ref_text": transcript,
            "response_format": "pcm",
        },
    )
    assert (status, headers["X-Finish-Reason"]) == (200, "stop")
    # The prompt holds the reference's 21.5 frames a second.
    assert int(headers["X-Prompt-Tokens"]) > 11 * 21
    heard = transcript_words(asr_server, pcm)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_fish_seed_repeats_its_speech(fish_server: Server) -> None:
    body = {"input": "Good morning.", "response_format": "pcm", "seed": 7}
    first = speak(fish_server, body)
    second = speak(fish_server, body)
    other = speak(fish_server, {**body, "seed": 8})
    assert first[0] == second[0] == other[0] == 200
    assert first[2] == second[2]
    assert first[2] != other[2]


def test_fish_refuses_a_rate_it_cannot_resample(fish_server: Server) -> None:
    """A 44101 Hz reference needed a 15 GB resampling table."""
    status, response = fish_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps({"input": "hi", "ref_audio": wav_data_url(44101)}).encode(),
    )
    assert status == 400
    assert json.loads(response)["detail"] == (
        "reference audio at 44101 Hz cannot be resampled to 44100 Hz"
    )


def test_fish_bounds_all_references_before_resampling(fish_server: Server) -> None:
    """Many short low-rate references each passed the 60 s check and were
    all resampled to 44.1 kHz before their total was checked."""
    reference = {"data": wav_data_url(8000, 20.0)}
    status, response = fish_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps({"input": "hi", "references": [reference] * 4}).encode(),
    )
    assert (status, json.loads(response)) == (
        400,
        {"detail": "the references are longer than 60 s in all"},
    )


@pytest.mark.parametrize(
    "stream_fields",
    [{}, {"stream": True, "response_format": "pcm"}],
    ids=["file", "stream"],
)
def test_fish_answers_a_prompt_too_long_with_400(
    fish_server: Server, stream_fields: dict
) -> None:
    """A prompt past the model's 32768 positions was a 500, and an empty 200
    for a stream."""
    reference = {"vq_codes": [[1]] * 10, "text": "\u4f60" * 4096}
    status, response = fish_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(
            {"input": "hi", "references": [reference] * 9, **stream_fields}
        ).encode(),
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(
        "the prompt and its audio exceed the model's 32768 positions"
    )


@pytest.mark.parametrize(
    "wav",
    [
        # A streamed WAV declares more data than it holds.
        wav_bytes(16000)[:40] + b"\xff\xff\xff\xff" + wav_bytes(16000)[44:],
        wav_bytes(22050, channels=2),
    ],
    ids=["oversized-data-chunk", "stereo"],
)
def test_fish_reads_reference_wavs_cuda_reads(fish_server: Server, wav: bytes) -> None:
    status, _, _ = speak(
        fish_server,
        {
            "input": "hi",
            "ref_audio": data_url(wav),
            "response_format": "pcm",
            "max_new_tokens": 1,
        },
    )
    assert status == 200


def test_fish_answers_a_stream_with_unusable_audio_with_400(
    fish_server: Server,
) -> None:
    """Reference audio decodes once the request is admitted; a stream whose
    audio is not a WAV got an empty 200."""
    status, response = fish_server.request(
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


def test_fish_clones_from_a_reference_at_the_cap(fish_server: Server) -> None:
    """A 60 s reference is encoded in windows, within the server's memory."""
    status, headers, _ = speak(
        fish_server,
        {
            "input": "hi",
            "ref_audio": wav_data_url(16000, 59.9),
            "response_format": "pcm",
            "max_new_tokens": 1,
        },
    )
    assert status == 200
    # About 21.5 frames a second of reference.
    assert int(headers["X-Prompt-Tokens"]) > 59 * 21


def test_fish_takes_reference_codes(fish_server: Server) -> None:
    codes = [[index % 1024 for index in range(20)] for _ in range(10)]
    status, headers, _ = speak(
        fish_server,
        {
            "input": "hi",
            "references": [{"vq_codes": codes, "text": "hello there"}],
            "response_format": "pcm",
            "max_new_tokens": 1,
        },
    )
    assert status == 200
    assert int(headers["X-Prompt-Tokens"]) > 20


def test_fish_caps_max_new_tokens_as_cuda(fish_server: Server) -> None:
    status, headers, _ = speak(
        fish_server,
        {"input": "hi", "response_format": "pcm", "max_new_tokens": 100000},
    )
    assert (status, headers["X-Finish-Reason"]) == (200, "stop")


def test_fish_trims_and_lowercases_its_response_format(
    fish_server: Server,
) -> None:
    status, headers, _ = speak(
        fish_server,
        {"input": "hi", "response_format": "\u3000PCM\u00a0", "max_new_tokens": 1},
    )
    assert (status, headers["Content-Type"]) == (200, "audio/pcm")


def test_fish_rejects_another_model_than_its_chosen_name() -> None:
    named_server = Server(
        "fish_s2_pro_server",
        "fish_s2_pro",
        fish_directory(),
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
        ({"input": "hi", "temperature": -1}, "temperature must be a non-negative"),
        ({"input": "hi", "temperature": "hot"}, "temperature must be a number"),
        ({"input": "hi", "top_p": 0}, "top_p must be in (0, 1]"),
        ({"input": "hi", "top_p": 1.5}, "top_p must be in (0, 1]"),
        ({"input": "hi", "top_k": 0}, "top_k must be -1 or between 1 and 30"),
        ({"input": "hi", "top_k": 31}, "top_k must be -1 or between 1 and 30"),
        ({"input": "hi", "top_k": 1.5}, "top_k must be -1 or between 1 and 30"),
        ({"input": "hi", "top_k": 2**64 - 1}, "top_k must be -1 or between 1 and 30"),
        (
            {
                "input": "hi",
                "ref_audio": data_url(
                    wav_bytes(16000)[:22] + b"\x00\x00" + wav_bytes(16000)[24:]
                ),
            },
            "ref_audio: reference audio WAV has no channels",
        ),
        (
            {"input": "hi", "repetition_penalty": 0},
            "repetition_penalty must be in (0, 2]",
        ),
        ({"input": "hi", "seed": "1"}, "seed must be an integer"),
        ({"input": "hi", "seed": -1}, "seed must be greater than or equal to 0"),
        ({"input": "hi", "seed": 2**64 - 1}, "seed must be in [-2**63, 2**63 - 1]"),
        ({"input": "hi", "ref_text": "x" * 4097}, "ref_text must be at most 4096"),
        (
            {
                "input": "hi",
                "references": [{"vq_codes": [[1]] * 10, "text": "x" * 4097}],
            },
            "references[0].text must be at most 4096",
        ),
        ({"input": "hi", "ref_audio": 5}, "ref_audio must be a string"),
        ({"input": "hi", "ref_audio": "/tmp/a.wav"}, "ref_audio must be a WAV file"),
        ({"input": "hi", "ref_audio": "https://a/b.wav"}, "ref_audio must be a WAV"),
        ({"input": "hi", "ref_audio": "data:audio/wav,abc"}, "ref_audio must be a "),
        ({"input": "hi", "ref_audio": "aGVsbG8="}, "ref_audio: reference audio must"),
        ({"input": "hi", "ref_audio": "@@@"}, "ref_audio is not valid base64"),
        ({"input": "hi", "ref_text": 5}, "ref_text must be a string"),
        ({"input": "hi", "references": [5]}, "references[0] must be an object"),
        ({"input": "hi", "references": [{}]}, "references[0] has no audio"),
        (
            {"input": "hi", "references": [{"audio_path": "/tmp/a.wav"}]},
            "references[0].audio_path is not supported",
        ),
        (
            {"input": "hi", "references": [{"vq_codes": [1, 2, 3]}]},
            "references[0].vq_codes must have shape (10, T)",
        ),
        (
            {"input": "hi", "references": [{"vq_codes": [[4096]] * 10}]},
            "references[0].vq_codes row 0 must be integers in [0, 4096)",
        ),
        (
            {"input": "hi", "references": [{"vq_codes": [[5]] + [[1024]] * 9}]},
            "references[0].vq_codes row 1 must be integers in [0, 1024)",
        ),
        (
            {"input": "hi", "references": [{"data": "", "text": 5}]},
            "references[0].text must be a string",
        ),
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
            {"input": "hi", "x_vector_only_mode": False, "language": "en"},
            "Fish S2-Pro does not support these /v1/audio/speech fields: "
            "language, x_vector_only_mode. ",
        ),
    ],
)
def test_fish_rejects_invalid_requests(
    fish_server: Server, body: object, detail: str
) -> None:
    status, response = fish_server.request(
        "POST", "/v1/audio/speech", json.dumps(body).encode()
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(detail)


def test_fish_refuses_a_declared_body_too_large_without_reading_it(
    fish_server: Server,
) -> None:
    # Note (Minyi Tang): the body is never sent, so a server that waits to
    # read it never answers.
    connection = http.client.HTTPConnection("127.0.0.1", fish_server.port, timeout=10)
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
    ("body_bytes", "with_references", "status", "detail"),
    [
        (PLAIN_BODY_BYTES, False, 400, "input must be at most 4096 characters"),
        (
            PLAIN_BODY_BYTES + 1,
            False,
            413,
            f"request body must be at most {PLAIN_BODY_BYTES} bytes without "
            "ref_audio or references",
        ),
        (MAX_REQUEST_BODY_BYTES, True, 400, "input must be at most 4096 characters"),
        (
            MAX_REQUEST_BODY_BYTES + 1,
            True,
            413,
            f"request body must be at most {MAX_REQUEST_BODY_BYTES} bytes",
        ),
    ],
)
def test_fish_caps_a_chunked_body_it_cannot_size_in_advance(
    fish_server: Server,
    body_bytes: int,
    with_references: bool,
    status: int,
    detail: str,
) -> None:
    """Without reference audio a body holds 1 MiB; with it, 16 MiB."""
    prefix = '{"references": [], "input": "' if with_references else '{"input": "'
    body = (prefix + "x" * (body_bytes - len(prefix) - 2) + '"}').encode()
    assert len(body) == body_bytes
    connection = http.client.HTTPConnection("127.0.0.1", fish_server.port, timeout=60)
    # Note (Minyi Tang): a body of one chunk is read to its end before the
    # answer, so the socket closes with nothing unread and is not reset.
    connection.request("POST", "/v1/audio/speech", body=[body])
    response = connection.getresponse()
    assert response.status == status
    assert json.loads(response.read())["detail"].startswith(detail)
    connection.close()


def test_fish_bounds_the_json_values_it_parses(fish_server: Server) -> None:
    """A 16 MiB body of small vq_codes built a document of millions of values
    before the request was admitted."""
    codes = [[0] * 7000 for _ in range(10)]
    status, response = fish_server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps({"input": "hi", "references": [{"vq_codes": codes}]}).encode(),
    )
    assert (status, json.loads(response)) == (
        400,
        {"detail": "speech request body must hold at most 65536 JSON values"},
    )


def test_fish_server_serves_no_transcriptions(fish_server: Server) -> None:
    # Note (Minyi Tang): civetweb answers a path it does not serve without
    # reading the request's body, and a socket closed with bytes unread is
    # reset, which can drop the answer; so the request has no body.
    status, _ = fish_server.request("POST", "/v1/audio/transcriptions")
    assert status == 404
