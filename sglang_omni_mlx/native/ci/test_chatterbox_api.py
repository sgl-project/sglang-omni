# SPDX-License-Identifier: Apache-2.0
"""API tests for chatterbox_server, run against the real model; the speech it
returns is checked by transcribing it with qwen3_asr_server.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_chatterbox_api.py
"""

from __future__ import annotations

import base64
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
from test_server_api import DATA_ROOT, RUNTIME_BIN, Server
from test_voxtral_tts_api import transcript_words, word_errors, words

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
CHATTERBOX_REPO = "mlx-community/chatterbox-turbo-fp16"
CHOSEN_MODEL_NAME = "ResembleAI/chatterbox-turbo"
SAMPLE_RATE = 24000
REFERENCE_DIRECTORY = Path(__file__).resolve().parents[3] / "docs" / "_static" / "audio"
MALE_REFERENCE = REFERENCE_DIRECTORY / "male-voice.wav"
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."
# Note (Jiaxin Deng): one T3 pass speaks about 10 s (250 tokens) of any longer
# text, which takes seconds to generate.
PASSAGE = " ".join([SENTENCE] * 6)
# Note (Minyi Tang): generation samples noise, so the audio differs on every
# request; the transcript may miss one word without the speech being wrong.
MAX_WORD_ERRORS = 1
MAX_QUEUED_REQUESTS = 1
REQUEST_TIMEOUT_SECONDS = 20
SHORT_REQUEST_TIMEOUT_SECONDS = 1
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
# A cancelled generation stops within a few speech tokens.
CANCEL_SECONDS = 8.0


def chatterbox_directory() -> Path:
    return Path(DATA_ROOT) / "models" / CHATTERBOX_REPO.replace("/", "_")


def data_url(wav: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav).decode()


def tone_wav(seconds: float, rate: int = 16000, channels: int = 1) -> bytes:
    samples = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * rate)) / rate)
    samples = np.repeat(samples, channels)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


@pytest.fixture(scope="module")
def chatterbox_server() -> Iterator[Server]:
    running = Server(
        "chatterbox_server",
        "chatterbox",
        chatterbox_directory(),
        (
            "--max-queued-requests",
            str(MAX_QUEUED_REQUESTS),
            "--request-timeout-s",
            str(REQUEST_TIMEOUT_SECONDS),
            "--allowed-local-media-path",
            str(REFERENCE_DIRECTORY),
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


def send_speech_request(server: Server, body: dict) -> socket.socket:
    """Sends a speech request on a socket of its own, without reading the answer."""
    encoded_body = json.dumps(body)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(300)
    client.connect(("127.0.0.1", server.port))
    client.sendall(
        b"POST /v1/audio/speech HTTP/1.1\r\nHost: test\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(encoded_body)}\r\n\r\n{encoded_body}".encode()
    )
    return client


def request_states(server: Server) -> dict[str, int]:
    return json.loads(server.request("GET", "/health")[1])["request_states"]


def wait_for_states(
    server: Server, states: dict[str, int], timeout_seconds: float = CANCEL_SECONDS
) -> None:
    started_at = time.monotonic()
    while request_states(server) != states:
        assert (
            time.monotonic() - started_at < timeout_seconds
        ), f"request states stayed {request_states(server)}, not {states}"
        time.sleep(0.05)


def pcm_of(wav: bytes) -> bytes:
    with wave.open(io.BytesIO(wav)) as reader:
        assert (reader.getnchannels(), reader.getsampwidth()) == (1, 2)
        assert reader.getframerate() == SAMPLE_RATE
        return reader.readframes(reader.getnframes())


def test_chatterbox_speaks_its_input_in_the_builtin_voice(
    chatterbox_server: Server, asr_server: Server
) -> None:
    status, headers, audio = speak(
        chatterbox_server,
        {
            "input": SENTENCE,
            "model": chatterbox_server.ready["model_name"],
            "seed": 1,
        },
    )
    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Content-Disposition"] == 'attachment; filename="speech.wav"'
    assert headers["X-Request-Id"].startswith("speech-")
    assert headers["X-Finish-Reason"] == "stop"
    assert int(headers["X-SGLang-Omni-Total-Tokens"]) == int(
        headers["X-Prompt-Tokens"]
    ) + int(headers["X-Completion-Tokens"])
    heard = transcript_words(asr_server, pcm_of(audio))
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


@pytest.mark.parametrize(
    "reference_fields",
    [
        lambda: {"references": [{"audio_path": data_url(MALE_REFERENCE.read_bytes())}]},
        lambda: {"ref_audio": data_url(MALE_REFERENCE.read_bytes()), "ref_text": "hi"},
        lambda: {
            "references": [
                {"data": base64.b64encode(MALE_REFERENCE.read_bytes()).decode()}
            ]
        },
        lambda: {"references": [{"audio_path": str(MALE_REFERENCE)}]},
        lambda: {"ref_audio": MALE_REFERENCE.as_uri()},
    ],
    ids=["data_url", "ref_audio", "base64_data", "allowed_path", "file_url"],
)
def test_chatterbox_speaks_in_a_cloned_voice(
    chatterbox_server: Server, asr_server: Server, reference_fields
) -> None:
    status, headers, audio = speak(
        chatterbox_server,
        {"input": SENTENCE, "seed": 2, **reference_fields()},
    )
    assert (status, headers["X-Finish-Reason"]) == (200, "stop")
    heard = transcript_words(asr_server, pcm_of(audio))
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_chatterbox_same_seed_gives_the_same_audio(chatterbox_server: Server) -> None:
    body = {
        "input": "Seeded speech repeats.",
        "response_format": "pcm",
        "seed": 3,
        "ref_audio": data_url(MALE_REFERENCE.read_bytes()),
    }
    first = speak(chatterbox_server, body)
    second = speak(chatterbox_server, body)
    assert first[0] == second[0] == 200
    assert first[2] == second[2]
    other = speak(chatterbox_server, {**body, "seed": 4})
    assert other[2] != first[2]


def test_chatterbox_stops_at_max_new_tokens(chatterbox_server: Server) -> None:
    status, headers, pcm = speak(
        chatterbox_server,
        {"input": SENTENCE, "response_format": "pcm", "max_new_tokens": 10, "seed": 5},
    )
    assert status == 200
    assert headers["Content-Type"] == "audio/pcm"
    assert (headers["X-Finish-Reason"], headers["X-Completion-Tokens"]) == (
        "length",
        "10",
    )
    # At most ten tokens and three trailing silence tokens, 40 ms each.
    assert 0 < len(pcm) // 2 <= 13 * SAMPLE_RATE // 25


def test_chatterbox_disconnect_stops_its_generation(chatterbox_server: Server) -> None:
    with send_speech_request(chatterbox_server, {"input": PASSAGE}):
        wait_for_states(chatterbox_server, {"running": 1})
    wait_for_states(chatterbox_server, {})


def test_chatterbox_refuses_requests_beyond_its_queue(
    chatterbox_server: Server,
) -> None:
    with (
        send_speech_request(chatterbox_server, {"input": PASSAGE}),
        send_speech_request(chatterbox_server, {"input": PASSAGE}),
    ):
        wait_for_states(
            chatterbox_server, {"running": 1, "queued": MAX_QUEUED_REQUESTS}
        )
        status, _, body = speak(chatterbox_server, {"input": "hi"})
        assert (status, json.loads(body)) == (
            503,
            {"detail": "The request queue is full."},
        )
    wait_for_states(chatterbox_server, {}, 2 * CANCEL_SECONDS)


def test_chatterbox_times_out_a_request_past_its_deadline() -> None:
    impatient_server = Server(
        "chatterbox_server",
        "chatterbox",
        chatterbox_directory(),
        ("--request-timeout-s", str(SHORT_REQUEST_TIMEOUT_SECONDS)),
    )
    try:
        started_at = time.monotonic()
        status, _, body = speak(impatient_server, {"input": PASSAGE})
        assert (status, json.loads(body)) == (
            504,
            {"detail": "Speech generation timed out."},
        )
        assert time.monotonic() - started_at >= SHORT_REQUEST_TIMEOUT_SECONDS
        wait_for_states(impatient_server, {})
    finally:
        impatient_server.stop()


def test_chatterbox_ignores_unsupported_fields_left_empty(
    chatterbox_server: Server,
) -> None:
    status, _, _ = speak(
        chatterbox_server,
        {
            "input": "hi",
            "response_format": "pcm",
            "max_new_tokens": 1,
            "language": "",
            "references": [],
            "stage_params": {},
            "voice": "default",
        },
    )
    assert status == 200


def test_chatterbox_does_not_normalize_unicode_before_tokenizing(
    chatterbox_server: Server,
) -> None:
    """The GPT-2 tokenizer composed a decomposed accent, giving other text ids."""
    status, headers, _ = speak(
        chatterbox_server,
        {"input": "Cafe\u0301", "response_format": "pcm", "max_new_tokens": 1},
    )
    # The checkpoint's tokenizer: "Cafe\u0301." is 5 ids, "Caf\u00e9." is 4.
    assert (status, headers["X-Prompt-Tokens"]) == (200, "5")


def test_chatterbox_accepts_unsigned_64_bit_seed_and_top_k(
    chatterbox_server: Server,
) -> None:
    status, _, _ = speak(
        chatterbox_server,
        {
            "input": "hi",
            "response_format": "pcm",
            "max_new_tokens": 1,
            "seed": 2**64 - 1,
            "top_k": 2**64 - 1,
        },
    )
    assert status == 200


def test_chatterbox_rejects_another_model_than_its_chosen_name() -> None:
    named_server = Server(
        "chatterbox_server",
        "chatterbox",
        chatterbox_directory(),
        ("--model-name", CHOSEN_MODEL_NAME),
    )
    try:
        status, _, body = speak(named_server, {"input": "hi", "model": "tts-1"})
        assert (status, json.loads(body)) == (
            404,
            {"detail": "Unknown model: tts-1"},
        )
        status, _, body = speak(
            named_server, {"input": "hi", "ref_audio": str(MALE_REFERENCE)}
        )
        assert status == 400
        assert json.loads(body)["detail"].startswith(
            "local reference audio paths need --allowed-local-media-path"
        )
    finally:
        named_server.stop()


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ([1], "speech request body must be a JSON object"),
        ({"input": "  "}, "input must be a non-empty string"),
        ({"input": "x" * 4097}, "input must be at most 4096 characters"),
        ({"input": "hi", "model": 5}, "model must be a string"),
        ({"input": "hi", "voice": "nobody"}, "Chatterbox-Turbo has no voice presets"),
        ({"input": "hi", "voice": None}, "voice must be a string"),
        ({"input": "hi", "response_format": "mp3"}, "response_format must be one of"),
        ({"input": "hi", "stream": True}, "stream=true is not supported"),
        ({"input": "hi", "stream_format": "sse"}, "stream_format must be audio"),
        ({"input": "hi", "speed": 2.0}, "speed must be 1.0 on this model"),
        ({"input": "hi", "max_new_tokens": 0}, "max_new_tokens must be greater than 0"),
        (
            {"input": "hi", "max_new_tokens": 1025},
            "max_new_tokens must be at most 1024",
        ),
        ({"input": "hi", "max_new_tokens": 1.5}, "max_new_tokens must be an integer"),
        ({"input": "hi", "temperature": 0}, "temperature must be in (0"),
        ({"input": "hi", "temperature": "hot"}, "temperature must be a number"),
        ({"input": "hi", "top_p": 1.5}, "top_p must be in (0"),
        ({"input": "hi", "top_k": 0}, "top_k must be -1 (off) or at least 1"),
        ({"input": "hi", "repetition_penalty": 3}, "repetition_penalty must be in (0"),
        ({"input": "hi", "seed": -1}, "seed must be a non-negative integer"),
        ({"input": "hi", "references": {}}, "references must be a list"),
        ({"input": "hi", "references": [1]}, "each reference must be an object"),
        (
            {"input": "hi", "references": [{"text": "x"}, {"text": "y"}]},
            "Chatterbox-Turbo clones one voice",
        ),
        ({"input": "hi", "references": [{"text": "x"}]}, "references[0] has no audio"),
        (
            {"input": "hi", "references": [{"vq_codes": [1, 2]}]},
            "Chatterbox-Turbo does not take vq_codes",
        ),
        (
            {"input": "hi", "references": [{"data": "x"}], "ref_audio": "y"},
            "give references or ref_audio, not both",
        ),
        ({"input": "hi", "ref_text": "only text"}, "ref_text needs ref_audio"),
        ({"input": "hi", "ref_audio": 5}, "ref_audio must be a string"),
        (
            {"input": "hi", "ref_audio": "https://example.com/a.wav"},
            "reference audio URLs are not supported",
        ),
        (
            {"input": "hi", "ref_audio": "data:audio/wav,abc"},
            "a reference audio data URL",
        ),
        ({"input": "hi", "ref_audio": "data:,"}, "a reference audio data URL"),
        (
            {"input": "hi", "ref_audio": "data:audio/wav;base64,@@@"},
            "reference audio is not valid base64",
        ),
        (
            {"input": "hi", "ref_audio": data_url(b"RIFF\x00\x00\x00\x00WAVE")},
            "",
        ),
        (
            {"input": "hi", "ref_audio": data_url(tone_wav(4.0))},
            "reference audio must be longer than 5 seconds",
        ),
        (
            {"input": "hi", "ref_audio": data_url(tone_wav(61.0, rate=1000))},
            "reference audio must be at most 60 seconds",
        ),
        (
            {"input": "hi", "ref_audio": data_url(tone_wav(6.0, channels=3))},
            "WAV file has too many channels",
        ),
        (
            {"input": "hi", "ref_audio": data_url(tone_wav(6.0, rate=65519))},
            "WAV sample rate is out of range",
        ),
        (
            {"input": "hi", "ref_audio": "/etc/hosts"},
            "reference audio path is outside --allowed-local-media-path",
        ),
        (
            {"input": "hi", "ref_audio": str(REFERENCE_DIRECTORY / ".." / "x.wav")},
            "reference audio path is outside --allowed-local-media-path",
        ),
        (
            {"input": "hi", "ref_audio": str(REFERENCE_DIRECTORY / "missing.wav")},
            "reference audio file not found",
        ),
        (
            {
                "input": "hi",
                "references": [{"data": "eA==", "media_type": "audio/mp3"}],
            },
            "reference audio must be a WAV file",
        ),
        ({"input": "hi", "stage_params": []}, "stage_params must be an object"),
        (
            {"input": "hi", "language": "en", "instructions": "calm"},
            "Chatterbox-Turbo does not support these /v1/audio/speech fields: "
            "instructions, language",
        ),
    ],
)
def test_chatterbox_rejects_invalid_requests(
    chatterbox_server: Server, body: object, detail: str
) -> None:
    status, response = chatterbox_server.request(
        "POST", "/v1/audio/speech", json.dumps(body).encode()
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(detail)


def test_chatterbox_rejects_garbage_reference_audio(chatterbox_server: Server) -> None:
    """Any bytes as reference audio get 400, never a failure of the server."""
    generator = np.random.default_rng(0)
    for size in (0, 1, 12, 44, 45, 1000):
        garbage = generator.integers(0, 256, size, dtype=np.uint8).tobytes()
        for wav in (garbage, b"RIFF" + garbage[4:] + b"WAVEfmt " + garbage):
            status, response = chatterbox_server.request(
                "POST",
                "/v1/audio/speech",
                json.dumps(
                    {
                        "input": "hi",
                        "ref_audio": data_url(wav) if wav else "data:;base64,",
                    }
                ).encode(),
            )
            assert status == 400, response
    assert request_states(chatterbox_server) == {}


def test_chatterbox_refuses_a_declared_body_too_large_without_reading_it(
    chatterbox_server: Server,
) -> None:
    connection = http.client.HTTPConnection(
        "127.0.0.1", chatterbox_server.port, timeout=10
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


def test_chatterbox_server_serves_no_transcriptions(chatterbox_server: Server) -> None:
    status, _ = chatterbox_server.request("POST", "/v1/audio/transcriptions")
    assert status == 404
