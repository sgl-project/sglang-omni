# SPDX-License-Identifier: Apache-2.0
"""API tests for fishaudio_drama3_server, run against the real model; the speech it
returns is checked by transcribing it with qwen3_asr_server.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_fishaudio_drama3_api.py
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import struct
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
DRAMA3_REPO = "mlx-community/fish-audio-drama-3-4bit"
SAMPLE_RATE = 44100
# One frame is 2048 samples of 16-bit audio.
FRAME_BYTES = 2048 * 2
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."
CLONED_SENTENCE = "The train to the city leaves from platform three at eight."
# Minutes of speech, several text chunks: no Mac generates it within
# REQUEST_TIMEOUT_SECONDS.
PASSAGE = " ".join([SENTENCE] * 30)
# Note (Yitong Li): generation samples every frame, and the server takes no
# seed, as CUDA's does not; the transcript may miss a word without the speech
# being wrong.
MAX_WORD_ERRORS = 2
MAX_QUEUED_REQUESTS = 1
REQUEST_TIMEOUT_SECONDS = 20
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
# A cancelled generation stops within a frame, or a block of prefill rows.
CANCEL_SECONDS = 8.0
# The most frames a reference's codes may hold: 30 s of audio.
MAX_REFERENCE_FRAMES = 646
VOICE_WAV = (
    Path(__file__).resolve().parent / "fishaudio_drama3_voice.wav"
).read_bytes()
VOICE_TEXT = "We asked over twenty different people, and they all said it was his."
GOLDEN_PATH = (
    Path(__file__).resolve().parent / "model_golden" / "fish-audio-drama-3-4bit.json"
)


@pytest.fixture(scope="module")
def drama3_server() -> Iterator[Server]:
    running = Server(
        "fishaudio_drama3_server",
        "fishaudio_drama3",
        Path(DATA_ROOT) / "models" / DRAMA3_REPO.replace("/", "_"),
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


def pcm_of(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes)) as reader:
        assert (reader.getnchannels(), reader.getsampwidth()) == (1, 2)
        assert reader.getframerate() == SAMPLE_RATE
        return reader.readframes(reader.getnframes())


def wav_of(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


def test_drama3_wav_speaks_its_input(drama3_server: Server, asr_server: Server) -> None:
    status, headers, audio = speak(
        drama3_server,
        {"input": SENTENCE, "model": drama3_server.ready["model_name"]},
    )
    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Content-Disposition"] == 'attachment; filename="speech.wav"'
    assert headers["X-Request-Id"].startswith("speech-")
    assert headers["X-Finish-Reason"] == "stop"
    frame_count = int(headers["X-Completion-Tokens"])
    prompt_row_count = int(headers["X-Prompt-Tokens"])
    assert int(headers["X-SGLang-Omni-Total-Tokens"]) == prompt_row_count + frame_count
    pcm = pcm_of(audio)
    assert len(pcm) == frame_count * FRAME_BYTES
    heard = transcript_words(asr_server, pcm, SAMPLE_RATE)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_drama3_clones_a_reference_voice(
    drama3_server: Server, asr_server: Server
) -> None:
    """Both reference forms give the same prompt: the transcript and the frames
    of the reference audio, then the input."""
    encoded = base64.b64encode(VOICE_WAV).decode()
    listed_body = {
        "input": CLONED_SENTENCE,
        "references": [{"data": encoded, "text": VOICE_TEXT}],
    }
    url_body = {
        "input": CLONED_SENTENCE,
        "ref_audio": f"data:audio/wav;base64,{encoded}",
        "ref_text": VOICE_TEXT,
    }
    status, headers, audio = speak(drama3_server, listed_body)
    assert status == 200
    assert int(headers["X-Prompt-Tokens"]) > len(pcm_of(VOICE_WAV)) // FRAME_BYTES
    url_status, url_headers, _ = speak(drama3_server, url_body)
    assert (url_status, url_headers["X-Prompt-Tokens"]) == (
        200,
        headers["X-Prompt-Tokens"],
    )
    heard = transcript_words(asr_server, pcm_of(audio), SAMPLE_RATE)
    assert word_errors(words(CLONED_SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_drama3_streams_pcm_while_it_speaks(
    drama3_server: Server, asr_server: Server
) -> None:
    connection = http.client.HTTPConnection(
        "127.0.0.1", drama3_server.port, timeout=300
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
    # Audio comes every seven frames: the first well before the last.
    assert first_audio_at - started_at < (finished_at - started_at) / 2
    heard = transcript_words(asr_server, pcm, SAMPLE_RATE)
    assert word_errors(words(SENTENCE), heard) <= MAX_WORD_ERRORS, heard


def test_drama3_stops_at_max_new_tokens(drama3_server: Server) -> None:
    status, headers, pcm = speak(
        drama3_server,
        {
            "input": SENTENCE,
            "speaker": "Default",
            "response_format": "pcm",
            "max_new_tokens": 10,
        },
    )
    assert status == 200
    assert (headers["X-Finish-Reason"], headers["X-Completion-Tokens"]) == (
        "length",
        "10",
    )
    assert len(pcm) == 10 * FRAME_BYTES


def test_drama3_clones_reference_codes(
    drama3_server: Server,
) -> None:
    golden = json.loads(GOLDEN_PATH.read_text())["cases"]["clone_en"]["reference"]
    codes = [[int(code) for code in codebook.split()] for codebook in golden["codes"]]
    status, headers, pcm = speak(
        drama3_server,
        {
            "input": "<|speaker:0|> Hi there. <|speaker:1|> Hello.",
            "references": [{"text": golden["text"], "vq_codes": codes}],
            "response_format": "pcm",
            "max_new_tokens": 5,
        },
    )
    assert status == 200
    assert int(headers["X-Prompt-Tokens"]) > len(codes[0])
    assert len(pcm) == int(headers["X-Completion-Tokens"]) * FRAME_BYTES


def test_drama3_disconnect_stops_its_generation(drama3_server: Server) -> None:
    with open_speech_stream(drama3_server, {"input": PASSAGE}) as client:
        assert client.recv(65536)
        assert request_states(drama3_server) == {"running": 1}
    wait_until_idle(drama3_server, CANCEL_SECONDS)


def test_drama3_disconnect_stops_a_generation_it_has_not_answered(
    drama3_server: Server,
) -> None:
    with send_speech_request(drama3_server, {"input": PASSAGE}):
        wait_until_running(drama3_server, CANCEL_SECONDS)
    wait_until_idle(drama3_server, CANCEL_SECONDS)


def test_drama3_refuses_requests_beyond_its_queue(drama3_server: Server) -> None:
    with (
        open_speech_stream(drama3_server, {"input": PASSAGE}) as running,
        open_speech_stream(drama3_server, {"input": PASSAGE}),
    ):
        assert running.recv(65536)
        assert request_states(drama3_server) == {
            "running": 1,
            "queued": MAX_QUEUED_REQUESTS,
        }
        status, _, body = speak(drama3_server, {"input": "hi"})
        assert (status, json.loads(body)) == (
            503,
            {"detail": "The request queue is full."},
        )
    wait_until_idle(drama3_server, 2 * CANCEL_SECONDS)


def test_drama3_times_out_a_request_past_its_deadline(drama3_server: Server) -> None:
    started_at = time.monotonic()
    status, _, body = speak(drama3_server, {"input": PASSAGE, "response_format": "pcm"})
    assert (status, json.loads(body)) == (
        504,
        {"detail": "Speech generation timed out."},
    )
    assert time.monotonic() - started_at >= REQUEST_TIMEOUT_SECONDS
    wait_until_idle(drama3_server, CANCEL_SECONDS)


def test_drama3_rejects_an_invalid_checkpoint_before_ready(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    failed = Server("fishaudio_drama3_server", "fishaudio_drama3", tmp_path)
    assert failed.ready["event"] == "failed"
    assert "fish_drama3" in failed.ready["reason"]
    assert failed.process.wait(timeout=30) != 0


def reference_body(reference: dict) -> dict:
    return {"input": "hi", "references": [{"text": "a", **reference}]}


def float_wav(samples: list[float]) -> bytes:
    """A mono 44.1 kHz float32 WAV, which the wave module cannot write."""
    data = struct.pack(f"<{len(samples)}f", *samples)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def encoded(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode()


QUIET_WAV = encoded(wav_of(b"\0\0" * 1600, 16000))
EMPTY_WAV = encoded(wav_of(b"", SAMPLE_RATE))
LONG_WAV = encoded(wav_of(b"\0\0" * (30 * SAMPLE_RATE + 1), SAMPLE_RATE))
NAN_WAV = encoded(float_wav([0.0, float("nan"), 0.0]))


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ([1], "speech request body must be a JSON object"),
        ({"input": "  "}, "input must be a non-empty string"),
        ({"input": "x" * 4097}, "input must be at most 4096 characters"),
        ({"input": "<|im_start|> hi"}, "input: text must not contain <|im_start|>"),
        ({"input": "<|speaker:16|> hi"}, "input: speaker tags run from"),
        ({"input": "hi", "voice": "alloy"}, 'unknown voice "alloy"'),
        ({"input": "hi", "speaker": "alloy"}, 'unknown voice "alloy"'),
        (
            {"input": "hi", "voice": None, "speaker": "default"},
            "voice must be a string",
        ),
        (
            reference_body({"vq_codes": [[0]] * 10, "speaker_embedding": [0.0]}),
            "references.0.speaker_embedding is not supported",
        ),
        (
            {"input": "hi", "language": "en", "instructions": "calm"},
            "instructions, language not supported",
        ),
        ({"input": "hi", "task_type": "VoiceDesign"}, 'task_type "VoiceDesign"'),
        ({"input": "hi", "stream": True}, "stream=true requires response_format='pcm'"),
        ({"input": "hi", "speed": 2.0}, "speed must be 1.0 on this model"),
        ({"input": "hi", "temperature": -1}, "temperature must be"),
        ({"input": "hi", "top_p": 0}, "top_p must be a number in (0, 1]"),
        ({"input": "hi", "top_k": 0}, "top_k must be an integer of at least 1"),
        ({"input": "hi", "repetition_penalty": 1.1}, "repetition_penalty must be 1.0"),
        ({"input": "hi", "max_new_tokens": 4097}, "max_new_tokens must be an integer"),
        ({"input": "hi", "seed": 1}, "seed not supported"),
        (
            {
                "input": "hi",
                "references": [
                    {"text": "a", "vq_codes": [[0] * MAX_REFERENCE_FRAMES] * 10},
                    # One frame of rounding is allowed per further reference.
                    {"text": "b", "vq_codes": [[0, 0]] * 10},
                ],
            },
            "references must total at most 30 seconds",
        ),
        (
            {
                "input": "hi",
                "references": [{"text": "a" * 600, "vq_codes": [[0]] * 10}] * 2,
            },
            "reference transcripts must total at most 1024 characters",
        ),
        (
            {"input": "hi", "ref_audio": "data:,", "ref_text": "a" * 1025},
            "ref_text must be at most 1024 characters",
        ),
        ({"input": "hi", "references": {}}, "references must be a list"),
        (
            {"input": "hi", "references": [{}] * 5},
            "at most 4 references are supported",
        ),
        ({"input": "<|speaker:1|>   "}, "input has nothing to speak"),
        ({"input": "hi", "top_p": 1.5}, "top_p must be a number in (0, 1]"),
        ({"input": "hi", "references": [3]}, "references.0 must be an object"),
        (
            {"input": "hi", "references": [{}], "ref_audio": "data:,"},
            "give references or ref_audio, not both",
        ),
        ({"input": "hi", "ref_text": "a"}, "ref_text needs ref_audio"),
        (reference_body({}), "references.0 needs audio or vq_codes"),
        (
            reference_body({"text": "a" * 1025, "data": QUIET_WAV}),
            "references.0.text must be at most",
        ),
        (reference_body({"data": EMPTY_WAV}), "references.0 audio is empty"),
        (reference_body({"data": LONG_WAV}), "references.0 audio must be at most 30"),
        (reference_body({"data": NAN_WAV}), "references.0 audio must be finite"),
        (
            reference_body({"audio": "data:audio/wav," + QUIET_WAV}),
            "references.0 data URL must include base64 media data",
        ),
        (
            reference_body({"vq_codes": [[1.5]] * 10}),
            "references.0.vq_codes must be a [10, frames] grid",
        ),
        (
            reference_body({"vq_codes": [[0] * (MAX_REFERENCE_FRAMES + 1)] * 10}),
            f"references.0.vq_codes must have at most {MAX_REFERENCE_FRAMES} frames",
        ),
        (reference_body({"text": 3, "data": QUIET_WAV}), "references.0.text must be"),
        (reference_body({"data": "abc"}), "references.0 must be valid base64"),
        (reference_body({"data": QUIET_WAV}), "references.0: audio must be 44.1 kHz"),
        (
            reference_body({"audio_path": "/tmp/reference.wav"}),
            "references.0.audio_path must be a base64 WAV data URL",
        ),
        (
            reference_body({"vq_codes": [[1]]}),
            "references.0.vq_codes must be a [10, frames] grid",
        ),
        (
            reference_body({"vq_codes": [[4096]] * 10}),
            "references.0.vq_codes must be in [0, 4096)",
        ),
        (
            {"input": "hi", "ref_audio": "https://example.com/a.wav", "ref_text": "a"},
            "ref_audio must be a base64 WAV data URL",
        ),
        (
            {"input": "hi", "ref_audio": f"data:audio/wav;base64,{QUIET_WAV}"},
            "reference transcript is required",
        ),
    ],
)
def test_drama3_rejects_invalid_requests(
    drama3_server: Server, body: object, detail: str
) -> None:
    status, response = drama3_server.request(
        "POST", "/v1/audio/speech", json.dumps(body).encode()
    )
    assert status == 400
    assert json.loads(response)["detail"].startswith(detail)


def test_drama3_refuses_a_declared_body_too_large_without_reading_it(
    drama3_server: Server,
) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", drama3_server.port, timeout=10)
    connection.putrequest("POST", "/v1/audio/speech")
    connection.putheader("Content-Length", str(MAX_REQUEST_BODY_BYTES + 1))
    connection.endheaders()
    response = connection.getresponse()
    assert (response.status, json.loads(response.read())) == (
        413,
        {"detail": f"request body must be at most {MAX_REQUEST_BODY_BYTES} bytes"},
    )
    connection.close()
