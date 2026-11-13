# SPDX-License-Identifier: Apache-2.0
"""Real-model Parakeet API regressions; needs NATIVE_RUNTIME_BIN and CI_DATA_ROOT."""

from __future__ import annotations

import http.client
import io
import json
import os
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest

from sglang_omni_mlx.native.tests.native_server import (
    REQUEST_TIMEOUT_SECONDS,
    STATE_POLL_SECONDS,
    VOXT_SAMPLE_RATE,
    HttpResponse,
    SupervisedServer,
    multipart_form,
    silent_wav,
    sse_payloads,
)

RUNTIME_BIN = os.environ.get("NATIVE_RUNTIME_BIN")
DATA_ROOT = os.environ.get("CI_DATA_ROOT")
pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
PARAKEET_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
PCM16_BYTES_PER_SAMPLE = 2
# Note (Mengwei Liu): cancellation is observed within the current bounded model chunk.
PARAKEET_CANCEL_TIMEOUT_SECONDS = 2.0


@pytest.fixture(scope="module")
def parakeet_server() -> Iterator[SupervisedServer]:
    server = SupervisedServer(
        Path(RUNTIME_BIN) / "parakeet_server",
        "parakeet",
        Path(DATA_ROOT) / "models" / PARAKEET_REPO.replace("/", "_"),
    )
    try:
        yield server
    finally:
        server.stop()


def corpus_wav(clip_id: str) -> bytes:
    return (Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{clip_id}.wav").read_bytes()


def pcm_bytes_to_wav(pcm_bytes: bytes) -> bytes:
    """Wrap mono 16-bit PCM at VOXT_SAMPLE_RATE in a WAV header."""
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        output.setframerate(VOXT_SAMPLE_RATE)
        output.writeframes(pcm_bytes)
    return wav_bytes.getvalue()


def test_parakeet_final_request_returns_text_and_sentences(
    parakeet_server: SupervisedServer,
) -> None:
    model_name = parakeet_server.first_event["model_name"]
    assert isinstance(model_name, str)
    assert model_name.startswith("voxt-parakeet-")
    response: HttpResponse = parakeet_server.post_form(
        {"chunk_duration": "1200"}, corpus_wav("0006_en_short")
    )
    assert response.status == 200
    result = response.json()
    assert result["text"].strip()
    [segment] = result["segments"]
    assert segment["text"].strip() == result["text"]
    assert 0 <= segment["start"] < segment["end"] <= 3.1


@pytest.mark.parametrize(
    "fields", [{}, {"chunk_duration": "1200"}, {"chunk_duration": "30"}]
)
@pytest.mark.parametrize("stream", [False, True])
def test_parakeet_final_keeps_speech_after_five_seconds(
    parakeet_server: SupervisedServer, fields: dict[str, str], stream: bool
) -> None:
    """The five-second final windows silently dropped the spoken tail of this clip."""
    audio = Path(__file__).resolve().parents[3] / "Voxt/testdata/eval/en-3.wav"
    response: HttpResponse = parakeet_server.post_form(
        {**fields, "stream": str(stream).lower()}, audio.read_bytes()
    )
    assert response.status == 200
    result = json.loads(sse_payloads(response.body)[0]) if stream else response.json()
    words = result["text"].lower().rstrip(".").split()
    assert words[:3] == ["please", "remind", "me"]
    assert words[-5:] == ["at", "nine", "in", "the", "morning"]
    segments = result["segments"]
    assert all(0 <= segment["start"] < segment["end"] <= 7.15 for segment in segments)
    assert [segment["start"] for segment in segments] == sorted(
        segment["start"] for segment in segments
    )


@pytest.mark.parametrize("stream", [False, True])
def test_parakeet_final_keeps_both_ends_across_automatic_chunks(
    parakeet_server: SupervisedServer, stream: bool
) -> None:
    """A whole recording longer than 30 seconds uses the bounded final path."""
    response: HttpResponse = parakeet_server.post_form(
        {"stream": str(stream).lower()}, corpus_wav("0344_en_long")
    )
    assert response.status == 200
    result = json.loads(sse_payloads(response.body)[0]) if stream else response.json()
    assert result["text"].lower().split()[:10] == [
        "in",
        "every",
        "way",
        "they",
        "sought",
        "to",
        "undermine",
        "the",
        "authority",
        "of",
    ]
    assert "neither by men nor by man" in result["text"].lower()
    segments = result["segments"]
    assert len(segments) > 1
    assert all(0 <= segment["start"] < segment["end"] <= 64.4 for segment in segments)
    assert [segment["start"] for segment in segments] == sorted(
        segment["start"] for segment in segments
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("seconds", [30.25, 31.0, 32.0])
def test_parakeet_final_keeps_a_short_last_tail(
    parakeet_server: SupervisedServer, stream: bool, seconds: float
) -> None:
    """Near-cap recordings lost spoken words when their final crop decoded blank."""
    with wave.open(io.BytesIO(corpus_wav("0347_en_long"))) as source:
        pcm_bytes = source.readframes(round(seconds * VOXT_SAMPLE_RATE))
    response: HttpResponse = parakeet_server.post_form(
        {"stream": str(stream).lower()}, pcm_bytes_to_wav(pcm_bytes)
    )
    assert response.status == 200
    result = json.loads(sse_payloads(response.body)[0]) if stream else response.json()
    assert "and from the adoration" in result["text"].lower()
    assert result["segments"][-1]["end"] > 29.5
    assert all(
        0 <= segment["start"] < segment["end"] <= seconds + 0.4
        for segment in result["segments"]
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "clip_ids, start_sample, sample_count, expected_phrases",
    [
        (
            ["0099_en_mid", "0106_en_mid", "0121_en_mid", "0053_en_short"],
            205745,
            452180,
            ["unusual severity", "before this travesty", "many years"],
        ),
        (
            ["0102_en_mid", "0067_en_mid", "0034_en_short", "0080_en_mid"],
            99318,
            416063,
            ["tumbled in a heap", "immensely popular", "without saying it positively"],
        ),
    ],
)
def test_parakeet_final_recovers_audible_blank_tails(
    parakeet_server: SupervisedServer,
    stream: bool,
    clip_ids: list[str],
    start_sample: int,
    sample_count: int,
    expected_phrases: list[str],
) -> None:
    """Audible windows returned no text or omitted their entire spoken tail."""
    # Note (Mengwei Liu): seed 2888's benchmark exposed these exact quiet-boundary failures.
    pcm_bytes = bytearray()
    for clip_id in clip_ids:
        with wave.open(io.BytesIO(corpus_wav(clip_id))) as source:
            pcm_bytes.extend(source.readframes(source.getnframes()))
    response: HttpResponse = parakeet_server.post_form(
        {"stream": str(stream).lower()},
        pcm_bytes_to_wav(
            bytes(pcm_bytes[2 * start_sample : 2 * (start_sample + sample_count)])
        ),
    )
    assert response.status == 200
    result = json.loads(sse_payloads(response.body)[0]) if stream else response.json()
    text = result["text"].lower()
    assert all(phrase in text for phrase in expected_phrases)
    assert result["segments"][-1]["end"] > 25
    assert all(
        0 <= segment["start"] < segment["end"] <= sample_count / VOXT_SAMPLE_RATE
        for segment in result["segments"]
    )
    segment_start_seconds = [segment["start"] for segment in result["segments"]]
    assert segment_start_seconds == sorted(segment_start_seconds)


@pytest.mark.parametrize("seconds", [5, 31, 65])
def test_parakeet_final_keeps_silence_empty(
    parakeet_server: SupervisedServer, seconds: int
) -> None:
    """Silence must remain empty through automatic splitting and recovery."""
    response: HttpResponse = parakeet_server.post_form(
        {}, silent_wav(seconds * VOXT_SAMPLE_RATE)
    )
    assert response.status == 200
    result = response.json()
    assert result["text"] == ""
    assert "segments" not in result
    response = parakeet_server.post_form({}, corpus_wav("0006_en_short"))
    assert response.status == 200
    assert response.json()["text"].strip()


def test_parakeet_disconnected_final_stops_and_recovers(
    parakeet_server: SupervisedServer,
) -> None:
    """A disconnected final request must release its worker for the next request."""
    with wave.open(io.BytesIO(corpus_wav("0344_en_long"))) as source:
        pcm_bytes = source.readframes(source.getnframes())
    audio_sample_count = 600 * VOXT_SAMPLE_RATE
    pcm_bytes = (pcm_bytes * (audio_sample_count * 2 // len(pcm_bytes) + 1))[
        : audio_sample_count * 2
    ]
    boundary, body = multipart_form({"stream": "true"}, pcm_bytes_to_wav(pcm_bytes))
    connection = http.client.HTTPConnection(
        "127.0.0.1", parakeet_server.port, timeout=REQUEST_TIMEOUT_SECONDS
    )
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    assert response.status == 200
    assert parakeet_server.request_states() == {"running": 1}
    closed_at = time.monotonic()
    response.close()
    connection.close()
    while parakeet_server.request_states() != {}:
        assert time.monotonic() - closed_at < PARAKEET_CANCEL_TIMEOUT_SECONDS
        time.sleep(STATE_POLL_SECONDS)
    recovered_response: HttpResponse = parakeet_server.post_form(
        {}, corpus_wav("0006_en_short")
    )
    assert recovered_response.status == 200
    assert recovered_response.json()["text"].strip()


def test_parakeet_merges_overlapping_chunks_into_sentences(
    parakeet_server: SupervisedServer,
) -> None:
    # Note (Dayuxiaoshui): explicit windows overlap by one second on this 64-second clip.
    response: HttpResponse = parakeet_server.post_form(
        {
            "chunk_duration": "20",
            "stream": "true",
            "include_generation_metadata": "true",
        },
        corpus_wav("0344_en_long"),
    )
    assert response.status == 200
    done = json.loads(sse_payloads(response.body)[0])
    assert done["text"].strip()
    assert done["generation_metadata"]["finish_reason"] == "stop"
    response = parakeet_server.post_form(
        {"chunk_duration": "20"}, corpus_wav("0344_en_long")
    )
    assert response.status == 200
    segments = response.json()["segments"]
    assert len(segments) > 1
    assert [segment["start"] for segment in segments] == sorted(
        segment["start"] for segment in segments
    )


@pytest.mark.parametrize("chunk_duration", ["nan", "1", "-5", "600"])
def test_parakeet_rejects_a_bad_chunk_duration(
    parakeet_server: SupervisedServer, chunk_duration: str
) -> None:
    response: HttpResponse = parakeet_server.post_form(
        {"chunk_duration": chunk_duration}, corpus_wav("0006_en_short")
    )
    assert response.status == 400
    assert "chunk_duration" in response.json()["detail"]
