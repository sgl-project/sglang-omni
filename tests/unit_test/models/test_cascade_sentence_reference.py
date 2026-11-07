# SPDX-License-Identifier: Apache-2.0
"""What the cascade conditions each sentence on: a fixed clip, or the sentence just spoken."""

from __future__ import annotations

import wave
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

import numpy as np
import pytest

from sglang_omni.models.streaming_cascade.synthesizers import (
    OpenAISpeechSynthesizer,
    OpenAISpeechSynthesizerConfig,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import TEXT, TEXT_DELTA

SAMPLE_RATE_HZ = 24000
# One request's audio: 3 samples at 24 kHz, the same shape the cascade fixtures serve.
SPEECH_SAMPLES = np.array([0, 16384, -16384], dtype="<i2")
STATIC_REFERENCE = "/tmp/ref.wav"


def text_chunk(text: str, turn: int) -> TimedChunk:
    return TimedChunk(TEXT, 0, 0, 0, {"text": text}, format=TEXT_DELTA, turn=turn)


@dataclass
class StubSpeechStream:
    pcm16: bytes
    headers: dict[str, str] = field(
        default_factory=lambda: {"X-Sample-Rate": str(SAMPLE_RATE_HZ)}
    )

    async def __aenter__(self) -> "StubSpeechStream":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        yield self.pcm16


class StubSpeechClient:
    """Records every request body and answers with the same short utterance."""

    def __init__(self, pcm16: bytes = SPEECH_SAMPLES.tobytes()) -> None:
        self.requests: list[dict[str, object]] = []
        self.pcm16 = pcm16

    def stream(
        self, method: str, url: str, *, json: dict[str, object]
    ) -> StubSpeechStream:
        """Answers any URL: the tests assert on the request body, not the route."""
        assert method == "POST" and url.endswith("/audio/speech")
        self.requests.append(json)
        return StubSpeechStream(self.pcm16)

    async def aclose(self) -> None:
        return None


def synthesizer(**overrides: object) -> OpenAISpeechSynthesizer:
    config = OpenAISpeechSynthesizerConfig(
        base_url="http://127.0.0.1:1", model="tts", **overrides
    )
    result = OpenAISpeechSynthesizer(config)
    result.client = StubSpeechClient()  # type: ignore[assignment]
    return result


async def deltas(chunks: Iterable[TimedChunk]) -> AsyncIterator[TimedChunk]:
    for chunk in chunks:
        yield chunk


async def speak(
    subject: OpenAISpeechSynthesizer, text: str
) -> tuple[list[TimedChunk], StubSpeechClient]:
    emitted: list[TimedChunk] = []
    await subject.speak_reply(deltas([text_chunk(text, 1)]), 1, emitted.append)
    client = subject.client
    assert isinstance(client, StubSpeechClient)
    return emitted, client


def reference_of(request: dict[str, object]) -> dict[str, object]:
    references = request.get("references")
    assert isinstance(references, list) and len(references) == 1
    entry = references[0]
    assert isinstance(entry, dict)
    return entry


def wav_samples(path: str) -> np.ndarray:
    with wave.open(path, "rb") as reference:
        assert reference.getnchannels() == 1
        assert reference.getsampwidth() == 2
        assert reference.getframerate() == SAMPLE_RATE_HZ
        return np.frombuffer(reference.readframes(reference.getnframes()), dtype="<i2")


@pytest.mark.asyncio
async def test_a_static_reference_conditions_every_sentence() -> None:
    subject = synthesizer(reference_audio_path=STATIC_REFERENCE, reference_text="ref")
    _, client = await speak(subject, "One two. Three four.")
    assert len(client.requests) == 2
    assert [reference_of(request) for request in client.requests] == [
        {"audio_path": STATIC_REFERENCE, "text": "ref"},
        {"audio_path": STATIC_REFERENCE, "text": "ref"},
    ]
    await subject.close()


@pytest.mark.asyncio
async def test_the_previous_sentence_is_the_next_sentence_reference() -> None:
    subject = synthesizer(reference_mode="previous_segment")
    _, client = await speak(subject, "One two. Three four.")
    assert len(client.requests) == 2
    # Nothing has been spoken yet, so the first sentence is conditioned on nothing.
    assert "references" not in client.requests[0]
    entry = reference_of(client.requests[1])
    assert entry["text"] == "One two."
    # The carried clip is this run's audio, not a configured file.
    assert wav_samples(str(entry["audio_path"])).tolist() == SPEECH_SAMPLES.tolist()
    await subject.close()


@pytest.mark.asyncio
async def test_the_carried_reference_stops_at_the_service_limit() -> None:
    subject = synthesizer(reference_mode="previous_segment", max_reference_seconds=0.5)
    long_pcm = np.zeros(SAMPLE_RATE_HZ * 2, dtype="<i2").tobytes()
    subject.client = StubSpeechClient(long_pcm)  # type: ignore[assignment]
    _, client = await speak(subject, "One two. Three four.")
    entry = reference_of(client.requests[1])
    carried = wav_samples(str(entry["audio_path"]))
    assert len(carried) == SAMPLE_RATE_HZ // 2
    await subject.close()


@pytest.mark.asyncio
async def test_the_first_sentence_falls_back_to_the_configured_clip() -> None:
    subject = synthesizer(
        reference_mode="previous_segment",
        reference_audio_path=STATIC_REFERENCE,
        reference_text="ref",
    )
    _, client = await speak(subject, "One two. Three four.")
    assert reference_of(client.requests[0]) == {
        "audio_path": STATIC_REFERENCE,
        "text": "ref",
    }
    assert reference_of(client.requests[1])["text"] == "One two."
    await subject.close()


@pytest.mark.asyncio
async def test_static_mode_carries_nothing_between_sentences() -> None:
    subject = synthesizer()
    _, client = await speak(subject, "One two. Three four.")
    assert [("references" in request) for request in client.requests] == [False, False]
    await subject.close()
