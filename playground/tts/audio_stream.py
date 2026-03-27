# SPDX-License-Identifier: Apache-2.0
"""Helpers for parsing and assembling streamed speech chunks."""

from __future__ import annotations

import base64
import io
import json
import wave
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeechStreamEvent:
    index: int
    audio_bytes: bytes | None = None
    sample_rate: int | None = None
    audio_format: str | None = None
    mime_type: str | None = None
    finish_reason: str | None = None
    is_done: bool = False


def parse_speech_stream_data(data: str) -> SpeechStreamEvent | None:
    data = data.strip()
    if not data:
        return None
    if data == "[DONE]":
        return SpeechStreamEvent(index=-1, is_done=True)

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid speech stream payload: {exc}") from exc

    audio = payload.get("audio")
    if not isinstance(audio, dict):
        return SpeechStreamEvent(
            index=int(payload.get("index", -1)),
            finish_reason=payload.get("finish_reason"),
        )

    raw_audio = audio.get("data")
    if not isinstance(raw_audio, str):
        return SpeechStreamEvent(
            index=int(payload.get("index", -1)),
            finish_reason=payload.get("finish_reason"),
        )

    try:
        audio_bytes = base64.b64decode(raw_audio)
    except Exception as exc:
        raise ValueError(f"Invalid base64 speech stream chunk: {exc}") from exc

    sample_rate = audio.get("sample_rate")
    return SpeechStreamEvent(
        index=int(payload.get("index", -1)),
        audio_bytes=audio_bytes,
        sample_rate=int(sample_rate) if sample_rate is not None else None,
        audio_format=audio.get("format"),
        mime_type=audio.get("mime_type"),
        finish_reason=payload.get("finish_reason"),
    )


def decode_wav_chunk(audio_bytes: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())

    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


class WavChunkAccumulator:
    """Collect streamed WAV chunks and write a final WAV artifact."""

    def __init__(self) -> None:
        self._channels: int | None = None
        self._sample_width: int | None = None
        self._sample_rate: int | None = None
        self._frames: list[bytes] = []

    def add_wav_chunk(self, audio_bytes: bytes) -> bytes:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if self._channels is None:
            self._channels = channels
            self._sample_width = sample_width
            self._sample_rate = sample_rate
        elif (
            channels != self._channels
            or sample_width != self._sample_width
            or sample_rate != self._sample_rate
        ):
            raise ValueError("Inconsistent WAV chunk format in speech stream")

        self._frames.append(frames)
        return audio_bytes

    def to_wav_bytes(self) -> bytes | None:
        if self._sample_rate is None or not self._frames:
            return None

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self._channels or 1)
            wav_file.setsampwidth(self._sample_width or 2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(b"".join(self._frames))
        return buffer.getvalue()
