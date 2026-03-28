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


def _read_wav_chunk_metadata(
    audio_bytes: bytes,
) -> tuple[int, int, int, bytes, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)
    return channels, sample_width, sample_rate, frames, frame_count


def _write_wav_bytes(
    *,
    channels: int,
    sample_width: int,
    sample_rate: int,
    frames: list[bytes],
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))
    return buffer.getvalue()


def decode_wav_chunk(audio_bytes: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())

    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def wav_duration_seconds(audio_bytes: bytes) -> float:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    if sample_rate <= 0:
        return 0.0
    return frame_count / sample_rate


class BufferedWavChunkEmitter:
    """Buffer short streamed WAV chunks into larger live-playback chunks."""

    def __init__(
        self,
        *,
        min_emit_duration_s: float = 1.0,
        max_buffered_chunks: int = 3,
    ) -> None:
        if min_emit_duration_s <= 0:
            raise ValueError("min_emit_duration_s must be positive")
        if max_buffered_chunks <= 0:
            raise ValueError("max_buffered_chunks must be positive")

        self._min_emit_duration_s = min_emit_duration_s
        self._max_buffered_chunks = max_buffered_chunks
        self._channels: int | None = None
        self._sample_width: int | None = None
        self._sample_rate: int | None = None
        self._frames: list[bytes] = []
        self._frame_count = 0
        self._chunk_count = 0

    def _validate_format(
        self,
        *,
        channels: int,
        sample_width: int,
        sample_rate: int,
    ) -> None:
        if self._channels is None:
            self._channels = channels
            self._sample_width = sample_width
            self._sample_rate = sample_rate
            return

        if (
            channels != self._channels
            or sample_width != self._sample_width
            or sample_rate != self._sample_rate
        ):
            raise ValueError("Inconsistent WAV chunk format in speech stream")

    def _should_emit(self) -> bool:
        if self._sample_rate is None or self._frame_count == 0:
            return False
        buffered_duration_s = self._frame_count / self._sample_rate
        return (
            buffered_duration_s >= self._min_emit_duration_s
            or self._chunk_count >= self._max_buffered_chunks
        )

    def _emit(self) -> bytes:
        audio_bytes = _write_wav_bytes(
            channels=self._channels or 1,
            sample_width=self._sample_width or 2,
            sample_rate=self._sample_rate or 24000,
            frames=self._frames,
        )
        self._frames = []
        self._frame_count = 0
        self._chunk_count = 0
        return audio_bytes

    def add_wav_chunk(self, audio_bytes: bytes) -> bytes | None:
        channels, sample_width, sample_rate, frames, frame_count = (
            _read_wav_chunk_metadata(audio_bytes)
        )
        self._validate_format(
            channels=channels,
            sample_width=sample_width,
            sample_rate=sample_rate,
        )
        self._frames.append(frames)
        self._frame_count += frame_count
        self._chunk_count += 1

        if self._should_emit():
            return self._emit()
        return None

    def flush(self) -> bytes | None:
        if not self._frames:
            return None
        return self._emit()


class WavChunkAccumulator:
    """Collect streamed WAV chunks and write a final WAV artifact."""

    def __init__(self) -> None:
        self._channels: int | None = None
        self._sample_width: int | None = None
        self._sample_rate: int | None = None
        self._frames: list[bytes] = []

    def add_wav_chunk(self, audio_bytes: bytes) -> bytes:
        channels, sample_width, sample_rate, frames, _ = _read_wav_chunk_metadata(
            audio_bytes
        )

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

        return _write_wav_bytes(
            channels=self._channels or 1,
            sample_width=self._sample_width or 2,
            sample_rate=self._sample_rate,
            frames=self._frames,
        )
