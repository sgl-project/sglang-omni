# SPDX-License-Identifier: Apache-2.0
"""Helpers for assembling streamed speech chunks."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

DEFAULT_S2PRO_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class SpeechStreamEvent:
    audio_bytes: bytes
    sample_rate: int | None = None
    channels: int = 1
    sample_width: int = 2


@dataclass(frozen=True)
class PcmStreamFormat:
    sample_rate: int = DEFAULT_S2PRO_SAMPLE_RATE
    channels: int = 1
    sample_width: int = 2

    @property
    def block_align(self) -> int:
        return self.channels * self.sample_width


class PcmChunkAssembler:
    """Emit only complete PCM frame blocks from arbitrary HTTP byte chunks."""

    def __init__(self, pcm_format: PcmStreamFormat) -> None:
        if pcm_format.sample_rate <= 0:
            raise ValueError("PCM sample rate must be positive")
        if pcm_format.channels <= 0:
            raise ValueError("PCM channels must be positive")
        if pcm_format.sample_width <= 0:
            raise ValueError("PCM sample width must be positive")

        self._format = pcm_format
        self._pending = bytearray()

    def add_chunk(self, audio_bytes: bytes) -> bytes | None:
        if not audio_bytes:
            return None

        self._pending.extend(audio_bytes)
        aligned_length = len(self._pending) - (
            len(self._pending) % self._format.block_align
        )
        if aligned_length == 0:
            return None

        aligned = bytes(self._pending[:aligned_length])
        del self._pending[:aligned_length]
        return aligned

    def flush(self) -> None:
        if self._pending:
            raise ValueError("PCM stream ended with a partial audio frame")


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


def wav_duration_seconds(audio_bytes: bytes) -> float:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    if sample_rate <= 0:
        return 0.0
    return frame_count / sample_rate


class BufferedWavChunkEmitter:
    """Buffer short streamed PCM chunks into larger WAV live-playback chunks."""

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
            raise ValueError("Inconsistent PCM chunk format in speech stream")

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
            sample_rate=self._sample_rate or DEFAULT_S2PRO_SAMPLE_RATE,
            frames=self._frames,
        )
        self._frames = []
        self._frame_count = 0
        self._chunk_count = 0
        return audio_bytes

    def add_pcm_chunk(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
    ) -> bytes | None:
        self._validate_format(
            channels=channels,
            sample_width=sample_width,
            sample_rate=sample_rate,
        )
        self._frames.append(audio_bytes)
        self._frame_count += len(audio_bytes) // (channels * sample_width)
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

    def add_pcm_chunk(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
    ) -> bytes:
        if self._channels is None:
            self._channels = channels
            self._sample_width = sample_width
            self._sample_rate = sample_rate
        elif (
            channels != self._channels
            or sample_width != self._sample_width
            or sample_rate != self._sample_rate
        ):
            raise ValueError("Inconsistent PCM chunk format in speech stream")

        self._frames.append(audio_bytes)
        return _write_wav_bytes(
            channels=channels,
            sample_width=sample_width,
            sample_rate=sample_rate,
            frames=[audio_bytes],
        )

    def to_wav_bytes(self) -> bytes | None:
        if self._sample_rate is None or not self._frames:
            return None

        return _write_wav_bytes(
            channels=self._channels or 1,
            sample_width=self._sample_width or 2,
            sample_rate=self._sample_rate,
            frames=self._frames,
        )
