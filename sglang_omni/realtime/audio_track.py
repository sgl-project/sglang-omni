# SPDX-License-Identifier: Apache-2.0
"""Outgoing audio track for the WebRTC prototype."""

from __future__ import annotations

import asyncio
import time
from fractions import Fraction

import av
import numpy as np
from aiortc import AudioStreamTrack

from sglang_omni.realtime.media import mono_float32, resample_linear


class BufferedAudioStreamTrack(AudioStreamTrack):
    """Server-driven mono audio track backed by a PCM buffer."""

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        frame_duration_s: float = 0.02,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_samples = int(round(sample_rate * frame_duration_s))
        self._buffer = np.zeros(0, dtype=np.int16)
        self._lock = asyncio.Lock()
        self._pts = 0
        self._start_time: float | None = None

    @property
    def pending_samples(self) -> int:
        return int(self._buffer.shape[0])

    async def clear(self) -> None:
        async with self._lock:
            self._buffer = np.zeros(0, dtype=np.int16)

    async def enqueue(self, audio: np.ndarray, sample_rate: int) -> None:
        pcm = mono_float32(audio)
        pcm = resample_linear(pcm, sample_rate, self.sample_rate)
        pcm_i16 = np.clip(pcm * 32767.0, -32768.0, 32767.0).astype(np.int16)
        async with self._lock:
            self._buffer = np.concatenate([self._buffer, pcm_i16])

    async def recv(self) -> av.AudioFrame:
        if self._start_time is None:
            self._start_time = time.monotonic()
        else:
            target_time = self._start_time + (self._pts / float(self.sample_rate))
            delay = target_time - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        async with self._lock:
            if self._buffer.shape[0] >= self.frame_samples:
                pcm_i16 = self._buffer[: self.frame_samples]
                self._buffer = self._buffer[self.frame_samples :]
            elif self._buffer.shape[0] > 0:
                pcm_i16 = np.zeros(self.frame_samples, dtype=np.int16)
                pcm_i16[: self._buffer.shape[0]] = self._buffer
                self._buffer = np.zeros(0, dtype=np.int16)
            else:
                pcm_i16 = np.zeros(self.frame_samples, dtype=np.int16)

        frame = av.AudioFrame.from_ndarray(
            pcm_i16.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, self.sample_rate)
        self._pts += self.frame_samples
        return frame
