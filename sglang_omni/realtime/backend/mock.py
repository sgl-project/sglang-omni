# SPDX-License-Identifier: Apache-2.0
"""Mock response backend for browser smoke tests."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import numpy as np

from sglang_omni.realtime.backend.base import (
    BackendCapabilities,
    ResponseBackend,
    ResponseEvent,
    TurnContext,
)


class MockResponseBackend(ResponseBackend):
    """Replay captured turn audio for end-to-end smoke tests."""

    def __init__(
        self,
        *,
        model: str = "mock-realtime",
        output_modalities: tuple[str, ...] = ("text", "audio"),
        response_text: str = "Mock backend replaying captured user audio.",
        sample_rate: int = 24000,
        chunk_duration_s: float = 0.24,
        inter_chunk_delay_s: float = 0.08,
        total_duration_s: float = 1.2,
        tone_hz: float = 660.0,
    ) -> None:
        self._model = model
        self._output_modalities = output_modalities
        self._response_text = response_text.strip() or "Mock backend response."
        self._sample_rate = sample_rate
        self._chunk_duration_s = chunk_duration_s
        self._inter_chunk_delay_s = inter_chunk_delay_s
        self._total_duration_s = total_duration_s
        self._tone_hz = tone_hz
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._capabilities = BackendCapabilities(
            accepts_audio_input=True,
            accepts_video_input=True,
            returns_text="text" in output_modalities,
            returns_audio="audio" in output_modalities,
            supports_cancel=True,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    async def stream_response(
        self,
        turn: TurnContext,
    ) -> AsyncIterator[ResponseEvent]:
        response_id = uuid.uuid4().hex
        cancel_event = asyncio.Event()
        self._cancel_events[response_id] = cancel_event
        try:
            yield ResponseEvent(type="response_started", response_id=response_id)

            if self._capabilities.returns_text:
                for text_delta in self._split_text(self._response_text):
                    if cancel_event.is_set():
                        yield ResponseEvent(
                            type="done",
                            response_id=response_id,
                            finish_reason="cancelled",
                        )
                        return
                    yield ResponseEvent(
                        type="text_delta",
                        response_id=response_id,
                        text=text_delta,
                    )

            if self._capabilities.returns_audio:
                audio_chunks, sample_rate = self._build_audio_chunks(turn)
                for chunk in audio_chunks:
                    if cancel_event.is_set():
                        yield ResponseEvent(
                            type="done",
                            response_id=response_id,
                            finish_reason="cancelled",
                        )
                        return
                    yield ResponseEvent(
                        type="audio_chunk",
                        response_id=response_id,
                        audio=chunk,
                        sample_rate=sample_rate,
                    )
                    if self._inter_chunk_delay_s > 0:
                        await asyncio.sleep(self._inter_chunk_delay_s)

            finish_reason = "cancelled" if cancel_event.is_set() else "stop"
            yield ResponseEvent(
                type="done",
                response_id=response_id,
                finish_reason=finish_reason,
            )
        finally:
            self._cancel_events.pop(response_id, None)

    async def cancel(self, response_id: str) -> None:
        event = self._cancel_events.get(response_id)
        if event is not None:
            event.set()

    def _split_text(self, text: str) -> list[str]:
        parts = [segment.strip() for segment in text.split(".") if segment.strip()]
        if not parts:
            return [text]
        return [f"{part}. " for part in parts[:-1]] + [f"{parts[-1]}."]

    def _build_audio_chunks(
        self,
        turn: TurnContext,
    ) -> tuple[list[np.ndarray], int]:
        waveform, sample_rate = self._resolve_response_audio(turn)
        total_samples = int(waveform.size)
        chunk_samples = max(1, int(round(sample_rate * self._chunk_duration_s)))
        return [
            waveform[start : start + chunk_samples]
            for start in range(0, total_samples, chunk_samples)
        ], sample_rate

    def _resolve_response_audio(self, turn: TurnContext) -> tuple[np.ndarray, int]:
        if turn.user_audio is not None:
            waveform = np.asarray(turn.user_audio, dtype=np.float32).reshape(-1)
            if waveform.size > 0:
                sample_rate = int(turn.user_audio_sample_rate or self._sample_rate)
                return waveform, sample_rate

        sample_rate = self._sample_rate
        total_samples = max(1, int(round(sample_rate * self._total_duration_s)))
        waveform = self._build_demo_waveform(total_samples)
        return waveform, sample_rate

    def _build_demo_waveform(self, num_samples: int) -> np.ndarray:
        t = np.arange(num_samples, dtype=np.float32) / float(self._sample_rate)
        carrier = np.sin(2.0 * np.pi * self._tone_hz * t)
        modulator = 0.55 + 0.45 * np.sin(2.0 * np.pi * 2.0 * t)
        waveform = 0.18 * carrier * modulator

        fade_samples = min(num_samples // 8, max(1, self._sample_rate // 50))
        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = fade_in[::-1]
            waveform[:fade_samples] *= fade_in
            waveform[-fade_samples:] *= fade_out
        return waveform.astype(np.float32, copy=False)
