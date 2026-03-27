# SPDX-License-Identifier: Apache-2.0
"""HTTP client helpers for the S2-Pro TTS playground."""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx

from playground.tts.audio_stream import SpeechStreamEvent, parse_speech_stream_data
from playground.tts.models import NonStreamingSpeechResult, SpeechSynthesisRequest


class SpeechDemoClientError(RuntimeError):
    """Raised when the playground speech request fails."""


class SpeechDemoClient:
    """Small API client for the TTS playground."""

    def __init__(self, api_base: str, *, timeout_s: float = 120.0) -> None:
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s

    def synthesize(self, request: SpeechSynthesisRequest) -> NonStreamingSpeechResult:
        request.validate()

        started_at = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._api_base}/v1/audio/speech",
                json=request.to_payload(),
                timeout=self._timeout_s,
            )
            response.raise_for_status()
        except Exception as exc:
            raise SpeechDemoClientError(str(exc)) from exc

        return NonStreamingSpeechResult(
            audio_bytes=response.content,
            elapsed_s=time.perf_counter() - started_at,
        )

    def stream_synthesize(
        self, request: SpeechSynthesisRequest
    ) -> Iterator[SpeechStreamEvent]:
        request.validate()

        payload = request.to_payload()
        payload["stream"] = True
        saw_terminal_event = False
        saw_done_marker = False

        try:
            with httpx.stream(
                "POST",
                f"{self._api_base}/v1/audio/speech",
                json=payload,
                timeout=None,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    event = parse_speech_stream_data(line[len("data: ") :])
                    if event is None:
                        continue
                    if event.is_done:
                        saw_done_marker = True
                        break
                    if event.finish_reason is not None:
                        saw_terminal_event = True
                    yield event
        except Exception as exc:
            raise SpeechDemoClientError(str(exc)) from exc

        if not saw_terminal_event or not saw_done_marker:
            raise SpeechDemoClientError(
                "Speech stream ended before the terminal completion markers were received."
            )
