# SPDX-License-Identifier: Apache-2.0
"""HTTP client helpers for the S2-Pro TTS playground."""

from __future__ import annotations

import time

import httpx

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
