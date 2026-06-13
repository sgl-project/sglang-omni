# SPDX-License-Identifier: Apache-2.0
"""HTTP client helpers for the S2-Pro TTS playground."""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx

from playground.s2pro.audio_stream import (
    DEFAULT_S2PRO_SAMPLE_RATE,
    PcmChunkAssembler,
    PcmStreamFormat,
    SpeechStreamEvent,
)
from playground.s2pro.models import NonStreamingSpeechResult, SpeechSynthesisRequest


class SpeechDemoClientError(RuntimeError):
    """Raised when the playground speech request fails."""


def _response_pcm_format(response: httpx.Response) -> PcmStreamFormat:
    bit_depth = int(response.headers.get("x-bit-depth", "16"))
    if bit_depth <= 0 or bit_depth % 8 != 0:
        raise SpeechDemoClientError(f"Unsupported PCM bit depth: {bit_depth}")

    return PcmStreamFormat(
        sample_rate=int(
            response.headers.get("x-sample-rate", str(DEFAULT_S2PRO_SAMPLE_RATE))
        ),
        channels=int(response.headers.get("x-channels", "1")),
        sample_width=bit_depth // 8,
    )


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
        payload["response_format"] = "pcm"
        saw_audio = False

        try:
            with httpx.stream(
                "POST",
                f"{self._api_base}/v1/audio/speech",
                json=payload,
                timeout=None,
            ) as response:
                response.raise_for_status()
                pcm_format = _response_pcm_format(response)
                assembler = PcmChunkAssembler(pcm_format)
                for chunk in response.iter_raw():
                    aligned_chunk = assembler.add_chunk(chunk)
                    if not aligned_chunk:
                        continue
                    saw_audio = True
                    yield SpeechStreamEvent(
                        audio_bytes=aligned_chunk,
                        sample_rate=pcm_format.sample_rate,
                        channels=pcm_format.channels,
                        sample_width=pcm_format.sample_width,
                    )
                assembler.flush()
        except Exception as exc:
            raise SpeechDemoClientError(str(exc)) from exc

        if not saw_audio:
            raise SpeechDemoClientError("Speech stream ended without audio bytes.")
