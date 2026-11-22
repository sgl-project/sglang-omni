# SPDX-License-Identifier: Apache-2.0
"""Deterministic model output for testing the real speech client and serving paths."""

from collections.abc import AsyncIterator

import numpy as np
from numpy.typing import NDArray

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest


class FixedSpeechClient(Client):
    def __init__(self, audio: NDArray[np.float32], sample_rate: int = 24000) -> None:
        self.audio: NDArray[np.float32] = audio
        self.sample_rate: int = sample_rate
        self.requests: list[GenerateRequest] = []
        self.aborted: list[str] = []

    def health(self) -> dict[str, bool]:
        return {"running": True}

    async def generate(
        self, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        self.requests.append(request)
        chunk_samples = 137
        if request.stream:
            for start_sample in range(0, self.audio.size, chunk_samples):
                yield GenerateChunk(
                    request_id=request_id or "speech-test",
                    audio_data=self.audio[start_sample : start_sample + chunk_samples],
                    sample_rate=self.sample_rate,
                )
        else:
            pass
        yield GenerateChunk(
            request_id=request_id or "speech-test",
            audio_data=self.audio,
            sample_rate=self.sample_rate,
            finish_reason="stop",
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
