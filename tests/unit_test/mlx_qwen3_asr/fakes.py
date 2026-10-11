# SPDX-License-Identifier: Apache-2.0
"""A transcription worker that answers from the audio length, without a model."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

import numpy as np

from sglang_omni_mlx.qwen3_asr.transcriber import (
    TranscriptionOptions,
    TranscriptionResult,
)
from sglang_omni_mlx.transcription import FinishReason, TranscriptionCancelled


@dataclass
class TranscriptionCall:
    sample_count: int
    options: TranscriptionOptions


class PrefixTranscriber:
    """retained_prefix over whitespace words, one id per word."""

    def retained_prefix(
        self, text: str, rollback_token_count: int
    ) -> tuple[tuple[int, ...], str]:
        words = text.split(" ")
        retained = words[: max(len(words) - rollback_token_count, 0)]
        return tuple(range(len(retained))), " ".join(retained)


@dataclass
class FakeWorker:
    """Text is 'heard <sample count>' unless fixed; a gate can hold a transcription open."""

    language: str | None = "English"
    fixed_text: str | None = None
    failure: Exception | None = None
    calls: list[TranscriptionCall] = field(default_factory=list)
    gate: asyncio.Event | None = None
    transcriber: PrefixTranscriber = field(default_factory=PrefixTranscriber)

    def request_states(self) -> dict[str, int]:
        return {}

    async def transcribe(
        self,
        samples: np.ndarray,
        options: TranscriptionOptions,
        cancel: threading.Event,
    ) -> TranscriptionResult:
        self.calls.append(TranscriptionCall(len(samples), options))
        if self.gate is not None:
            await self.gate.wait()
        else:
            pass
        if self.failure is not None:
            raise self.failure
        elif cancel.is_set():
            raise TranscriptionCancelled()
        else:
            pass
        return TranscriptionResult(
            text=(
                self.fixed_text
                if self.fixed_text is not None
                else f"heard {len(samples)}"
            ),
            language=options.language or self.language,
            generated_token_count=2,
            finish_reason=FinishReason.STOP,
        )
