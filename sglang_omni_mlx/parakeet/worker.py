# SPDX-License-Identifier: Apache-2.0
"""Runs every Parakeet transcription on one thread, the thread that loaded the model."""

from __future__ import annotations

from pathlib import Path

from sglang_omni_mlx.parakeet.transcriber import (
    ParakeetTranscriber,
    TranscriptionOptions,
    TranscriptionResult,
)
from sglang_omni_mlx.transcription import SerialWorker


class TranscriptionWorker(SerialWorker[TranscriptionOptions, TranscriptionResult]):
    transcriber: ParakeetTranscriber

    def __init__(self, model_directory: Path, dtype: str) -> None:
        super().__init__(
            lambda: ParakeetTranscriber(model_directory, dtype), thread_name="parakeet"
        )
