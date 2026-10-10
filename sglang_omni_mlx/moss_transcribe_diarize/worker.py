# SPDX-License-Identifier: Apache-2.0
"""Runs MOSS-TD transcriptions on the worker thread that loaded the model."""

from __future__ import annotations

from pathlib import Path

from sglang_omni_mlx.moss_transcribe_diarize.transcriber import (
    MossTranscribeDiarizeTranscriber,
    TranscriptionOptions,
    TranscriptionResult,
)
from sglang_omni_mlx.transcription import SerialWorker


class TranscriptionWorker(SerialWorker[TranscriptionOptions, TranscriptionResult]):
    transcriber: MossTranscribeDiarizeTranscriber

    def __init__(self, model_directory: Path) -> None:
        super().__init__(
            lambda: MossTranscribeDiarizeTranscriber(model_directory),
            thread_name="moss-transcribe-diarize",
        )
