# SPDX-License-Identifier: Apache-2.0
"""Runs every Fun-ASR transcription on one thread, the thread that loaded the model."""

from __future__ import annotations

from pathlib import Path

from sglang_omni_mlx.fun_asr.transcriber import (
    FunASRTranscriber,
    TranscriptionOptions,
    TranscriptionResult,
)
from sglang_omni_mlx.transcription import SerialWorker


class TranscriptionWorker(SerialWorker[TranscriptionOptions, TranscriptionResult]):
    transcriber: FunASRTranscriber

    def __init__(self, model_directory: Path) -> None:
        super().__init__(
            lambda: FunASRTranscriber(model_directory), thread_name="fun-asr"
        )
