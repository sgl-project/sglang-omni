# SPDX-License-Identifier: Apache-2.0
"""Runs every Qwen3-ASR transcription on one thread, the thread that loaded the model."""

from __future__ import annotations

from pathlib import Path

from sglang_omni_mlx.qwen3_asr.transcriber import (
    Qwen3ASRTranscriber,
    TranscriptionOptions,
    TranscriptionResult,
)
from sglang_omni_mlx.transcription import SerialWorker


class TranscriptionWorker(SerialWorker[TranscriptionOptions, TranscriptionResult]):
    transcriber: Qwen3ASRTranscriber

    def __init__(self, model_directory: Path) -> None:
        super().__init__(
            lambda: Qwen3ASRTranscriber(model_directory), thread_name="qwen3-asr"
        )
