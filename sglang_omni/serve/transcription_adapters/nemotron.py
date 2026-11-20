# SPDX-License-Identifier: Apache-2.0
"""Nemotron Transcribe verbose_json and subtitle capability.

srt and vtt are accepted so the request can be aligned. The cues themselves
are built from the CTC words after decode. This adapter does not invent a
segment from 0 to the file duration.
"""

from __future__ import annotations

from sglang_omni.serve.protocol import TranscriptionVerboseResponse
from sglang_omni.serve.transcription_adapters.base import (
    DefaultTranscriptionAdapter,
    register_transcription_adapter,
)


@register_transcription_adapter("Nemotron3_5Transcribe")
class NemotronTranscriptionAdapter(DefaultTranscriptionAdapter):
    @property
    def supports_segment_timestamps(self) -> bool:
        return True

    @property
    def supports_word_timestamps(self) -> bool:
        return True

    def build_timestamped_response(
        self,
        text: str,
        language: str | None,
        audio_duration_s: float,
    ) -> TranscriptionVerboseResponse:
        del text, language, audio_duration_s
        raise ValueError("subtitles require aligned word timestamps")
