# SPDX-License-Identifier: Apache-2.0
"""Chunk-context orchestration for Whisper transcription."""

from __future__ import annotations

from sglang_omni.serve.transcription_adapters.base import (
    DefaultTranscriptionAdapter,
    register_transcription_adapter,
)


@register_transcription_adapter("Whisper")
class WhisperTranscriptionAdapter(DefaultTranscriptionAdapter):
    @property
    def requires_ordered_chunk_decoding(self) -> bool:
        return True

    def chunk_prompt(
        self,
        *,
        caller_prompt: str | None,
        previous_text: str | None,
        is_first_decoded_chunk: bool,
    ) -> str | None:
        if is_first_decoded_chunk:
            return caller_prompt
        return previous_text
