# SPDX-License-Identifier: Apache-2.0
"""Chunk-context orchestration for Whisper transcription."""

from __future__ import annotations

import unicodedata

from sglang_omni.serve.transcription_adapters.base import (
    DefaultTranscriptionAdapter,
    register_transcription_adapter,
)

_MIN_REPEAT_PERIOD_UNITS = 8
_MAX_REPEAT_PERIOD_UNITS = 128
_REPEAT_COPIES = 3


def _has_sustained_terminal_repetition(text: str) -> bool:
    """Detect 8-128 normalized units repeated at least three times at the end."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    units = tuple(
        char
        for char in normalized
        if char.isalnum() or unicodedata.category(char).startswith("M")
    )
    max_period = min(_MAX_REPEAT_PERIOD_UNITS, len(units) // _REPEAT_COPIES)
    for period in range(_MIN_REPEAT_PERIOD_UNITS, max_period + 1):
        suffix = units[-period:]
        if all(
            units[-copy * period : -(copy - 1) * period] == suffix
            for copy in range(2, _REPEAT_COPIES + 1)
        ):
            return True
    return False


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

    def should_retry_chunk_without_context(self, text: str) -> bool:
        return _has_sustained_terminal_repetition(text)
