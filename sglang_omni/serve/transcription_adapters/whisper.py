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


def _is_repeat_unit(char: str) -> bool:
    return char.isalnum() or unicodedata.category(char).startswith("M")


def _is_single_latin_word(
    normalized: str,
    unit_offsets: tuple[int, ...],
    start: int,
    period: int,
) -> bool:
    candidate = normalized[unit_offsets[start] : unit_offsets[start + period - 1] + 1]
    return all(_is_repeat_unit(char) for char in candidate) and all(
        char.isdigit()
        or unicodedata.category(char).startswith("M")
        or "LATIN" in unicodedata.name(char, "")
        for char in candidate
    )


def _has_sustained_repetition(text: str) -> bool:
    """Detect 8-128 normalized units repeated at least three times."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    unit_offsets = tuple(
        index for index, char in enumerate(normalized) if _is_repeat_unit(char)
    )
    units = tuple(normalized[index] for index in unit_offsets)
    max_period = min(_MAX_REPEAT_PERIOD_UNITS, len(units) // _REPEAT_COPIES)
    for period in range(_MIN_REPEAT_PERIOD_UNITS, max_period + 1):
        span = _REPEAT_COPIES * period
        for start in range(len(units) - span + 1):
            pattern = units[start : start + period]
            if (
                units[start + period : start + 2 * period] == pattern
                and units[start + 2 * period : start + span] == pattern
            ):
                if _is_single_latin_word(
                    normalized,
                    unit_offsets,
                    start,
                    period,
                ):
                    continue
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
        return _has_sustained_repetition(text)
