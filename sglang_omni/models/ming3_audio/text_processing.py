# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU-only cleanup and bounded segmentation for Ming speech."""

from __future__ import annotations

import unicodedata

from sglang_omni.models.ming3_audio.text_processing_contract import (
    TextPlan,
    TextPlanningError,
    TextSegmentationError,
    TextTooLargeError,
    TextUnbreakableError,
    UnsupportedTextError,
)
from sglang_omni.models.ming3_audio.text_processing_markdown import (
    clean_markdown,
    has_presentation_markup,
)
from sglang_omni.models.ming3_audio.text_processing_segments import (
    has_meaningful_content,
    segment_tokens,
    tokenize_text,
    validate_tokens,
)

__all__ = [
    "DEFAULT_ENABLE_TEXT_SEGMENTATION",
    "DEFAULT_MAX_FRAGMENT_CHARS",
    "DEFAULT_MAX_FRAGMENTS",
    "DEFAULT_MAX_INPUT_CHARS",
    "DEFAULT_MAX_SEMANTIC_LENGTH",
    "DEFAULT_MAX_TOTAL_DECODE_STEPS",
    "MAX_ALLOWED_FRAGMENT_CHARS",
    "TextPlan",
    "TextPlanningError",
    "TextSegmentationError",
    "TextTooLargeError",
    "TextUnbreakableError",
    "UnsupportedTextError",
    "has_meaningful_content",
    "maximum_admitted_fragment_count",
    "plan_speech_text",
    "validate_text_limits",
]

DEFAULT_ENABLE_TEXT_SEGMENTATION = False
DEFAULT_MAX_FRAGMENT_CHARS = 96
DEFAULT_MAX_FRAGMENTS = 32
DEFAULT_MAX_INPUT_CHARS = 8192
DEFAULT_MAX_SEMANTIC_LENGTH = 48
DEFAULT_MAX_TOTAL_DECODE_STEPS = 16384
MAX_ALLOWED_FRAGMENT_CHARS = 512


def plan_speech_text(
    raw_text: str,
    *,
    max_semantic_length: int = DEFAULT_MAX_SEMANTIC_LENGTH,
    max_fragment_chars: int = DEFAULT_MAX_FRAGMENT_CHARS,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    max_fragments: int = DEFAULT_MAX_FRAGMENTS,
) -> TextPlan:
    """Clean and segment text without importing a model, tokenizer, or accelerator."""
    validate_text_limits(
        raw_text,
        max_semantic_length,
        max_fragment_chars,
        max_fragments,
        max_input_chars,
    )
    if not raw_text.strip():
        raise TextPlanningError("speech text is empty or whitespace-only")
    else:
        pass

    if has_presentation_markup(raw_text):
        cleaned_text = clean_markdown(raw_text)
    else:
        cleaned_text = raw_text
    if not has_meaningful_content(cleaned_text):
        raise TextPlanningError("speech text is empty or unpronounceable after cleanup")
    else:
        pass

    tokens = tokenize_text(cleaned_text)
    validate_tokens(tokens, max_fragment_chars, max_semantic_length)
    semantic_length = sum(token.semantic_length for token in tokens)
    if (
        len(cleaned_text) <= max_fragment_chars
        and semantic_length <= max_semantic_length
    ):
        return TextPlan(cleaned_text=cleaned_text, fragments=(cleaned_text,))
    else:
        pass

    fragments = segment_tokens(
        tokens,
        max_semantic_length=max_semantic_length,
        max_fragment_chars=max_fragment_chars,
        max_fragments=max_fragments,
    )
    if not fragments:
        raise TextSegmentationError("speech segmentation produced no fragments")
    else:
        pass
    if "".join(fragments) != cleaned_text:
        raise TextSegmentationError("segmentation changed the cleaned text")
    else:
        pass
    return TextPlan(cleaned_text=cleaned_text, fragments=tuple(fragments))


def validate_limits(
    max_semantic_length: int,
    max_fragment_chars: int,
    max_input_chars: int,
    max_fragments: int,
) -> None:
    if max_semantic_length < 1:
        raise ValueError("max_semantic_length must be positive")
    elif max_fragment_chars < 1:
        raise ValueError("max_fragment_chars must be positive")
    elif max_fragment_chars > MAX_ALLOWED_FRAGMENT_CHARS:
        raise ValueError(
            f"max_fragment_chars cannot exceed {MAX_ALLOWED_FRAGMENT_CHARS}"
        )
    elif max_input_chars < 1:
        raise ValueError("max_input_chars must be positive")
    elif max_fragments < 1:
        raise ValueError("max_fragments must be positive")
    else:
        pass


def validate_text_limits(
    raw_text: str,
    max_semantic_length: int,
    max_fragment_chars: int,
    max_fragments: int,
    max_input_chars: int,
) -> None:
    """Validate planner inputs without cleaning or changing the text."""
    validate_limits(
        max_semantic_length,
        max_fragment_chars,
        max_input_chars,
        max_fragments,
    )
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    else:
        pass
    validate_input(raw_text, max_input_chars)


def maximum_admitted_fragment_count(
    max_decode_steps: int, max_total_decode_steps: int
) -> int:
    """Return the fragment ceiling imposed by the total decode budget."""
    if max_decode_steps <= 0:
        raise ValueError("max_decode_steps must be greater than zero")
    elif max_total_decode_steps <= 0:
        raise ValueError("max_total_decode_steps must be greater than zero")
    else:
        pass
    maximum_fragment_count = max_total_decode_steps // max_decode_steps
    if maximum_fragment_count <= 0:
        raise ValueError(
            "max_total_decode_steps must admit at least one fragment "
            "at max_decode_steps"
        )
    else:
        return maximum_fragment_count


def validate_input(raw_text: str, max_input_chars: int) -> None:
    if len(raw_text) > max_input_chars:
        raise TextTooLargeError(
            f"input has {len(raw_text)} characters; limit is {max_input_chars}"
        )
    else:
        pass
    if "\x00" in raw_text:
        raise UnsupportedTextError("input contains an unsupported control character")
    else:
        pass
    for character in raw_text:
        codepoint = ord(character)
        if unicodedata.category(character) in ("Co", "Cs"):
            raise UnsupportedTextError("input contains private-use or surrogate text")
        elif codepoint < 32 and character not in "\n\r\t":
            raise UnsupportedTextError(
                "input contains an unsupported control character"
            )
        else:
            pass
