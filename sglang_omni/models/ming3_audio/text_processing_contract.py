# SPDX-License-Identifier: Apache-2.0
"""Shared value types and errors for the Ming speech text frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TextTokenKind = Literal[
    "cjk",
    "emoji",
    "number",
    "other",
    "punctuation",
    "url",
    "whitespace",
    "word",
]


@dataclass(frozen=True, slots=True)
class TextPlan:
    """Cleaned speech text and its bounded, ordered synthesis fragments."""

    cleaned_text: str
    fragments: tuple[str, ...]


class TextPlanningError(ValueError):
    """Base error for input that cannot be safely planned for speech."""


class UnsupportedTextError(TextPlanningError):
    """Raised when the input contains unsupported or unsafe markup."""


class TextTooLargeError(TextPlanningError):
    """Raised when an input or output plan exceeds configured bounds."""


class TextUnbreakableError(TextPlanningError):
    """Raised when one protected token cannot fit in a fragment."""


class TextSegmentationError(TextPlanningError):
    """Raised when safe segmentation cannot satisfy the fragment contract."""


@dataclass(frozen=True, slots=True)
class TextToken:
    """Atomic text unit that the segmenter is allowed to place as a whole."""

    text: str
    kind: TextTokenKind
    semantic_length: int
