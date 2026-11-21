# SPDX-License-Identifier: Apache-2.0
"""Transcript tokens that word and character error rates count."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

# Unicode name prefixes of the scripts written without spaces between words.
CHARACTER_SCRIPTS = (
    "CJK",
    "IDEOGRAPHIC",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "THAI",
    "LAO",
    "KHMER",
    "MYANMAR",
)


def tokens(text: str, language: str) -> list[str]:
    """Words, except one token per CHARACTER_SCRIPTS character; a combining mark stays with the token before it."""
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "en":
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        return [token.strip("'") for token in text.split() if token.strip("'")]
    else:
        result: list[str] = []
        open_token: Literal["word", "character"] | None = None
        for character in text:
            category = unicodedata.category(character)[0]
            if category == "M":
                if open_token is not None:
                    result[-1] += character
                else:
                    pass
            elif category in "LN" and unicodedata.name(character, "").startswith(
                CHARACTER_SCRIPTS
            ):
                result.append(character)
                open_token = "character"
            elif category in "LN" or character == "'":
                if open_token == "word":
                    result[-1] += character
                else:
                    result.append(character)
                open_token = "word"
            else:
                open_token = None
        return result
