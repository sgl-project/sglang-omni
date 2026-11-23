# SPDX-License-Identifier: Apache-2.0
"""Transcript tokens that word and character error rates count."""

from __future__ import annotations

import re
import unicodedata


def tokens(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "en":
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        return [token.strip("'") for token in text.split() if token.strip("'")]
    else:
        return re.findall(r"[㐀-鿿豈-﫿]|[a-z0-9']+", text)
