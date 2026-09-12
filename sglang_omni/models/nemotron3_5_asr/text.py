# SPDX-License-Identifier: Apache-2.0
"""Nemotron transcript cleanup shared by offline and streaming paths."""

from __future__ import annotations

import re

_LOCALE_TAG_RE = re.compile(r"<(?P<locale>[A-Za-z]{2,3}-[A-Za-z]{2,3})>")


def clean_nemotron_text(raw_text: str) -> str:
    cleaned = _LOCALE_TAG_RE.sub("", raw_text)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def resolve_nemotron_locale(
    raw_text: str, requested_language: str | None
) -> str | None:
    detected: dict[str, str] = {}
    for match in _LOCALE_TAG_RE.finditer(raw_text):
        locale = match.group("locale")
        detected.setdefault(locale.casefold(), locale)
    if len(detected) == 1:
        return next(iter(detected.values()))
    if len(detected) > 1:
        return None

    requested = (requested_language or "").strip()
    if not requested or requested.casefold() == "auto":
        return None
    return requested_language


__all__ = ["clean_nemotron_text", "resolve_nemotron_locale"]
