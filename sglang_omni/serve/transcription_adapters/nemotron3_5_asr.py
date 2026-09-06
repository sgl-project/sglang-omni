# SPDX-License-Identifier: Apache-2.0
"""Output cleanup and locale detection for Nemotron 3.5 ASR."""

from __future__ import annotations

import re

from sglang_omni.serve.transcription_adapters.base import (
    DefaultTranscriptionAdapter,
    register_transcription_adapter,
)

_LOCALE_TAG_RE = re.compile(r"<(?P<locale>[A-Za-z]{2,3}-[A-Za-z]{2,3})>")


@register_transcription_adapter("Nemotron3_5Asr")
class Nemotron3_5ASRTranscriptionAdapter(DefaultTranscriptionAdapter):
    """Keep locale tags long enough to populate ``verbose_json.language``."""

    def resolve_language(
        self,
        raw_text: str,
        requested_language: str | None,
    ) -> str | None:
        detected: dict[str, str] = {}
        for match in _LOCALE_TAG_RE.finditer(raw_text):
            locale = match.group("locale")
            detected.setdefault(locale.casefold(), locale)
        if len(detected) == 1:
            return next(iter(detected.values()))
        if len(detected) > 1:
            # Note (LG-0927): One OpenAI response has one language field. Do not
            # report an invented single locale for genuinely mixed-locale output.
            return None

        requested = (requested_language or "").strip()
        if not requested or requested.casefold() == "auto":
            return None
        return requested_language

    def postprocess_text(self, text: str) -> str:
        cleaned = _LOCALE_TAG_RE.sub("", text)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    def postprocess_plain_text(self, text: str) -> str:
        return self.postprocess_text(text)


__all__ = ["Nemotron3_5ASRTranscriptionAdapter"]
