# SPDX-License-Identifier: Apache-2.0
"""Optional text normalization helpers for ZONOS2 TTS."""

from __future__ import annotations

import os
from typing import Any


_LANGUAGE_ALIASES = {
    "en": "en_us",
    "en-us": "en_us",
    "en_us": "en_us",
    "en-gb": "en_gb",
    "en_gb": "en_gb",
    "zh": "cmn",
    "zh-cn": "cmn",
    "zh_cn": "cmn",
    "cmn": "cmn",
    "fr": "fr_fr",
    "fr-fr": "fr_fr",
    "fr_fr": "fr_fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt": "pt_br",
    "pt-br": "pt_br",
    "pt_br": "pt_br",
    "ja": "ja",
    "jp": "ja",
    "ko": "ko",
}

_NEMO_LANGUAGE = {
    "en_us": "en",
    "en_gb": "en",
    "fr_fr": "fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt_br": "pt",
    "ja": "ja",
    "cmn": "zh",
    "ko": "ko",
}

_NORMALIZER: "Zonos2TextNormalizer | None" = None


def normalize_zonos2_language(language: Any) -> str | None:
    if language is None:
        return None
    normalized = str(language).strip().lower().replace("_", "-")
    if not normalized:
        return None
    return _LANGUAGE_ALIASES.get(normalized, normalized.replace("-", "_"))


def zonos2_text_normalization_enabled() -> bool:
    return os.environ.get("ZONOS2_TTS_NORM", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class Zonos2TextNormalizer:
    """Lazy wrapper around the optional ZONOS2/NeMo text normalizer."""

    def __init__(self) -> None:
        self._normalizers: dict[str, Any] = {}

    def _build(self, language: str) -> Any:
        del language
        from zonos2.tokenizer.textnorm import TTSTextNormalizer

        return TTSTextNormalizer()

    def normalize(self, text: str, language: str | None) -> str:
        normalized_language = normalize_zonos2_language(language)
        nemo_language = _NEMO_LANGUAGE.get(normalized_language or "")
        if nemo_language is None:
            return text
        try:
            normalizer = self._normalizers.get(nemo_language)
            if normalizer is None:
                normalizer = self._build(nemo_language)
                self._normalizers[nemo_language] = normalizer
            result = normalizer.normalize(text, nemo_language)
        except Exception:
            return text
        if isinstance(result, str) and result.strip():
            return result
        return text


def normalize_zonos2_text(
    text: str,
    *,
    language: str | None,
    enabled: bool = True,
) -> str:
    if not enabled or not zonos2_text_normalization_enabled():
        return text
    global _NORMALIZER
    if _NORMALIZER is None:
        _NORMALIZER = Zonos2TextNormalizer()
    return _NORMALIZER.normalize(text, language)
