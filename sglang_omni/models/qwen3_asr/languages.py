# SPDX-License-Identifier: Apache-2.0
"""Language-hint normalization for Qwen3-ASR."""

from __future__ import annotations

# Qwen3-ASR was trained with canonical full names in the forced-language
# suffix: ``language <NAME><asr_text>``.
LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

SUPPORTED_LANGUAGE_NAMES = frozenset(LANGUAGE_CODE_TO_NAME.values())
_LANGUAGE_NAME_BY_CASEFOLD = {
    name.casefold(): name for name in SUPPORTED_LANGUAGE_NAMES
}


def resolve_language(language: str) -> str:
    """Resolve a Qwen3-ASR language code or name to its canonical prompt name."""

    value = str(language).strip()
    if not value:
        raise ValueError(
            "Unsupported language: the language hint is empty. Use a supported "
            f"language code ({', '.join(sorted(LANGUAGE_CODE_TO_NAME))}) or "
            "canonical name."
        )

    normalized = value.casefold()
    if normalized == "cn" or normalized.startswith(("zh-", "zh_")):
        return "Chinese"

    resolved = LANGUAGE_CODE_TO_NAME.get(normalized)
    if resolved is None:
        resolved = _LANGUAGE_NAME_BY_CASEFOLD.get(normalized)
    if resolved is not None:
        return resolved

    raise ValueError(
        f"Unsupported language: {language!r}. Use a supported language code "
        f"({', '.join(sorted(LANGUAGE_CODE_TO_NAME))}) or canonical name."
    )


__all__ = [
    "LANGUAGE_CODE_TO_NAME",
    "SUPPORTED_LANGUAGE_NAMES",
    "resolve_language",
]
