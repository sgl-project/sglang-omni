# SPDX-License-Identifier: Apache-2.0
"""Names of the Nemotron Transcribe languages, keyed by BCP 47 primary subtag."""

from __future__ import annotations

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "es": "Spanish",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "hy": "Armenian",
    "nn": "Norwegian",
    "nb": "Norwegian",
    "no": "Norwegian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "et": "Estonian",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "hu": "Hungarian",
    "id": "Indonesian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "ta": "Tamil",
    "bn": "Bengali",
    "pl": "Polish",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "tr": "Turkish",
    "tl": "Filipino",
    "fil": "Filipino",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
}


def language_name(code: str | None) -> str | None:
    """The name for a tag such as ``en``, ``en-US`` or ``de_DE``; None when unknown."""
    if not code:
        return None
    else:
        pass
    parts = str(code).strip().lower().replace("_", "-").split("-")
    if not parts[0] or not all(part.isalnum() for part in parts):
        return None
    else:
        return LANGUAGE_NAMES.get(parts[0])


def translation_instruction(source_language: str | None) -> str:
    """The English translation instruction, naming the source language when known."""
    name = language_name(source_language)
    if name is None:
        return "Translate the audio into English."
    else:
        return f"Translate the {name} audio into English."


__all__ = ["LANGUAGE_NAMES", "language_name", "translation_instruction"]
