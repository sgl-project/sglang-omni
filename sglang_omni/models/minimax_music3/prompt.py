# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 prompt construction."""

from __future__ import annotations

import re
from typing import Any

SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|audio_end|>": 151670,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}
AUDIO_CODE_OFFSET = 151675

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def _strip_markdown_residuals(text: str) -> str:
    """Deterministically strip Markdown syntax left after optional HTML parsing."""

    if not text:
        return text
    lines_out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    return re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)


def _remove_markdown_format(text: str) -> str:
    """Remove the Markdown forms accepted by the model input contract."""

    return _strip_markdown_residuals(text).replace("• ", "").replace("    ", "")


def clean_caption(caption: str) -> str:
    """Apply source order: special tokens -> markdown -> newline collapse."""

    def _special(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_special, caption)
    text = _remove_markdown_format(text)
    # The source uses {2,}, not {3,}; this affects blank-line token positions.
    return re.sub(r"\n{2,}", "\n", text)


def _strip_text_after_leading_tags(text: str) -> str:
    """Keep only consecutive structural tags at the start of a line."""

    if not text:
        return text
    output: list[str] = []
    for line in text.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    return "\n".join(output)


def normalize_lyrics(lyrics: str) -> str:
    """Apply source _normalize_lyrics_text byte-for-byte for strings."""
    text = _strip_text_after_leading_tags(lyrics)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    text = f"[start]\n{text}"
    return text


def build_prompt(caption: str, lyrics: str) -> str:
    return (
        "<|im_start|><|caption_start|>"
        f"{clean_caption(caption)}"
        "<|caption_end|><|lyrics_start|>"
        f"{normalize_lyrics(lyrics)}"
        "<|lyrics_end|><|im_end|><|audio_start|>"
    )


def validate_tokenizer_ids(tokenizer: Any) -> None:
    """Fail fast when a tokenizer is not the supported music tokenizer."""

    for token, expected in SPECIAL_TOKEN_IDS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != expected:
            raise ValueError(
                f"MiniMax Music 3 tokenizer mismatch for {token}: "
                f"expected {expected}, got {token_id}"
            )


__all__ = [
    "AUDIO_CODE_OFFSET",
    "SPECIAL_TOKEN_IDS",
    "build_prompt",
    "clean_caption",
    "normalize_lyrics",
    "validate_tokenizer_ids",
]
