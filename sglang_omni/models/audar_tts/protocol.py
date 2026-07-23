# SPDX-License-Identifier: Apache-2.0
"""Audar prompt and acoustic-token protocol."""

from __future__ import annotations

import re

TARGET_CODES_END = "<|TARGET_CODES_END|>"
_SPEECH_TOKEN_RE = re.compile(r"<\|speech_(\d+)\|>")


def build_prompt(
    target_text: str, reference_text: str, reference_codes: list[int]
) -> str:
    reference = "".join(f"<|speech_{code}|>" for code in reference_codes)
    return (
        "user: Convert the text to speech:"
        f"<|REF_TEXT_START|>{reference_text}<|REF_TEXT_END|>"
        f"<|REF_SPEECH_START|>{reference}<|REF_SPEECH_END|>"
        f"<|TARGET_TEXT_START|>{target_text}<|TARGET_TEXT_END|>"
        "\nassistant:<|TARGET_CODES_START|>"
    )


def parse_speech_codes(text: str) -> list[int]:
    return [int(value) for value in _SPEECH_TOKEN_RE.findall(text)]
