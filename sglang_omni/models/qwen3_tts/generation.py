# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS sampling overrides and target-text generation limits."""

import math
import unicodedata
from collections.abc import Mapping

QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS = 2048
# note (AlistairShi): The 12Hz tokenizer emits 1920 samples per frame at 24000 Hz.
QWEN3_TTS_CODEC_FRAMES_PER_SECOND = 12.5
# note (AlistairShi): These are conservative safety allowances, not measured speaking rates; callers can override max_new_tokens for deliberately extended speech.
MIN_GENERATION_SECONDS = 10
GENERATION_PADDING_SECONDS = 5
LATIN_CHARACTER_SECONDS = 0.5
OTHER_CHARACTER_SECONDS = 1.0

GENERATION_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "subtalker_dosample",
    "subtalker_temperature",
    "subtalker_top_p",
    "subtalker_top_k",
    "max_new_tokens",
)


def text_generation_budget(text: str) -> int:
    """Bound generated codec frames using only the target text, excluding references."""
    character_seconds = 0.0
    for character in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(character)
        if character.isspace() or category[0] in ("M", "C"):
            continue
        elif category[0] == "L" and "LATIN" in unicodedata.name(character, ""):
            character_seconds += LATIN_CHARACTER_SECONDS
        else:
            character_seconds += OTHER_CHARACTER_SECONDS
    return min(
        QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS,
        math.ceil(
            max(MIN_GENERATION_SECONDS, GENERATION_PADDING_SECONDS + character_seconds)
            * QWEN3_TTS_CODEC_FRAMES_PER_SECOND
        ),
    )


def build_generation_kwargs(
    params: Mapping[str, object],
    *,
    text: str,
    tts_engine_params: Mapping[str, object],
) -> dict[str, object]:
    generation_kwargs: dict[str, object] = {}
    for field in GENERATION_FIELDS:
        stage_value = tts_engine_params.get(field)
        value = stage_value if stage_value is not None else params.get(field)
        if value is not None:
            generation_kwargs[field] = value
        else:
            pass
    max_new_tokens = generation_kwargs.get("max_new_tokens")
    if max_new_tokens is None:
        generation_kwargs["max_new_tokens"] = text_generation_budget(text)
    elif isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("Qwen3-TTS max_new_tokens must be a positive integer")
    elif max_new_tokens <= 0:
        raise ValueError("Qwen3-TTS max_new_tokens must be a positive integer")
    else:
        pass
    return generation_kwargs
