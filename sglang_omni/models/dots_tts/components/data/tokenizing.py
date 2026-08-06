from __future__ import annotations

import re
from typing import Any

from sglang_omni.models.dots_tts.components.utils.tokenizer import (
    AUDIO_GEN_SPAN_TOKEN,
    AUDIO_GEN_START_TOKEN,
    require_token_id,
)

TEMPLATE_PATTERN = re.compile(r"\{text\}|\{audio\}|[^\{]+")


def build_generation_schedule(
    *,
    text: str,
    tokenizer: Any,
    template: str,
    max_audio_tokens: int,
) -> list[int]:
    """Flat token schedule for one request: template text plus the audio spans.

    The audio slot expands to ``<|audio_gen_start|>`` followed by
    ``max_audio_tokens`` span placeholders; the engine splits that run into the
    prompt-audio prefill prefix and the decode budget.
    """
    if max_audio_tokens <= 0:
        raise ValueError("max_audio_tokens must be positive for generation.")
    parts = re.findall(TEMPLATE_PATTERN, template)
    if "{audio}" not in parts:
        raise ValueError("dots.tts generation template must contain {audio}.")

    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    schedule_ids: list[int] = []
    for part in parts:
        if part == "{audio}":
            schedule_ids.append(require_token_id(tokenizer, AUDIO_GEN_START_TOKEN))
            schedule_ids.extend(
                [require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)]
                * int(max_audio_tokens)
            )
        elif part == "{text}":
            schedule_ids.extend(text_tokens)
        else:
            schedule_ids.extend(tokenizer.encode(part, add_special_tokens=False))
    return schedule_ids


__all__ = ["build_generation_schedule"]
