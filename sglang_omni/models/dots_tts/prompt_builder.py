# SPDX-License-Identifier: Apache-2.0
"""Generation-schedule assembly for dots.tts.

Ports the minimal request-shaping logic from the upstream ``DotsTtsRuntime``
(``_prepare_inputs`` / ``_build_runtime_generation_schedule``): continuation
cloning concatenates the reference transcript and the target text into a
single ``{text}`` slot of the TTS template, and the audio slot expands into
one ``<|audio_gen_start|>`` plus ``max_audio_patches`` span placeholders.
Language tagging and text normalization are deliberately left out of this
first serving revision.
"""

from __future__ import annotations

from typing import Any

from sglang_omni.models.dots_tts.components.data.tokenizing import (
    build_generation_schedule,
)
from sglang_omni.models.dots_tts.components.data.tts_pipeline import (
    DEFAULT_TRAIN_TEMPLATE,
)

DOTS_TTS_MAX_SEQUENCE_LENGTH = 2048


def normalize_prompt_text(prompt_text: str | None) -> str:
    """Reference transcript as the upstream runtime feeds it to the template."""
    if prompt_text is None:
        return ""
    stripped = prompt_text.strip()
    if not stripped:
        return ""
    # note (chenyang): upstream appends a newline so the continuation boundary
    # between the reference transcript and the target text stays tokenized the
    # same way it was during training.
    return f"{stripped}\n"


def build_dots_tts_schedule_ids(
    *,
    tokenizer: Any,
    text: str,
    prompt_text: str | None,
    max_audio_patches: int,
) -> list[int]:
    """Token ids of the flat generation schedule for one request."""
    if max_audio_patches <= 0:
        raise ValueError(
            f"dots.tts max_audio_patches must be positive, got {max_audio_patches}"
        )
    target_text = text.strip()
    if not target_text:
        raise ValueError("dots.tts requires non-empty target text")

    return build_generation_schedule(
        text=f"{normalize_prompt_text(prompt_text)}{target_text}",
        tokenizer=tokenizer,
        template=DEFAULT_TRAIN_TEMPLATE,
        max_audio_tokens=int(max_audio_patches),
    )


__all__ = [
    "DOTS_TTS_MAX_SEQUENCE_LENGTH",
    "build_dots_tts_schedule_ids",
    "normalize_prompt_text",
]
