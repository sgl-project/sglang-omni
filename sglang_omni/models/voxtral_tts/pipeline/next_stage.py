"""Stage routing for Voxtral TTS pipeline."""

from __future__ import annotations

from typing import Any

PREPROCESSING_STAGE = "preprocessing"
GENERATION_STAGE = "tts_generation"
VOCODER_STAGE = "vocoder"


def preprocessing_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return GENERATION_STAGE


def generation_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return VOCODER_STAGE


def vocoder_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return None
