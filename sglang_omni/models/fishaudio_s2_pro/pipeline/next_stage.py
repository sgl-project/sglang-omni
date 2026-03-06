# SPDX-License-Identifier: Apache-2.0
"""Stage routing callbacks for the FishAudio S2-Pro TTS pipeline."""

from __future__ import annotations

PREPROCESSING_STAGE = "preprocessing"
TTS_ENGINE_STAGE = "tts_engine"
VOCODER_STAGE = "vocoder"


def preprocessing_next(_payload) -> str | None:
    return TTS_ENGINE_STAGE


def tts_engine_next(_payload) -> str | None:
    return VOCODER_STAGE


def vocoder_next(_payload) -> str | None:
    return None
