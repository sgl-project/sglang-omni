# SPDX-License-Identifier: Apache-2.0
"""Stage routing callbacks for the FishAudio S2-Pro TTS pipeline."""

from __future__ import annotations

from typing import Any

from sglang_omni.proto import StagePayload

PREPROCESSING_STAGE = "preprocessing"
TTS_ENGINE_STAGE = "tts_engine"
VOCODER_STAGE = "vocoder"


def preprocessing_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return TTS_ENGINE_STAGE


def tts_engine_next(request_id: str, output: Any) -> str | None:
    del request_id
    if isinstance(output, StagePayload) and output.request.params.get("stream"):
        return None
    return VOCODER_STAGE


def vocoder_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return None
