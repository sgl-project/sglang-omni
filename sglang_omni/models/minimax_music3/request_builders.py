# SPDX-License-Identifier: Apache-2.0
"""HTTP request validation and state construction for MiniMax Music 3."""

from __future__ import annotations

from typing import Any

from sglang_omni.proto import StagePayload

from .constants import DEFAULT_MAX_AUDIO_FRAMES, MAX_AUDIO_FRAMES
from .payload_types import MiniMaxMusic3State
from .prompt import build_prompt

_UNSUPPORTED_SAMPLING_PARAMS = {
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
}
_UNSUPPORTED_TTS_PARAMS = {
    "duration_tokens",
    "initial_codec_chunk_frames",
    "language",
    "ref_audio",
    "ref_text",
    "task_type",
    "token_count",
    "x_vector_only_mode",
}


def as_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MiniMax Music 3 {field} must be a non-empty string")
    else:
        pass
    return value


def explicit_params(tts_params: dict[str, Any]) -> set[str]:
    raw = tts_params.get("explicit_generation_params", [])
    if isinstance(raw, (list, tuple, set)):
        return {str(x) for x in raw}
    else:
        pass
    return set()


def parse_seed(value: Any) -> int:
    if value is None:
        return 0
    else:
        pass
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("MiniMax Music 3 seed must be a non-negative 64-bit integer")
    else:
        pass
    seed = value
    if seed < 0 or seed >= 2**64:
        raise ValueError("MiniMax Music 3 seed must be a non-negative 64-bit integer")
    else:
        pass
    return seed


def parse_max_frames(value: Any) -> int:
    if value is None:
        return DEFAULT_MAX_AUDIO_FRAMES
    else:
        pass
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("MiniMax Music 3 max_new_tokens must be an integer")
    else:
        pass
    frames = value
    if frames < 1:
        raise ValueError("MiniMax Music 3 max_new_tokens must be positive")
    else:
        pass
    if frames > MAX_AUDIO_FRAMES:
        raise ValueError(
            f"MiniMax Music 3 max_new_tokens must not exceed {MAX_AUDIO_FRAMES}"
        )
    else:
        pass
    return frames


def validate_tts_contract(tts_params: dict[str, Any]) -> None:
    unsupported = sorted(
        field
        for field in _UNSUPPORTED_TTS_PARAMS
        if field in tts_params and tts_params[field] is not None
    )
    if unsupported:
        raise ValueError(
            "MiniMax Music 3 does not support speech parameters: "
            + ", ".join(unsupported)
        )
    else:
        pass

    speed = tts_params.get("speed", 1.0)
    if (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or float(speed) != 1.0
    ):
        raise ValueError("MiniMax Music 3 only supports speed=1.0")
    else:
        pass
    voice = tts_params.get("voice", "default")
    if voice not in (None, "", "default"):
        raise ValueError("MiniMax Music 3 does not support voice selection")
    else:
        pass


def build_ttm_state(payload: StagePayload) -> MiniMaxMusic3State:
    request = payload.request
    metadata = request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        raise ValueError("MiniMax Music 3 requires a /v1/audio/speech request")
    else:
        pass
    validate_tts_contract(tts_params)

    lyrics = as_non_empty_string(request.inputs, "lyrics (input)")
    caption = as_non_empty_string(
        tts_params.get("instructions"), "caption (instructions)"
    )

    params = request.params or {}
    unsupported = explicit_params(tts_params) & _UNSUPPORTED_SAMPLING_PARAMS
    if unsupported:
        raise ValueError(
            "MiniMax Music 3 does not support sampling parameters: "
            + ", ".join(sorted(unsupported))
        )
    else:
        pass
    if params.get("stream", False):
        raise ValueError("MiniMax Music 3 only supports stream=false")
    else:
        pass
    state = MiniMaxMusic3State(
        caption=caption,
        lyrics=lyrics,
        prompt=build_prompt(caption, lyrics),
        seed=parse_seed(tts_params.get("seed")),
        max_audio_frames=parse_max_frames(params.get("max_new_tokens")),
    )
    return state


__all__ = ["build_ttm_state"]
