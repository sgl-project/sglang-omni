# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for CSM TTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sglang_omni.proto.request import EXPLICIT_GENERATION_PARAMS_KEY
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

DEFAULT_MAX_NEW_TOKENS = 125
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SPEAKER_ID = 0

# Speech and generate clients always materialize sentinel sampling values.
# Ignore those unless explicit_generation_params names the field.
_IMPLICIT_SAMPLING_DEFAULTS: dict[str, set[Any]] = {
    "temperature": {1.0, 0.8},
    "top_p": {1.0, 0.8},
    "top_k": {-1, 30},
    "repetition_penalty": {1.0, 1.1},
}


def explicit_generation_fields(tts_params: Mapping[str, Any] | None) -> set[str]:
    raw = (tts_params or {}).get(EXPLICIT_GENERATION_PARAMS_KEY)
    if isinstance(raw, (list, tuple, set)):
        return {str(field) for field in raw}
    return set()


def take_param(
    params: Mapping[str, Any],
    name: str,
    explicit_fields: set[str],
) -> Any:
    value = params.get(name)
    if value is None:
        return None
    if name in _IMPLICIT_SAMPLING_DEFAULTS and name not in explicit_fields:
        if value in _IMPLICIT_SAMPLING_DEFAULTS[name]:
            return None
    return value


def resolve_csm_generation_params(
    params: Mapping[str, Any] | None,
    tts_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve sampling. Ignore speech-API sentinels unless the field is explicit."""
    params = params or {}
    tts_params = tts_params or {}
    explicit_fields = explicit_generation_fields(tts_params)

    temperature = take_param(params, "temperature", explicit_fields)
    top_k = take_param(params, "top_k", explicit_fields)
    top_p = take_param(params, "top_p", explicit_fields)
    max_new_tokens = params.get("max_new_tokens")
    seed = params.get("seed")
    if seed is None:
        seed = tts_params.get("seed")

    resolved_temperature = (
        float(temperature) if temperature is not None else DEFAULT_TEMPERATURE
    )
    resolved_top_k = int(top_k) if top_k is not None else DEFAULT_TOP_K
    depth_temperature = params.get("depth_temperature")
    depth_top_k = params.get("depth_top_k")

    return {
        "max_new_tokens": (
            int(max_new_tokens) if max_new_tokens else DEFAULT_MAX_NEW_TOKENS
        ),
        "temperature": resolved_temperature,
        "top_k": resolved_top_k,
        "top_p": float(top_p) if top_p is not None else None,
        "depth_temperature": (
            float(depth_temperature)
            if depth_temperature is not None
            else resolved_temperature
        ),
        "depth_top_k": (
            int(depth_top_k) if depth_top_k is not None else resolved_top_k
        ),
        "seed": int(seed) if seed is not None else None,
        "stream": bool(params.get("stream", False)),
    }


@dataclass
class CsmTtsState(DeclarativeStateBase):
    """Pipeline state. max_new_tokens counts frames; vocoder trims the EOS frame."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    target_text: str | None = None
    speaker_id: int = DEFAULT_SPEAKER_ID
    context: list[dict[str, Any]] = wire(default_factory=list, codec="list")
    prompt_token_ids: list[int] = wire(default_factory=list, codec="list")
    context_codes: list[Any] | None = None
    num_ctx_codes_consumed: int = 0
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float | None = None
    depth_temperature: float = DEFAULT_TEMPERATURE
    depth_top_k: int = DEFAULT_TOP_K
    seed: int | None = None
    stream: bool = wire(False, emit="truthy", codec="bool")
    output_codes: list[Any] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0


__all__ = [
    "CsmTtsState",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SPEAKER_ID",
    "resolve_csm_generation_params",
]
