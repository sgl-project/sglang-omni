# SPDX-License-Identifier: Apache-2.0
"""HTTP request validation and state construction for YuE2."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sglang_omni.proto import StagePayload

from .constants import (
    CONTEXT_TOKENS,
    DEFAULT_ABC_MAX_TOKENS,
    DEFAULT_COT,
    DEFAULT_ODE_STEPS,
    DEFAULT_SEMANTIC_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_STYLE,
    DEFAULT_VAE_CORE_FRAMES,
    DEFAULT_VAE_HALO_FRAMES,
)
from .payload_types import Yue2State

_ALLOWED_COT = {"off", "melody", "full"}
_UNSUPPORTED_TTS_PARAMS = {
    "language",
    "ref_audio",
    "ref_text",
    "task_type",
    "x_vector_only_mode",
}


def as_string(value: object, default: str = "") -> str:
    if value is None:
        return default
    else:
        pass
    return str(value)


def as_int(value: object, default: int, minimum: int = 1) -> int:
    if value is None:
        return default
    else:
        pass
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"YuE2 expected an integer, got {type(value).__name__}")
    else:
        pass
    if value < minimum:
        raise ValueError(f"YuE2 value must be >= {minimum}, got {value}")
    else:
        pass
    return int(value)


def as_seed(value: object) -> int:
    if value is None:
        return DEFAULT_SEED
    else:
        pass
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= 2**63
    ):
        raise ValueError("YuE2 seed must be a non-negative 63-bit integer")
    else:
        pass
    return int(value)


def as_cot(value: object) -> str:
    cot = as_string(value, DEFAULT_COT)
    if cot not in _ALLOWED_COT:
        raise ValueError(f"YuE2 cot must be one of {sorted(_ALLOWED_COT)}, got {cot!r}")
    else:
        pass
    return cot


def validate_tts_contract(tts_params: Mapping[str, object]) -> None:
    unsupported = sorted(
        field
        for field in _UNSUPPORTED_TTS_PARAMS
        if field in tts_params and tts_params[field] is not None
    )
    if unsupported:
        raise ValueError(
            "YuE2 does not support speech parameters: " + ", ".join(unsupported)
        )
    else:
        pass
    voice = tts_params.get("voice", "default")
    if voice not in (None, "", "default"):
        raise ValueError("YuE2 does not support voice selection")
    else:
        pass


def parse_input_payload(raw: object) -> dict | None:
    """YuE2 accepts a JSON object as ``input`` to carry cot/abc/seed, so the
    four manual tasks (full/melody/harmony) fit the plain OpenAI schema."""
    if not isinstance(raw, str):
        return None
    else:
        pass
    text = raw.lstrip()
    if not text.startswith("{"):
        return None
    else:
        pass
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    else:
        pass
    if isinstance(value, dict):
        return value
    else:
        pass
    return None


def build_yue2_state(payload: StagePayload) -> Yue2State:
    """OpenAI ``input``=lyrics (or a JSON object), ``instructions``=style."""
    request = payload.request
    metadata = request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        raise ValueError("YuE2 requires a /v1/audio/speech request")
    else:
        pass
    validate_tts_contract(tts_params)

    params = request.params or {}
    obj = parse_input_payload(request.inputs)
    if obj is not None:
        style = as_string(obj.get("style") or obj.get("tags"), DEFAULT_STYLE)
        lyrics = as_string(obj.get("lyrics"), "")
        cot = as_string(obj.get("cot"), DEFAULT_COT)
        seed = as_seed(obj.get("seed"))
        abc = as_string(obj.get("abc"), "") or None
        cfg_scale = float(obj.get("cfg_scale") or 1.0)
        abc_max_tokens = as_int(obj.get("abc_max_tokens"), DEFAULT_ABC_MAX_TOKENS)
        semantic_max_tokens = as_int(
            obj.get("semantic_max_tokens"), DEFAULT_SEMANTIC_MAX_TOKENS)
        ode_steps = as_int(obj.get("ode_steps"), DEFAULT_ODE_STEPS)
        vae_core_frames = as_int(obj.get("vae_core_frames"), DEFAULT_VAE_CORE_FRAMES)
        vae_halo_frames = as_int(
            obj.get("vae_halo_frames"), DEFAULT_VAE_HALO_FRAMES, minimum=0)
    else:
        style = as_string(tts_params.get("instructions"), DEFAULT_STYLE) or DEFAULT_STYLE
        lyrics = as_string(request.inputs, "")
        cot = as_string(tts_params.get("cot"), DEFAULT_COT)
        seed = as_seed(tts_params.get("seed"))
        abc = as_string(tts_params.get("abc"), "") or None
        cfg_scale = float(tts_params.get("cfg_scale") or 1.0)
        abc_max_tokens = as_int(
            tts_params.get("abc_max_tokens"), DEFAULT_ABC_MAX_TOKENS)
        semantic_max_tokens = as_int(
            params.get("max_new_tokens"), DEFAULT_SEMANTIC_MAX_TOKENS)
        ode_steps = as_int(tts_params.get("ode_steps"), DEFAULT_ODE_STEPS)
        vae_core_frames = as_int(
            tts_params.get("vae_core_frames"), DEFAULT_VAE_CORE_FRAMES)
        vae_halo_frames = as_int(
            tts_params.get("vae_halo_frames"), DEFAULT_VAE_HALO_FRAMES, minimum=0)

    return Yue2State(
        style=style,
        lyrics=lyrics,
        cot=as_cot(cot),
        seed=seed,
        abc=abc,
        cfg_scale=cfg_scale,
        abc_max_tokens=abc_max_tokens,
        semantic_max_tokens=min(CONTEXT_TOKENS, semantic_max_tokens),
        ode_steps=ode_steps,
        vae_core_frames=vae_core_frames,
        vae_halo_frames=vae_halo_frames,
    )


__all__ = ["build_yue2_state"]
