# SPDX-License-Identifier: Apache-2.0
"""Request lowering for dots.tts."""

from __future__ import annotations

import math
from typing import Any

from sglang_omni.models.dots_tts.payload_types import (
    DOTS_TTS_DEFAULT_GUIDANCE_SCALE,
    DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES,
    DOTS_TTS_DEFAULT_NUM_STEPS,
    DOTS_TTS_DEFAULT_SPEAKER_SCALE,
    DotsTtsState,
    store_dots_tts_state,
)
from sglang_omni.proto import StagePayload

_REFERENCE_CONTRACT_ERROR = (
    "dots.tts requires exactly one reference with a local audio path and a "
    "non-empty reference transcript"
)
_INLINE_REFERENCE_ERROR = (
    "dots.tts reference audio must be a local file path; inline or URL "
    "reference audio is not supported"
)
_REFERENCE_AUDIO_FIELDS = ("audio_path", "ref_audio", "prompt_wav_path", "audio")
_INLINE_REFERENCE_AUDIO_FIELDS = ("data", "base64", "bytes", "url")
_DIRECT_REFERENCE_AUDIO_FIELDS = ("audio_path", "ref_audio", "prompt_wav_path")
_REFERENCE_TEXT_FIELDS = ("ref_text", "prompt_text")
_NUM_STEPS_FIELDS = ("num_steps", "dots_num_steps")
_GUIDANCE_SCALE_FIELDS = ("guidance_scale", "dots_guidance_scale")
_SPEAKER_SCALE_FIELDS = ("speaker_scale", "dots_speaker_scale")
_MAX_AUDIO_PATCHES_FIELDS = ("max_audio_patches", "max_generate_length")
# note (chenyang): the speech endpoint fills these in with its own defaults, so
# they only mean "the caller asked for it" when listed in
# explicit_generation_params. dots.tts decodes with a flow-matching solver and
# has no logits sampling knobs at all, hence the hard rejection.
_UNSUPPORTED_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "repetition_penalty")


def preprocess_dots_tts_payload(payload: StagePayload) -> StagePayload:
    """Validate the request and store the dots.tts state on the payload."""
    return store_dots_tts_state(payload, build_dots_tts_state(payload))


def build_dots_tts_state(payload: StagePayload) -> DotsTtsState:
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}
    stage_params = _tts_engine_stage_params(params)

    if params.get("stream"):
        raise ValueError("dots.tts currently generates the full waveform only")

    explicit_fields = _explicit_generation_fields(tts_params)
    _reject_unsupported_sampling(
        stage_params, tts_params, params, explicit_fields=explicit_fields
    )

    inputs = payload.request.inputs
    text, references = _normalize_inputs(inputs)
    if not text:
        raise ValueError("dots.tts requires non-empty input text")

    input_dict = inputs if isinstance(inputs, dict) else {}
    ref_audio_path, prompt_text = _resolve_reference(
        references, input_dict, tts_params, params
    )

    return DotsTtsState(
        text=text,
        prompt_text=prompt_text,
        ref_audio_path=ref_audio_path,
        **_resolve_generation_params(
            stage_params, tts_params, params, explicit_fields=explicit_fields
        ),
    )


def _resolve_generation_params(
    stage_params: dict[str, Any],
    tts_params: dict[str, Any],
    params: dict[str, Any],
    *,
    explicit_fields: set[str],
) -> dict[str, Any]:
    """Solver knobs, most specific source first: stage params, tts_params, params."""
    sources = (stage_params, tts_params, params)
    return {
        "num_steps": _resolve_positive_int(
            "num_steps",
            _first_present(*sources, names=_NUM_STEPS_FIELDS),
            default=DOTS_TTS_DEFAULT_NUM_STEPS,
        ),
        "max_audio_patches": _resolve_positive_int(
            "max_audio_patches",
            _first_present(*sources, names=_MAX_AUDIO_PATCHES_FIELDS),
            default=DOTS_TTS_DEFAULT_MAX_AUDIO_PATCHES,
        ),
        "guidance_scale": _resolve_positive_float(
            "guidance_scale",
            _first_present(*sources, names=_GUIDANCE_SCALE_FIELDS),
            default=DOTS_TTS_DEFAULT_GUIDANCE_SCALE,
        ),
        "speaker_scale": _resolve_positive_float(
            "speaker_scale",
            _first_present(*sources, names=_SPEAKER_SCALE_FIELDS),
            default=DOTS_TTS_DEFAULT_SPEAKER_SCALE,
        ),
        "seed": _resolve_seed(stage_params, tts_params, params, explicit_fields),
    }


def _tts_engine_stage_params(params: dict[str, Any]) -> dict[str, Any]:
    stage_params = params.get("stage_params")
    if not isinstance(stage_params, dict):
        return {}
    tts_engine_params = stage_params.get("tts_engine")
    return tts_engine_params if isinstance(tts_engine_params, dict) else {}


def _explicit_generation_fields(tts_params: dict[str, Any]) -> set[str]:
    explicit = tts_params.get("explicit_generation_params")
    if isinstance(explicit, (list, tuple, set)):
        return {str(field) for field in explicit}
    return set()


def _reject_unsupported_sampling(
    stage_params: dict[str, Any],
    tts_params: dict[str, Any],
    params: dict[str, Any],
    *,
    explicit_fields: set[str],
) -> None:
    for field in _UNSUPPORTED_SAMPLING_FIELDS:
        requested = (
            stage_params.get(field) is not None
            or tts_params.get(field) is not None
            or (field in explicit_fields and params.get(field) is not None)
        )
        if requested:
            raise ValueError(f"dots.tts does not use logits sampling field {field!r}")


def _normalize_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(inputs, str):
        return inputs.strip(), []
    if not isinstance(inputs, dict):
        return (str(inputs).strip() if inputs is not None else ""), []

    raw_references = inputs.get("references") or []
    if not isinstance(raw_references, list):
        raise ValueError("dots.tts references must be a list")
    if any(not isinstance(reference, dict) for reference in raw_references):
        raise ValueError("dots.tts references must be objects")
    raw_text = inputs.get("text", inputs.get("input", ""))
    text = str(raw_text).strip() if raw_text is not None else ""
    return text, [dict(reference) for reference in raw_references]


def _resolve_reference(
    references: list[dict[str, Any]],
    input_dict: dict[str, Any],
    tts_params: dict[str, Any],
    params: dict[str, Any],
) -> tuple[str, str]:
    if len(references) > 1:
        raise ValueError(_REFERENCE_CONTRACT_ERROR)

    reference = references[0] if references else {}
    if _first_present(reference, names=_INLINE_REFERENCE_AUDIO_FIELDS) is not None:
        raise ValueError(_INLINE_REFERENCE_ERROR)

    ref_audio = _first_present(reference, names=_REFERENCE_AUDIO_FIELDS)
    if ref_audio is None:
        ref_audio = _first_present(
            input_dict, tts_params, params, names=_DIRECT_REFERENCE_AUDIO_FIELDS
        )
    if ref_audio is not None and not isinstance(ref_audio, str):
        raise ValueError(_INLINE_REFERENCE_ERROR)
    ref_audio_path = ref_audio.strip() if isinstance(ref_audio, str) else ""
    if not ref_audio_path:
        raise ValueError(_REFERENCE_CONTRACT_ERROR)

    ref_text = reference.get("text")
    if ref_text is None:
        ref_text = _first_present(
            input_dict, tts_params, params, names=_REFERENCE_TEXT_FIELDS
        )
    prompt_text = str(ref_text).strip() if ref_text is not None else ""
    if not prompt_text:
        raise ValueError(_REFERENCE_CONTRACT_ERROR)
    return ref_audio_path, prompt_text


def _first_present(*sources: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for source in sources:
        for name in names:
            if source.get(name) is not None:
                return source[name]
    return None


def _resolve_seed(
    stage_params: dict[str, Any],
    tts_params: dict[str, Any],
    params: dict[str, Any],
    explicit_fields: set[str],
) -> int | None:
    seed = _first_present(stage_params, tts_params, names=("seed",))
    if seed is None and "seed" in explicit_fields:
        seed = params.get("seed")
    if seed is None:
        return None
    return _resolve_int("seed", seed)


def _resolve_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"dots.tts {name} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"dots.tts {name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dots.tts {name} must be an integer") from exc


def _resolve_positive_int(name: str, value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    resolved = _resolve_int(name, value)
    if resolved <= 0:
        raise ValueError(f"dots.tts {name} must be > 0, got {resolved}")
    return resolved


def _resolve_positive_float(name: str, value: Any, *, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        raise ValueError(f"dots.tts {name} must be a number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dots.tts {name} must be a number") from exc
    # NaN slips past the range check below because every comparison is False.
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"dots.tts {name} must be a finite number > 0")
    return resolved


__all__ = [
    "build_dots_tts_state",
    "preprocess_dots_tts_payload",
]
