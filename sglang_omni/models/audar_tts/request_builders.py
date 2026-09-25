# SPDX-License-Identifier: Apache-2.0
"""Request lowering for Audar-TTS."""

from __future__ import annotations

from collections.abc import Mapping

from sglang_omni.models.audar_tts.payload_types import AudarTTSState
from sglang_omni.proto import StagePayload

_DEFAULT_GENERATION = {
    "max_new_tokens": 2048,
    "temperature": 1.0,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
_IMPLICIT_SAMPLING_DEFAULTS = {
    "temperature": {0.8, 1.0},
    "top_p": {0.8, 1.0},
    "top_k": {-1, 30},
    "repetition_penalty": {1.0, 1.1},
}


def build_audar_state(payload: StagePayload) -> AudarTTSState:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}
    else:
        pass

    target_text, references = normalize_inputs(inputs)
    if not target_text.strip():
        raise ValueError("Audar-TTS requires non-empty target text")
    else:
        pass
    if len(references) > 1:
        raise ValueError("Audar-TTS accepts exactly one reference")
    else:
        pass

    reference = references[0] if references else None
    if reference is None and tts_params.get("ref_audio") is not None:
        reference = reference_from_value(tts_params["ref_audio"])
    else:
        pass
    if reference is None:
        raise ValueError("Audar-TTS requires reference audio")
    else:
        pass

    reference_text = reference.get("text") or tts_params.get("ref_text")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ValueError("Audar-TTS requires the reference transcript")
    else:
        pass

    return AudarTTSState(
        target_text=target_text,
        reference_text=reference_text,
        reference_audio=normalize_reference_audio(reference),
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
    )


def build_generation_kwargs(
    params: Mapping[str, object], *, tts_params: Mapping[str, object]
) -> dict[str, int | float]:
    generation = dict(_DEFAULT_GENERATION)
    explicit = tts_params.get("explicit_generation_params")
    explicit_fields = (
        {str(field) for field in explicit}
        if isinstance(explicit, (list, tuple, set))
        else set()
    )

    max_new_tokens = params.get("max_new_tokens")
    if max_new_tokens is not None:
        generation["max_new_tokens"] = int(max_new_tokens)
    else:
        pass
    for field in ("temperature", "top_k", "top_p", "repetition_penalty"):
        value = params.get(field)
        if value is None:
            continue
        else:
            pass
        if field not in explicit_fields and value in _IMPLICIT_SAMPLING_DEFAULTS[field]:
            continue
        else:
            pass
        generation[field] = int(value) if field == "top_k" else float(value)

    seed = tts_params.get("seed")
    if seed is None:
        seed = params.get("seed")
    else:
        pass
    if seed is not None:
        generation["seed"] = int(seed)
    else:
        pass
    validate_generation_kwargs(generation)
    return generation


def normalize_inputs(inputs: object) -> tuple[str, list[dict[str, object]]]:
    if isinstance(inputs, str):
        return inputs, []
    else:
        pass
    if not isinstance(inputs, dict):
        return str(inputs) if inputs is not None else "", []
    else:
        pass
    references = inputs.get("references") or []
    if not isinstance(references, list):
        raise ValueError("Audar-TTS references must be a list")
    else:
        pass
    if any(not isinstance(reference, dict) for reference in references):
        raise ValueError("Audar-TTS references must be objects")
    else:
        pass
    return str(inputs.get("text", inputs.get("input", ""))), [
        dict(reference) for reference in references
    ]


def reference_from_value(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    else:
        pass
    if isinstance(value, str) and value.startswith("data:"):
        header, separator, data = value.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Audar-TTS ref_audio data URI must be base64 encoded")
        else:
            pass
        media_type = header.removeprefix("data:").split(";", 1)[0] or "audio/wav"
        return {"data": data, "media_type": media_type}
    else:
        pass
    return {"audio_path": str(value)}


def normalize_reference_audio(reference: Mapping[str, object]) -> dict[str, object]:
    if reference.get("audio_path") is not None:
        return {"audio_path": str(reference["audio_path"])}
    else:
        pass
    for key in ("ref_audio", "audio"):
        if reference.get(key) is not None:
            return reference_from_value(reference[key])
        else:
            pass
    if reference.get("bytes") is not None:
        return {"bytes": bytes(reference["bytes"])}
    else:
        pass
    data = reference.get("base64") or reference.get("data")
    if data is not None:
        return {
            "data": str(data),
            "media_type": str(reference.get("media_type") or "audio/wav"),
        }
    else:
        pass
    raise ValueError("Audar-TTS reference has no audio payload")


def validate_generation_kwargs(generation: Mapping[str, int | float]) -> None:
    if generation["max_new_tokens"] <= 0:
        raise ValueError("Audar-TTS max_new_tokens must be positive")
    else:
        pass
    if generation["temperature"] < 0:
        raise ValueError("Audar-TTS temperature must be non-negative")
    else:
        pass
    if not 0 < generation["top_p"] <= 1:
        raise ValueError("Audar-TTS top_p must be in (0, 1]")
    else:
        pass
    if generation["repetition_penalty"] <= 0:
        raise ValueError("Audar-TTS repetition_penalty must be positive")
    else:
        pass
