# SPDX-License-Identifier: Apache-2.0
"""Public request mapping helpers for Ming-Omni-TTS."""

from __future__ import annotations

import math
from typing import Any

from sglang_omni.models.ming_tts.payload_types import (
    MING_TTS_DEFAULT_MAX_DECODE_STEPS,
    MingTTSState,
    store_ming_tts_state,
)
from sglang_omni.models.ming_tts.prompt_builder import build_ming_tts_prompt
from sglang_omni.models.ming_tts.tokenizer import MingTTSTokenizerBundle
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM

_REFERENCE_CONTRACT_ERROR = (
    "Ming-Omni-TTS currently supports only one local reference audio path "
    "with non-empty reference text"
)
_LOGITS_SAMPLING_FIELDS = ("do_sample", "top_p", "top_k", "repetition_penalty")
_REFERENCE_AUDIO_FIELDS = ("audio_path", "ref_audio", "audio", "prompt_wav_path")
_DIRECT_REFERENCE_AUDIO_FIELDS = ("audio_path", "ref_audio", "prompt_wav_path")
_REFERENCE_TEXT_FIELDS = ("ref_text", "prompt_text")
_UNSUPPORTED_REFERENCE_FIELDS = (
    "prompt_waveform",
    "prompt_latents",
    "prompt_latent",
    "speaker_embedding",
    "spk_emb",
    "use_spk_emb",
    "use_zero_spk_emb",
)
_DURATION_FIELDS = ("token_count", "duration_tokens", "tokens")
_DEFAULT_TASK_TYPES = {"", "default", "tts", "speech"}


def preprocess_ming_tts_payload(
    payload: StagePayload,
    *,
    tokenizer: MingTTSTokenizerBundle,
    context_length: int,
    max_decode_steps_cap: int | None = None,
) -> StagePayload:
    def optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text_value = str(value).strip()
        return text_value or None

    def has_non_empty_value(source: dict[str, Any], field: str) -> bool:
        if field not in source:
            return False
        value = source[field]
        if value is None or value is False:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, (list, tuple, dict, set)):
            return bool(value)
        return True

    def explicit_generation_fields(tts_source: dict[str, Any]) -> set[str]:
        raw = tts_source.get("explicit_generation_params")
        if isinstance(raw, (list, tuple, set)):
            return {str(field) for field in raw}
        return set()

    def first_present(*sources: dict[str, Any], names: tuple[str, ...]) -> Any | None:
        for source in sources:
            for name in names:
                if source.get(name) is not None:
                    return source[name]
        return None

    def resolve_int(name: str, value: Any, default: int | None = None) -> int:
        if value is None:
            if default is None:
                raise ValueError(f"Ming-Omni-TTS {name} must be an integer")
            return int(default)
        if isinstance(value, bool):
            raise ValueError(f"Ming-Omni-TTS {name} must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Ming-Omni-TTS {name} must be an integer")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Ming-Omni-TTS {name} must be an integer") from exc

    def resolve_float(name: str, value: Any, default: float) -> float:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            raise ValueError(f"Ming-Omni-TTS {name} must be a number")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Ming-Omni-TTS {name} must be a number") from exc
        # NaN slips through the range checks below (every comparison is
        # False) and non-finite values reach CFM as invalid latent scales.
        if not math.isfinite(result):
            raise ValueError(f"Ming-Omni-TTS {name} must be a finite number")
        return result

    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}
    stage_params = params.get("stage_params")
    tts_engine_params = {}
    if isinstance(stage_params, dict) and isinstance(
        stage_params.get("tts_engine"), dict
    ):
        tts_engine_params = stage_params["tts_engine"]

    inputs = payload.request.inputs or {}
    if isinstance(inputs, str):
        text, input_prompt, references = inputs, None, []
    elif isinstance(inputs, dict):
        raw_references = inputs.get("references") or []
        if not isinstance(raw_references, list):
            raise ValueError("Ming-Omni-TTS references must be a list")
        raw_text = inputs.get("text", inputs.get("input", ""))
        text = str(raw_text) if raw_text is not None else ""
        input_prompt = optional_text(inputs.get("prompt"))
        references = [
            dict(reference)
            for reference in raw_references
            if isinstance(reference, dict)
        ]
    else:
        text, input_prompt, references = (
            str(inputs) if inputs is not None else "",
            None,
            [],
        )
    text = text.strip()
    if not text:
        raise ValueError("Ming-Omni-TTS requires non-empty input text")

    if len(references) > 1:
        raise ValueError("Ming-Omni-TTS currently supports only one reference")

    input_dict = (
        payload.request.inputs if isinstance(payload.request.inputs, dict) else {}
    )
    reference = references[0] if references else {}
    ref_audio = first_present(
        reference,
        names=_REFERENCE_AUDIO_FIELDS,
    )
    if ref_audio is None:
        ref_audio = first_present(
            input_dict,
            tts_params,
            params,
            names=_DIRECT_REFERENCE_AUDIO_FIELDS,
        )
    if isinstance(ref_audio, str):
        ref_audio = ref_audio.strip() or None
    elif ref_audio is not None:
        raise ValueError(_REFERENCE_CONTRACT_ERROR)
    if reference and ref_audio is None:
        if reference.get("data") is not None:
            raise ValueError(
                "Ming-Omni-TTS reference audio must be a local file path; "
                "inline or URL reference audio is not supported"
            )
        raise ValueError(_REFERENCE_CONTRACT_ERROR)

    ref_text_value = reference.get("text") if reference else None
    if ref_text_value is None:
        ref_text_value = first_present(
            input_dict,
            tts_params,
            params,
            names=_REFERENCE_TEXT_FIELDS,
        )
    ref_text = optional_text(ref_text_value)
    if ref_audio is not None and ref_text is None:
        raise ValueError(_REFERENCE_CONTRACT_ERROR)
    if ref_audio is None and ref_text is not None:
        raise ValueError("Ming-Omni-TTS reference text requires reference audio")

    if isinstance(payload.request.inputs, dict):
        for field in _UNSUPPORTED_REFERENCE_FIELDS:
            if has_non_empty_value(payload.request.inputs, field):
                raise ValueError(
                    "Ming-Omni-TTS speaker embedding and prompt latent inputs "
                    "must be produced by the reference_encode stage"
                )

    for source in (tts_params, params):
        for field in _UNSUPPORTED_REFERENCE_FIELDS:
            if has_non_empty_value(source, field):
                raise ValueError(
                    "Ming-Omni-TTS speaker embedding and prompt latent inputs "
                    "must be produced by the reference_encode stage"
                )
        for field in _DURATION_FIELDS:
            if source.get(field) is not None:
                raise ValueError(
                    f"Ming-Omni-TTS field {field!r} is currently unsupported"
                )

    voice = tts_params.get("voice", params.get("voice"))
    if voice is not None and str(voice).strip().lower() not in ("", "default"):
        raise ValueError(
            "Ming-Omni-TTS currently supports only the default voice; " f"got {voice!r}"
        )

    language = tts_params.get("language", params.get("language"))
    if language is not None and str(language).strip().lower() not in ("", "auto"):
        raise ValueError(
            "Ming-Omni-TTS language selection is currently unsupported; "
            f"got {language!r}"
        )

    instructions = (
        tts_params.get("instructions")
        or tts_params.get("instruct")
        or params.get("instructions")
        or params.get("instruct")
    )
    if optional_text(instructions):
        raise ValueError("Ming-Omni-TTS instructions are currently unsupported")

    task_type = tts_params.get("task_type", params.get("task_type"))
    if (
        task_type is not None
        and str(task_type).strip().lower() not in _DEFAULT_TASK_TYPES
    ):
        raise ValueError(
            "Ming-Omni-TTS currently supports only default text-to-speech; "
            f"got task_type={task_type!r}"
        )

    speed = tts_params.get("speed", params.get("speed"))
    if speed is not None and float(speed) != 1.0:
        raise ValueError(
            "Ming-Omni-TTS speed control is currently unsupported; " f"got {speed!r}"
        )

    if params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) is not None:
        raise ValueError(
            "Ming-Omni-TTS currently generates final waveform only; "
            f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM!r} is currently unsupported"
        )

    explicit_fields = explicit_generation_fields(tts_params)
    for field in _LOGITS_SAMPLING_FIELDS:
        if has_non_empty_value(tts_params, field) or has_non_empty_value(
            tts_engine_params, field
        ):
            raise ValueError(
                f"Ming-Omni-TTS does not use logits sampling field {field!r}"
            )
        if field in explicit_fields and params.get(field) is not None:
            raise ValueError(
                f"Ming-Omni-TTS does not use logits sampling field {field!r}"
            )

    if (
        first_present(tts_engine_params, tts_params, params, names=("seed",))
        is not None
    ):
        raise ValueError(
            "Ming-Omni-TTS seed is currently unsupported because "
            "FlowLoss sampling is unseeded"
        )

    max_steps_value = first_present(
        tts_engine_params,
        tts_params,
        params,
        names=("max_decode_steps", "ming_max_decode_steps"),
    )
    if max_steps_value is None:
        max_steps_value = params.get("max_new_tokens")
    max_decode_steps = resolve_int(
        "max_decode_steps",
        max_steps_value,
        default=MING_TTS_DEFAULT_MAX_DECODE_STEPS,
    )
    if max_decode_steps <= 0:
        raise ValueError(
            "Ming-Omni-TTS max_decode_steps must be > 0, " f"got {max_decode_steps}"
        )
    if max_decode_steps_cap is not None and max_decode_steps > int(
        max_decode_steps_cap
    ):
        raise ValueError(
            "Ming-Omni-TTS max_decode_steps exceeds serving cap: "
            f"{max_decode_steps} > {int(max_decode_steps_cap)}"
        )

    temperature_value = first_present(
        tts_engine_params,
        names=("temperature", "ming_temperature"),
    )
    if temperature_value is None:
        temperature_value = first_present(
            tts_params,
            names=("temperature", "ming_temperature"),
        )
    if temperature_value is None and params.get("ming_temperature") is not None:
        temperature_value = params["ming_temperature"]
    if (
        temperature_value is None
        and "temperature" in explicit_fields
        and params.get("temperature") is not None
    ):
        temperature_value = params["temperature"]

    prompt = (
        input_prompt
        or optional_text(tts_params.get("prompt"))
        or optional_text(params.get("prompt"))
    )
    state = MingTTSState(
        text=text,
        prompt=prompt,
        voice=None,
        language=None,
        ref_audio=ref_audio,
        ref_text=ref_text,
        max_decode_steps=max_decode_steps,
        cfg=resolve_float(
            "cfg",
            first_present(tts_engine_params, tts_params, params, names=("cfg",)),
            default=2.0,
        ),
        sigma=resolve_float(
            "sigma",
            first_present(tts_engine_params, tts_params, params, names=("sigma",)),
            default=0.25,
        ),
        temperature=resolve_float("temperature", temperature_value, default=0.0),
    )
    if state.cfg < 1e-5 or state.cfg == 1.0:
        raise ValueError(
            "Ming-Omni-TTS currently supports only guided-branch cfg values; "
            "cfg must be >= 1e-5 and not equal to 1.0"
        )
    if state.sigma < 0:
        raise ValueError(f"Ming-Omni-TTS sigma must be >= 0, got {state.sigma}")
    if state.temperature < 0:
        raise ValueError(
            f"Ming-Omni-TTS temperature must be >= 0, got {state.temperature}"
        )

    if state.ref_audio is None:
        plan = build_ming_tts_prompt(state, tokenizer)
        if plan.prompt_tokens + state.max_decode_steps > int(context_length):
            raise ValueError(
                "Ming-Omni-TTS request exceeds context length: "
                f"prompt_tokens={plan.prompt_tokens}, "
                f"max_decode_steps={state.max_decode_steps}, "
                f"context_length={context_length}"
            )

        state.prompt = plan.effective_prompt
        state.input_ids = plan.input_ids
        state.prompt_tokens = plan.prompt_tokens
        state.spk_token_positions = plan.spk_token_positions
        state.spk_injection_positions = plan.spk_injection_positions
        state.audio_token_position = plan.audio_token_position
        state.prompt_latent_start_position = plan.prompt_latent_start_position
        state.prompt_latent_token_count = plan.prompt_latent_token_count
    return store_ming_tts_state(payload, state)


__all__ = [
    "preprocess_ming_tts_payload",
]
