# SPDX-License-Identifier: Apache-2.0
"""Request mapping helpers for AuK."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from sglang_omni.models.auk import constants as C
from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
from sglang_omni.models.auk.payload_types import AuKState
from sglang_omni.proto import StagePayload
from sglang_omni.utils.audio import decode_audio_data_uri, load_audio
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference


@dataclass
class AuKPreprocessingContext:
    config: AuKRuntimeConfig
    default_seconds: float = C.DEFAULT_SECONDS
    max_seconds: float = C.MAX_SECONDS


_CONTEXT: AuKPreprocessingContext | None = None


def set_auk_preprocessing_context(context: AuKPreprocessingContext) -> None:
    global _CONTEXT
    _CONTEXT = context


def clear_auk_preprocessing_context() -> None:
    global _CONTEXT
    _CONTEXT = None


def _get_context() -> AuKPreprocessingContext:
    if _CONTEXT is None:
        raise RuntimeError("AuK preprocessing context is not initialized")
    return _CONTEXT


def _normalize_inputs(inputs: Any) -> tuple[str, list[dict[str, Any]], Any | None]:
    """Accept flat text, a dict payload, or a structured references list."""
    if isinstance(inputs, str):
        return inputs, [], None
    if not isinstance(inputs, dict):
        return (str(inputs) if inputs is not None else ""), [], None

    raw_references = inputs.get("references") or []
    if not isinstance(raw_references, list):
        raise ValueError("AuK references must be a list")
    if any(not isinstance(reference, dict) for reference in raw_references):
        raise ValueError("AuK references must be objects")
    if len(raw_references) > 1:
        raise ValueError("AuK accepts at most one reference audio clip")
    references = raw_references

    text = str(
        inputs.get("instruction") or inputs.get("text") or inputs.get("input") or ""
    )
    ref_audio = inputs.get("ref_audio") or inputs.get("audio") or inputs.get("file")
    return text, references, ref_audio


def _resolve_reference(
    references: list[dict[str, Any]], fallback: Any | None
) -> Any | None:
    if fallback is not None:
        return fallback
    if not references:
        return None
    reference = references[0]
    return (
        reference.get("audio_path")
        or reference.get("ref_audio")
        or reference.get("audio")
        or audio_data_uri_from_reference(reference)
    )


def _resolve_float(raw: Any, default: float | None) -> float | None:
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AuK expected a number, got {raw!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"AuK expected a finite number, got {raw!r}")
    return value


def _resolve_seed(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("AuK seed must be an integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AuK seed must be an integer, got {raw!r}") from exc


def _load_reference(source: Any, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    if isinstance(source, str):
        decoded = decode_audio_data_uri(source)
        if decoded is not None:
            source = decoded
        elif source.startswith(("http://", "https://")):
            import httpx

            response = httpx.get(source, timeout=5, follow_redirects=True)
            response.raise_for_status()
            source = response.content
        elif source.startswith("file://"):
            source = unquote(urlparse(source).path)
    vae_audio = load_audio(source, source_name="AuK", target_sample_rate=sample_rate)
    qwen_audio, _ = librosa.load(
        io.BytesIO(source) if isinstance(source, bytes) else source,
        sr=C.QWEN_AUDIO_SAMPLE_RATE,
        mono=True,
    )
    return np.asarray(vae_audio, dtype=np.float32).reshape(-1), qwen_audio


def build_auk_state(payload: StagePayload, config: AuKRuntimeConfig) -> AuKState:
    """Build the AuK state from an incoming request."""
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}
    stage_params = params.get("stage_params") or {}
    engine_params = stage_params.get("auk_engine") or {}
    for source in (tts_params, params, engine_params):
        for name in ("nfe", "cfg_strength", "sway_sampling_coef", "max_seconds"):
            if source.get(name) is not None:
                raise ValueError(f"AuK {name} is a server-level setting")

    text, references, inline_ref = _normalize_inputs(inputs)
    ref_source = _resolve_reference(references, inline_ref) or tts_params.get(
        "ref_audio"
    )
    is_speech = metadata.get("task") == "tts"
    if is_speech:
        if not text.strip():
            raise ValueError("AuK speech requires nonempty input text")
        if ref_source is not None:
            instruction = f'Say the following with the same voice: "{text}"'
        else:
            description = str(tts_params.get("instructions") or "").strip()
            description = description or C.DEFAULT_VOICE_DESCRIPTION
            instruction = (
                f'Generate speech based on the following description: "{description}". '
                f'The content to speak is: "{text}".'
            )
    else:
        instruction = str(params.get("instruction") or text).strip()
        if not instruction:
            raise ValueError("AuK requires a natural-language instruction")

    gen_seconds = _resolve_float(
        engine_params.get(
            "gen_seconds", tts_params.get("gen_seconds", params.get("gen_seconds"))
        ),
        None,
    )
    if gen_seconds is not None and gen_seconds <= 0:
        raise ValueError(f"AuK gen_seconds must be positive, got {gen_seconds}")

    clip_seconds = _get_context().max_seconds if _CONTEXT is not None else C.MAX_SECONDS

    ref_audio: np.ndarray | None = None
    qwen_audio: np.ndarray | None = None
    ref_seconds = 0.0
    if ref_source is not None:
        ref_audio, qwen_audio = _load_reference(ref_source, config.sample_rate)
        ref_seconds = ref_audio.shape[-1] / float(config.sample_rate)

    if gen_seconds is None and is_speech:
        ref_text = tts_params.get("ref_text")
        if references and inline_ref is None:
            ref_text = references[0].get("text") or ref_text
        if not ref_text or ref_seconds <= 0:
            raise ValueError(
                "AuK speech requires stage_params.auk_engine.gen_seconds "
                "or reference audio with ref_text"
            )
        gen_seconds = (
            ref_seconds * len(text.encode("utf-8")) / len(ref_text.encode("utf-8"))
        )

    if gen_seconds is None:
        gen_frames = (
            max(1, ref_audio.shape[-1] // config.downsample_rate)
            if ref_seconds > 0
            else config.seconds_to_frames(
                _get_context().default_seconds if _CONTEXT else C.DEFAULT_SECONDS
            )
        )
    else:
        gen_frames = config.seconds_to_frames(gen_seconds)
    gen_frames = min(gen_frames, config.seconds_to_frames(clip_seconds))

    return AuKState(
        sample_rate=config.sample_rate,
        instruction=instruction,
        ref_audio=ref_audio,
        qwen_audio=qwen_audio,
        ref_seconds=float(ref_seconds),
        gen_frames=gen_frames,
        seed=_resolve_seed(tts_params.get("seed", params.get("seed"))),
    )


def preprocess_auk_payload(payload: StagePayload) -> StagePayload:
    """Preprocessing-stage entry point: validate, load audio, size the output."""
    context = _get_context()
    state = build_auk_state(payload, context.config)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )
