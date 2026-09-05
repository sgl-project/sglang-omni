# SPDX-License-Identifier: Apache-2.0
"""GPU-free Breeze speech request validation."""

import math
import secrets
from dataclasses import dataclass, fields
from typing import Any

from .sampling import SamplingConfig


@dataclass(frozen=True)
class BreezeRequest:
    text: str
    instructions: str
    ref_audio: Any
    ref_text: str
    sampling: SamplingConfig


def _text(value: Any, name: str, *, required: bool = False) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"Breeze-TTS-2 {name} must be a non-empty string")
    return value


def parse_request(payload) -> BreezeRequest:
    request = payload.request
    inputs = request.inputs
    params = request.params or {}
    metadata = request.metadata or {}
    tts = metadata.get("tts_params", {})
    if isinstance(inputs, str):
        text, references = inputs, []
    elif isinstance(inputs, dict):
        text, references = inputs.get("text"), inputs.get("references") or []
    else:
        raise ValueError("Breeze-TTS-2 requires text input")
    text = _text(text, "input", required=True)
    instructions = _text(tts.get("instructions"), "instructions")
    if len(references) > 1:
        raise ValueError("Breeze-TTS-2 accepts at most one reference")
    ref_audio, ref_text = tts.get("ref_audio"), tts.get("ref_text")
    if references:
        from sglang_omni.utils.audio_payload import audio_data_uri_from_reference

        reference = references[0]
        if not isinstance(reference, dict) or reference.get("vq_codes") is not None:
            raise ValueError(
                "Breeze-TTS-2 requires reference audio, not precomputed codes"
            )
        ref_audio = (
            reference.get("audio_path")
            or reference.get("ref_audio")
            or reference.get("audio")
            or audio_data_uri_from_reference(reference)
        )
        ref_text = reference.get("text", ref_text)
    if ref_audio is not None:
        ref_text = _text(ref_text, "ref_text", required=True)
    elif ref_text:
        raise ValueError("Breeze-TTS-2 ref_text requires ref_audio")
    elif not instructions.strip():
        raise ValueError(
            "Breeze-TTS-2 requires a reference or voice-design instructions"
        )
    if tts.get("voice", "default") not in (None, "", "default") and not tts.get(
        "uploaded_voice_name"
    ):
        raise ValueError(
            "Breeze-TTS-2 has no built-in named voices; use instructions or a reference"
        )
    if tts.get("speed", 1.0) != 1.0:
        raise ValueError(
            "Breeze-TTS-2 only supports speed=1; direct pace using instructions"
        )
    if tts.get("language") not in (
        None,
        "auto",
        "Auto",
        "en",
        "zh",
        "English",
        "Chinese",
    ):
        raise ValueError("Breeze-TTS-2 supports English and Chinese only")
    if tts.get("task_type") not in (None, "Base", "VoiceDesign"):
        raise ValueError("Breeze-TTS-2 does not support this task_type")
    for name in (
        "x_vector_only_mode",
        "token_count",
        "duration_tokens",
        "suppress_bootstrap_silence",
    ):
        if tts.get(name) not in (None, False):
            raise ValueError(f"Breeze-TTS-2 does not support {name}")
    # HTTP injects Fish defaults. Only explicit fields may replace Breeze's
    # sampling defaults; internal callers without tts_params own their params.
    explicit = (
        set(tts.get("explicit_generation_params", ()))
        if "tts_params" in metadata
        else set(params)
    )
    values = {f.name: f.default for f in fields(SamplingConfig)}
    for name in (
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "max_new_tokens",
    ):
        if name in explicit and params.get(name) is not None:
            values[name] = params[name]
    values["cfg_scale"] = tts.get("cfg_scale", params.get("cfg_scale", 1.0))
    seed = tts.get("seed", params.get("seed"))
    values["seed"] = secrets.randbits(63) if seed is None else seed
    for name in ("top_k", "max_new_tokens", "seed"):
        if type(values[name]) is not int:
            raise ValueError(f"Breeze-TTS-2 {name} must be an integer")
    if not -1 <= values["top_k"] <= 2048:
        raise ValueError("Breeze-TTS-2 top_k must be -1, 0, or 1..2048")
    if not 1 <= values["max_new_tokens"] <= 750:
        raise ValueError("Breeze-TTS-2 max_new_tokens must be in 1..750")
    if not 0 <= values["seed"] < 2**64:
        raise ValueError("Breeze-TTS-2 seed must be an unsigned 64-bit integer")
    for name in ("temperature", "top_p", "repetition_penalty", "cfg_scale"):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Breeze-TTS-2 {name} must be finite")
    if values["temperature"] < 0 or values["cfg_scale"] < 0:
        raise ValueError("Breeze-TTS-2 temperature and cfg_scale must be nonnegative")
    if not 0 < values["top_p"] <= 1 or values["repetition_penalty"] <= 0:
        raise ValueError(
            "Breeze-TTS-2 requires 0 < top_p <= 1 and repetition_penalty > 0"
        )
    return BreezeRequest(
        text, instructions, ref_audio, ref_text or "", SamplingConfig(**values)
    )
