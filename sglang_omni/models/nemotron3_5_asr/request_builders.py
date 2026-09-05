# SPDX-License-Identifier: Apache-2.0
"""Request preparation for Nemotron 3.5 ASR."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto import StagePayload

NEMOTRON_ASR_SAMPLE_RATE = 16000


@dataclass(slots=True)
class Nemotron3_5ASRRequest:
    waveform: np.ndarray
    duration_s: float
    language: str
    stage_payload: StagePayload
    max_new_tokens: int | None = None
    started_at_s: float = 0.0


def normalize_nemotron_language(
    value: Any,
    prompt_dictionary: Mapping[str, int],
) -> str:
    if value is None:
        language = "auto"
    elif not isinstance(value, str):
        raise ValueError("language must be a string")
    else:
        language = value.strip() or "auto"

    # The checkpoint owns the language vocabulary. A case-insensitive lookup
    # accepts normal HTTP spellings such as en-us without maintaining a second
    # locale table in SGLang-Omni.
    canonical = {key.casefold(): key for key in prompt_dictionary}
    resolved = canonical.get(language.casefold())
    if resolved is None:
        raise ValueError(
            f"Unknown language={language!r}. Supported values: "
            f"{sorted(prompt_dictionary)}"
        )
    return resolved


def validate_nemotron_greedy_params(params: Mapping[str, Any]) -> int | None:
    try:
        temperature = float(params.get("temperature") or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a number") from exc
    if temperature != 0.0:
        raise ValueError(
            "Nemotron 3.5 ASR supports greedy RNN-T decoding only; "
            "temperature must be 0"
        )

    prompt = params.get("prompt")
    if prompt is not None and (not isinstance(prompt, str) or prompt.strip()):
        raise ValueError("Nemotron 3.5 ASR does not support a text prompt")

    task = str(params.get("task") or "transcribe").strip().lower()
    if task != "transcribe":
        raise ValueError("Nemotron 3.5 ASR supports transcription only")

    raw_max_new_tokens = params.get("max_new_tokens")
    if raw_max_new_tokens is None:
        return None
    if (
        isinstance(raw_max_new_tokens, bool)
        or not isinstance(raw_max_new_tokens, int)
        or raw_max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    return raw_max_new_tokens


# Keep the pre-Phase-3 private names importable for downstream tests and local
# integrations that used the original offline builder helpers.
_normalize_language = normalize_nemotron_language
_validate_greedy_params = validate_nemotron_greedy_params


def make_nemotron3_5_asr_request_builder(
    *, prompt_dictionary: Mapping[str, int]
) -> Callable[[StagePayload], Nemotron3_5ASRRequest]:
    """Build requests using the processor's authoritative locale mapping."""

    prompt_dictionary = dict(prompt_dictionary)
    if not prompt_dictionary:
        raise ValueError("Nemotron processor prompt_dictionary must not be empty")

    def request_builder(payload: StagePayload) -> Nemotron3_5ASRRequest:
        started_at_s = time.perf_counter()
        params = payload.request.params or {}
        max_new_tokens = validate_nemotron_greedy_params(params)
        language = normalize_nemotron_language(
            params.get("language"), prompt_dictionary
        )
        prepared = prepare_audio(
            payload,
            source_name="Nemotron 3.5 ASR",
            target_sample_rate=NEMOTRON_ASR_SAMPLE_RATE,
        )
        return Nemotron3_5ASRRequest(
            waveform=prepared.waveform,
            duration_s=prepared.duration_s,
            language=language,
            max_new_tokens=max_new_tokens,
            started_at_s=started_at_s,
            stage_payload=payload,
        )

    return request_builder


__all__ = [
    "NEMOTRON_ASR_SAMPLE_RATE",
    "Nemotron3_5ASRRequest",
    "make_nemotron3_5_asr_request_builder",
    "normalize_nemotron_language",
    "validate_nemotron_greedy_params",
]
