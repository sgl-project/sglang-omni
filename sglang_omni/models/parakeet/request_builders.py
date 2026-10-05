# SPDX-License-Identifier: Apache-2.0
"""StagePayload adapters for Parakeet ASR."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto import StagePayload
from sglang_omni.utils.audio import AudioDecodeError


@dataclass(slots=True)
class ParakeetASRRequest:
    waveform: np.ndarray
    duration_s: float
    language: str | None
    stage_payload: StagePayload
    started_at_s: float = 0.0


def validate_parakeet_params(params: Mapping[str, object]) -> str | None:
    """Reject what greedy Parakeet decoding cannot honor; return the language.

    Parakeet has no prompt, no translation task, and no length budget, and
    multilingual checkpoints detect the language themselves, so ``language``
    is echoed back but does not steer decoding.
    """
    try:
        temperature = float(params.get("temperature") or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Parakeet ASR supports a numeric temperature only") from exc
    if temperature != 0.0:
        raise ValueError(
            "Parakeet ASR supports greedy decoding only; temperature must be 0"
        )
    else:
        pass

    prompt = params.get("prompt")
    if prompt is not None and (not isinstance(prompt, str) or prompt.strip()):
        raise ValueError("Parakeet ASR does not support a text prompt")
    else:
        pass

    task = str(params.get("task") or "transcribe").strip().lower()
    if task != "transcribe":
        raise ValueError("Parakeet ASR supports transcription only")
    else:
        pass

    if params.get("max_new_tokens") is not None:
        raise ValueError("Parakeet ASR does not support max_new_tokens")
    else:
        pass

    language = params.get("language")
    if language is None:
        return None
    elif not isinstance(language, str):
        raise ValueError("Parakeet ASR supports a string language only")
    else:
        return language.strip() or None


def make_parakeet_request_builder(
    *, sample_rate: int
) -> Callable[[StagePayload], ParakeetASRRequest]:
    def request_builder(payload: StagePayload) -> ParakeetASRRequest:
        started_at_s = time.perf_counter()
        language = validate_parakeet_params(payload.request.params or {})
        try:
            prepared = prepare_audio(
                payload,
                source_name="Parakeet ASR",
                target_sample_rate=sample_rate,
            )
        except AudioDecodeError as exc:
            raise ValueError(
                "Parakeet ASR could not decode the uploaded audio; provide a "
                "valid audio file."
            ) from exc
        return ParakeetASRRequest(
            waveform=prepared.waveform,
            duration_s=prepared.duration_s,
            language=language,
            stage_payload=payload,
            started_at_s=started_at_s,
        )

    return request_builder


def build_parakeet_result(
    request: ParakeetASRRequest, *, text: str, model_latency_s: float
) -> StagePayload:
    payload = request.stage_payload
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data={
            "text": text,
            "language": request.language,
            "duration_s": request.duration_s,
            "asr_latency_s": time.perf_counter() - request.started_at_s,
            "model_latency_s": model_latency_s,
            "finish_reason": "stop",
            "usage": {"engine_time_s": model_latency_s},
            "modality": "text",
        },
    )


__all__ = [
    "ParakeetASRRequest",
    "build_parakeet_result",
    "make_parakeet_request_builder",
    "validate_parakeet_params",
]
