# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Whisper ASR."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import GenerationConfig

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int, load_audio

_WHISPER_SAMPLE_RATE = 16000
_MAX_AUDIO_DURATION_S = 30.0
_SUPPORTED_TASKS = frozenset({"transcribe", "translate"})
_PREFIX_TOKEN_LOCK = threading.Lock()
_LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
}


@dataclass
class WhisperASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str | None = "english"
    task: str = "transcribe"
    engine_start_s: float = 0.0


def _audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file"):
            value = inputs.get(key)
            if value is not None:
                return value
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


def _load_audio(source: Any) -> np.ndarray:
    return load_audio(
        source,
        source_name="Whisper ASR",
        target_sample_rate=_WHISPER_SAMPLE_RATE,
    )


def _resolve_task(value: Any) -> str:
    task = str(value or "transcribe").strip().lower()
    if task not in _SUPPORTED_TASKS:
        raise ValueError(
            f"Whisper task must be one of {sorted(_SUPPORTED_TASKS)}, got {task!r}"
        )
    return task


def _resolve_language(value: Any, *, task: str) -> str:
    language = "" if value is None else str(value).strip().lower()
    if not language:
        if task == "translate":
            raise ValueError(
                "Whisper translation requires an explicit source language until "
                "automatic language detection is implemented"
            )
        return "english"
    return _LANGUAGE_ALIASES.get(language, language)


def _build_logit_bias(generation_config: GenerationConfig) -> dict[str, float] | None:
    try:
        suppress_tokens = generation_config.suppress_tokens
    except AttributeError:
        suppress_tokens = None
    if not suppress_tokens:
        return None
    return {str(int(token_id)): -1.0e9 for token_id in suppress_tokens if token_id >= 0}


def _build_prefix_tokens(
    tokenizer: Any, *, language: str | None, task: str
) -> list[int]:
    """Build a request-local prefix without leaking tokenizer state.

    Whisper tokenizers store language/task as mutable attributes. Request
    builders run concurrently, so serialize the short prefix construction and
    restore the prior state before returning.
    """

    missing = object()
    with _PREFIX_TOKEN_LOCK:
        try:
            previous_language = tokenizer.language
        except AttributeError:
            previous_language = missing
        try:
            previous_task = tokenizer.task
        except AttributeError:
            previous_task = missing
        try:
            previous_predict_timestamps = tokenizer.predict_timestamps
        except AttributeError:
            previous_predict_timestamps = missing

        tokenizer.language = language
        tokenizer.task = task
        tokenizer.predict_timestamps = False
        try:
            return list(tokenizer.prefix_tokens)
        finally:
            if previous_language is missing:
                del tokenizer.language
            else:
                tokenizer.language = previous_language
            if previous_task is missing:
                del tokenizer.task
            else:
                tokenizer.task = previous_task
            if previous_predict_timestamps is missing:
                del tokenizer.predict_timestamps
            else:
                tokenizer.predict_timestamps = previous_predict_timestamps


def _request_token_budget(params: dict[str, Any], max_new_tokens: int) -> int:
    explicit = params.get("max_new_tokens")
    if explicit is None:
        return max_new_tokens
    if isinstance(explicit, bool):
        raise ValueError("max_new_tokens must be an integer")
    if isinstance(explicit, int):
        requested = explicit
    elif isinstance(explicit, str):
        try:
            requested = int(explicit)
        except ValueError as exc:
            raise ValueError("max_new_tokens must be an integer") from exc
    else:
        raise ValueError("max_new_tokens must be an integer")
    if requested < 1 or requested > max_new_tokens:
        raise ValueError(
            f"max_new_tokens must be between 1 and {max_new_tokens}, got {requested}"
        )
    return requested


def make_whisper_scheduler_adapters(
    *,
    processor: Any,
    tokenizer: Any,
    generation_config: GenerationConfig,
    encoder_token_count: int,
    max_new_tokens: int,
) -> tuple[
    Callable[[StagePayload], WhisperASRRequestData], Callable[[Any], StagePayload]
]:
    logit_bias = _build_logit_bias(generation_config)
    eos_token_id = int(tokenizer.eos_token_id)
    pad_token_id = int(tokenizer.pad_token_id or eos_token_id)
    vocab_size = int(tokenizer.vocab_size)

    def request_builder(payload: StagePayload) -> WhisperASRRequestData:
        params = payload.request.params or {}
        task = _resolve_task(params.get("task"))
        language = _resolve_language(params.get("language"), task=task)
        request_max_new_tokens = _request_token_budget(params, max_new_tokens)
        temperature = float(params.get("temperature") or 0.0)
        prompt_token_ids = _build_prefix_tokens(
            tokenizer,
            language=language,
            task=task,
        )
        input_ids = [pad_token_id] * encoder_token_count + prompt_token_ids
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
            logit_bias=logit_bias,
        )
        sampling_params.normalize(tokenizer=None)

        try:
            audio = _load_audio(_audio_source_from_payload(payload))
        except Exception as exc:
            raise ValueError(
                "Whisper ASR could not decode the uploaded audio; provide a valid "
                "audio file."
            ) from exc
        audio_duration_s = float(len(audio) / _WHISPER_SAMPLE_RATE)
        if audio_duration_s > _MAX_AUDIO_DURATION_S:
            raise ValueError(
                "Whisper ASR accepts audio up to 30.0 seconds; split longer audio "
                "before inference."
            )
        fingerprint = audio_fingerprint(audio)
        features = processor.feature_extractor(
            audio,
            sampling_rate=_WHISPER_SAMPLE_RATE,
            return_tensors="pt",
        ).input_features
        mm_inputs = MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    hash=audio_fingerprint_int(fingerprint),
                    feature=features,
                )
            ],
            num_image_tokens=encoder_token_count,
        )

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            extra_key=fingerprint,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None

        return WhisperASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=audio_duration_s,
            language="english" if task == "translate" else language,
            task=task,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: WhisperASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language,
                "task": data.task,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "usage": {"engine_time_s": engine_time_s},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


__all__ = [
    "WhisperASRRequestData",
    "load_audio",
    "make_whisper_scheduler_adapters",
]
