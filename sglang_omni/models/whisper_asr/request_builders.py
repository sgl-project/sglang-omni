# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Whisper ASR."""

from __future__ import annotations

import hashlib
import io
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

_WHISPER_SAMPLE_RATE = 16000
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
    language: str = "en"
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


def load_audio(source: Any) -> np.ndarray:
    import torchaudio

    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, bytearray):
        source = bytes(source)

    if isinstance(source, bytes):
        audio, sample_rate = torchaudio.load(io.BytesIO(source))
    elif isinstance(source, str):
        audio, sample_rate = torchaudio.load(source)
    else:
        raise ValueError(
            f"Unsupported Whisper ASR audio input: {type(source).__name__}"
        )

    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0).to(torch.float32)
    if sample_rate != _WHISPER_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, _WHISPER_SAMPLE_RATE)
    return audio.cpu().numpy()


def _resolve_language(value: Any) -> str:
    if value is None:
        return "english"
    language = str(value).strip().lower()
    if not language:
        return "english"
    return _LANGUAGE_ALIASES.get(language, language)


def _audio_fingerprint(audio: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(audio, dtype=np.float32)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()


def _audio_fingerprint_int(fingerprint: str) -> int:
    return int(fingerprint[:16], 16)


def _build_logit_bias(generation_config: GenerationConfig) -> dict[str, float] | None:
    suppress_tokens = getattr(generation_config, "suppress_tokens", None)
    if not suppress_tokens:
        return None
    return {str(int(token_id)): -1.0e9 for token_id in suppress_tokens if token_id >= 0}


def _build_prefix_tokens(tokenizer: Any, *, language: str, task: str) -> list[int]:
    tokenizer.set_prefix_tokens(
        language=language,
        task=task,
        predict_timestamps=False,
    )
    return list(tokenizer.prefix_tokens)


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
        audio = load_audio(_audio_source_from_payload(payload))
        audio_duration_s = float(len(audio) / _WHISPER_SAMPLE_RATE)
        fingerprint = _audio_fingerprint(audio)

        language = _resolve_language(params.get("language"))
        task = str(params.get("task") or "transcribe")
        prompt_token_ids = _build_prefix_tokens(
            tokenizer,
            language=language,
            task=task,
        )
        input_ids = [pad_token_id] * encoder_token_count + prompt_token_ids

        features = processor.feature_extractor(
            audio,
            sampling_rate=_WHISPER_SAMPLE_RATE,
            return_tensors="pt",
        ).input_features
        mm_inputs = MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    hash=_audio_fingerprint_int(fingerprint),
                    feature=features,
                )
            ],
            num_image_tokens=encoder_token_count,
        )

        temperature = float(params.get("temperature") or 0.0)
        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
            logit_bias=logit_bias,
        )
        sampling_params.normalize(tokenizer=None)

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
            language=language,
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
