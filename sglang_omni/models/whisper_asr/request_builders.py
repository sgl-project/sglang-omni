# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Whisper ASR."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import GenerationConfig

from sglang_omni.models.whisper_asr.config import WHISPER_MAX_INPUT_SECONDS
from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

_WHISPER_SAMPLE_RATE = 16000

_MAX_ENGINE_CLIP_S = float(WHISPER_MAX_INPUT_SECONDS)
_MAX_ENGINE_CLIP_MESSAGE = (
    f"Whisper ASR accepts audio up to {WHISPER_MAX_INPUT_SECONDS} seconds per "
    "request (the model's mel window); send longer audio through "
    "/v1/audio/transcriptions, which splits it into chunks"
)
# note (jiannan-17): Previous context = 1 start-of-prev token + up to 223 prompt tokens.
MAX_PREV_CONTEXT_TOKENS = 224
# note (jiannan-17): Standard Whisper decoder context is 448 positions.
_DEFAULT_DECODER_CONTEXT_LEN = 448
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
    detect_language: bool = False
    engine_start_s: float = 0.0


def _resolve_language(value: Any) -> str:
    if value is None:
        return "english"
    language = str(value).strip().lower()
    if not language:
        return "english"
    return _LANGUAGE_ALIASES.get(language, language)


def _build_logit_bias(generation_config: GenerationConfig) -> dict[str, float] | None:
    suppress_tokens = generation_config.suppress_tokens
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


def _decoder_token_budgets(
    *,
    decoder_context_len: int,
    prefix_len: int,
    requested_max_new_tokens: int,
) -> tuple[int, int]:
    """Split the decoder position budget between generation and prev context.

    Allocation order: the mandatory task prefix first, then generation
    (clamped so it can never overflow the learned position table), and only
    the remaining positions go to previous-text context.
    """
    max_new_tokens = min(requested_max_new_tokens, decoder_context_len - prefix_len)
    max_prev_tokens = min(
        MAX_PREV_CONTEXT_TOKENS,
        decoder_context_len - prefix_len - max_new_tokens,
    )
    return max_new_tokens, max_prev_tokens


def _build_prev_context_tokens(
    tokenizer: Any, prompt: Any, *, max_prev_tokens: int
) -> list[int]:
    """Map the OpenAI ``prompt`` field to Whisper prev-context tokens."""
    if max_prev_tokens < 2 or prompt is None:
        return []
    text = str(prompt).strip()
    if not text:
        return []
    sot_prev_id, *text_ids = tokenizer.get_prompt_ids(text, return_tensors=None)
    # note (jiannan-17): Recent text is the most useful continuation context, so
    # truncate from the front while preserving the required <|startofprev|> marker.
    return [sot_prev_id] + text_ids[-(max_prev_tokens - 1) :]


def make_whisper_scheduler_adapters(
    *,
    processor: Any,
    tokenizer: Any,
    generation_config: GenerationConfig,
    encoder_token_count: int,
    max_new_tokens: int,
    decoder_context_len: int | None = None,
    audio_encoder_service: Any | None = None,
) -> tuple[
    Callable[[StagePayload], WhisperASRRequestData], Callable[[Any], StagePayload]
]:
    logit_bias = _build_logit_bias(generation_config)
    # note (Dayuxiaoshui): set_prefix_tokens mutates shared tokenizer state
    # across request-build workers.
    tokenizer_lock = Lock()
    eos_token_id = int(tokenizer.eos_token_id)
    pad_token_id = int(tokenizer.pad_token_id or eos_token_id)
    # note (Junnan Li): language identification samples language tokens, which
    # sit above the base vocabulary, so the sampling vocab must cover them.
    vocab_size = len(tokenizer)
    sot_token_id = int(tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))
    # note (Junnan Li): a uniform positive bias keeps the relative order of the
    # language tokens while guaranteeing one of them wins the detection step.
    lang_to_id = getattr(generation_config, "lang_to_id", None) or {}
    language_token_bias = {
        str(int(token_id)): 100.0 for token_id in lang_to_id.values()
    }
    id_to_language = {
        int(token_id): token.strip("<|>") for token, token_id in lang_to_id.items()
    }
    # note (jiannan-17): Prefer the decoder limit passed by the caller. Fall back
    # to generation_config.max_length, then to Whisper's default 448 positions.
    decoder_context_len = int(
        decoder_context_len
        or generation_config.max_length
        or _DEFAULT_DECODER_CONTEXT_LEN
    )

    def request_builder(payload: StagePayload) -> WhisperASRRequestData:
        params = payload.request.params or {}
        prepared = prepare_audio(
            payload,
            source_name="Whisper ASR",
            target_sample_rate=_WHISPER_SAMPLE_RATE,
            max_duration_s=_MAX_ENGINE_CLIP_S,
            max_duration_message=_MAX_ENGINE_CLIP_MESSAGE,
        )
        audio = prepared.waveform
        audio_duration_s = prepared.duration_s
        fingerprint = prepared.fingerprint

        language = _resolve_language(params.get("language"))
        task = str(params.get("task") or "transcribe")
        detect_language = bool(params.get("detect_language"))
        if detect_language:
            prompt_token_ids = [sot_token_id]
            request_max_new_tokens = 1
        else:
            with tokenizer_lock:
                prefix_token_ids = _build_prefix_tokens(
                    tokenizer,
                    language=language,
                    task=task,
                )
                request_max_new_tokens, max_prev_tokens = _decoder_token_budgets(
                    decoder_context_len=decoder_context_len,
                    prefix_len=len(prefix_token_ids),
                    requested_max_new_tokens=int(
                        params.get("max_new_tokens") or max_new_tokens
                    ),
                )
                prev_context_ids = _build_prev_context_tokens(
                    tokenizer, params.get("prompt"), max_prev_tokens=max_prev_tokens
                )
            prompt_token_ids = prev_context_ids + prefix_token_ids
        # note (jiannan-17): Keep the invariant at the assembly boundary so future
        # changes fail before an out-of-range decoder position reaches the GPU.
        if len(prompt_token_ids) + request_max_new_tokens > decoder_context_len:
            raise ValueError(
                "Whisper decoder budget exceeded: "
                f"{len(prompt_token_ids)} input tokens + "
                f"{request_max_new_tokens} max_new_tokens > {decoder_context_len}"
            )
        input_ids = [pad_token_id] * encoder_token_count + prompt_token_ids

        features = None
        cached_embedding = None
        if audio_encoder_service is not None:
            cached_embedding = audio_encoder_service.lookup_cached_embedding(
                fingerprint, encoder_token_count
            )
        if cached_embedding is None:
            features = processor.feature_extractor(
                audio,
                sampling_rate=_WHISPER_SAMPLE_RATE,
                return_tensors="pt",
            ).input_features

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=prepared.fingerprint_int,
            feature=features,
            model_specific_data={
                "audio_fingerprint": fingerprint,
                "num_audio_tokens": encoder_token_count,
            },
        )
        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=encoder_token_count,
        )
        if audio_encoder_service is not None:
            if cached_embedding is None:
                audio_encoder_service.encode_item(audio_item)
            else:
                audio_encoder_service.attach_embedding(audio_item, cached_embedding)

        temperature = float(params.get("temperature") or 0.0)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
            logit_bias=language_token_bias if detect_language else logit_bias,
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
            detect_language=detect_language,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: WhisperASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        if data.detect_language:
            text = id_to_language.get(output_ids[0], "") if output_ids else ""
        else:
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
    "MAX_PREV_CONTEXT_TOKENS",
    "WhisperASRRequestData",
    "make_whisper_scheduler_adapters",
]
