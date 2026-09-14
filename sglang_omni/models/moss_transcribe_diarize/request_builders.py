# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import logging
import math
import re
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

from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.token_text_streaming import (
    make_token_text_stream_output_builder,
)

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_AUDIO_PAD = "<|audio_pad|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END = "<|audio_end|>"
_SPECIAL_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
_WHISPER_ENCODER_STRIDE = 2
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50

# note (db-ol): dense multi speaker meetings decode to about 4.5 output
# tokens per audio second including time markers, so 10 leaves roughly 2x
# headroom without approaching the input token cost of the same audio.
_OUTPUT_TOKENS_PER_AUDIO_SECOND = 10
# Floor for the duration-scaled budget. Short, dense multi-speaker clips need
# room for transcript text plus repeated timestamp and speaker-label framing;
# a 128-token floor truncated valid outputs in the movies800times ASR set.
# This remains far below the legacy 5120-token default that allowed short
# non-speech repetition loops to run for thousands of tokens.
_MIN_SCALED_OUTPUT_TOKENS = 512
# A zero-duration input has no legitimate transcript to preserve, so retain a
# tighter fallback than the floor used for short, dense speech.
_EMPTY_AUDIO_OUTPUT_TOKENS = 128
# Note (yichi): MOSS-Transcribe-Diarize is an audio LLM: a Qwen3 text decoder
# over Whisper audio embeddings, trained on a fixed transcribe+diarize
# instruction with the timestamped/speaker-labelled transcript as the target
# output. This is the default instruction used when a request supplies no prompt.
DEFAULT_TRANSCRIBE_DIARIZE_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


@dataclass
class MossTranscribeDiarizeRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "auto"
    engine_start_s: float = 0.0
    # note (db-ol): the scheduler clamps max_new_tokens to the remaining
    # context, required once the duration scaled default can exceed it.
    enforce_request_limits: bool = True


def _only_audio(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "MOSS-Transcribe-Diarize supports exactly one audio per request, "
                f"got {len(value)} items"
            )
        return value[0]
    return value


def _audio_source_from_payload(payload: StagePayload) -> Any:
    """Extended source resolver: MOSS accepts more sources than the shared
    default (``audio_data``, single-item ``audios`` lists, metadata fallbacks,
    and ``{"data"|"path"|"url": ...}`` dict entries)."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file", "audio_data"):
            value = inputs.get(key)
            if value is not None:
                return _unwrap_source_dict(value)
        value = inputs.get("audios")
        if value is not None:
            return _unwrap_source_dict(_only_audio(value))
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return _unwrap_source_dict(value)

    metadata = payload.request.metadata or {}
    value = metadata.get("audios")
    if value is not None:
        return _unwrap_source_dict(_only_audio(value))
    for key in ("audio_data", "audio"):
        value = metadata.get(key)
        if value is not None:
            return _unwrap_source_dict(value)
    return _unwrap_source_dict(inputs)


def _has_metadata_audio_source(payload: StagePayload) -> bool:
    metadata = payload.request.metadata or {}
    return any(
        metadata.get(key) is not None for key in ("audios", "audio_data", "audio")
    )


def _unwrap_source_dict(source: Any) -> Any:
    if isinstance(source, dict):
        if source.get("data") is not None:
            return source["data"]
        if source.get("path") is not None:
            return source["path"]
        if source.get("url") is not None:
            return source["url"]
    return source


def _explicit_generation_fields(metadata: dict[str, Any]) -> set[str]:
    """Sampling fields the caller set explicitly (see EXPLICIT_GENERATION_PARAMS_KEY).

    Anything not listed here resolves to the model's own default, so a client
    layer that fills every SamplingParams field with a placeholder no longer
    shadows the MOSS defaults.
    """
    fields = metadata.get(EXPLICIT_GENERATION_PARAMS_KEY)
    if isinstance(fields, (list, tuple)):
        return {str(field) for field in fields}
    return set()


def _sampling_param(
    params: dict[str, Any],
    explicit_fields: set[str],
    field: str,
    default: Any,
    cast: Callable[[Any], Any],
) -> Any:
    if field not in explicit_fields:
        return default
    value = params.get(field)
    return default if value is None else cast(value)


def _decode_token_ids(
    tokenizer: Any, token_ids: list[int], skip_special_tokens: bool
) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def postprocess_moss_transcribe_diarize_text(text: str) -> str:
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def _render_prompt(processor: Any, input_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": ""},
                {"type": "text", "text": input_text},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _prompt_from_payload(
    payload: StagePayload,
    processor: Any,
    *,
    default_prompt: str | None = None,
) -> str:
    inputs = payload.request.inputs
    params = payload.request.params or {}

    if isinstance(inputs, dict) and "messages" in inputs:
        return processor.apply_chat_template(
            inputs["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

    input_text: Any = params.get("prompt")
    if isinstance(inputs, dict):
        input_text = inputs.get("prompt", inputs.get("text", input_text))
    elif isinstance(inputs, str) and _has_metadata_audio_source(payload):
        input_text = inputs

    if isinstance(input_text, list):
        input_text = processor.tokenizer.decode(input_text)
    input_text = input_text or ""
    if _AUDIO_PAD in input_text:
        return str(input_text)

    if not str(input_text).strip():
        if default_prompt is not None:
            return default_prompt
        input_text = DEFAULT_TRANSCRIBE_DIARIZE_PROMPT

    return _render_prompt(processor, str(input_text))


def _contiguous_offsets(input_ids: list[int], token_id: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(input_ids):
        if value == token_id:
            if start is None:
                start = idx
            continue
        if start is not None:
            offsets.append((start, idx - 1))
            start = None
    if start is not None:
        offsets.append((start, len(input_ids) - 1))
    return offsets


def _prompt_token_parts(
    prompt: str,
    tokenizer: Any,
    audio_token: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    audio_token_count = prompt.count(audio_token)
    if audio_token_count != 1:
        raise ValueError(
            f"Expected exactly one {audio_token!r} token per text sample, "
            f"got {audio_token_count}."
        )
    before_audio, after_audio = prompt.split(audio_token, maxsplit=1)
    return (
        tuple(tokenizer.encode(before_audio, add_special_tokens=False)),
        tuple(tokenizer.encode(after_audio, add_special_tokens=False)),
    )


def _audio_feature_lengths_from_waveform(
    processor: Any,
    num_samples: int,
) -> torch.Tensor:
    """Derive the processor's per-chunk token lengths without extracting mel."""
    feature_extractor = processor.feature_extractor
    chunk_samples = int(feature_extractor.n_samples)
    stride = (
        int(feature_extractor.hop_length)
        * _WHISPER_ENCODER_STRIDE
        * int(processor.audio_merge_size)
    )
    if chunk_samples <= 0 or stride <= 0:
        raise ValueError("MOSS-Transcribe-Diarize processor has invalid audio strides")
    return torch.tensor(
        [
            (min(chunk_samples, num_samples - start) - 1) // stride + 1
            for start in range(0, num_samples, chunk_samples)
        ],
        dtype=torch.long,
    )


def _extract_audio_features(
    processor: Any,
    audio: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    feature_extractor = processor.feature_extractor
    n_samples = int(feature_extractor.n_samples)
    audio_feature_lengths = _audio_feature_lengths_from_waveform(
        processor,
        int(audio.shape[0]),
    )
    chunks: list[np.ndarray] = []
    for start in range(0, audio.shape[0], n_samples):
        chunk = audio[start : start + n_samples]
        if chunk.shape[0] < n_samples:
            chunk = np.pad(chunk, (0, n_samples - chunk.shape[0]))
        chunks.append(chunk)

    input_features = feature_extractor(
        chunks,
        sampling_rate=int(feature_extractor.sampling_rate),
        padding="max_length",
        return_tensors="pt",
    )["input_features"]
    audio_feature_lengths = audio_feature_lengths.to(input_features.device)
    return (
        input_features,
        audio_feature_lengths,
        torch.zeros_like(audio_feature_lengths),
        int(audio_feature_lengths.sum().item()),
    )


def make_moss_transcribe_diarize_scheduler_adapters(
    processor: Any,
    tokenizer: Any,
    max_new_tokens: int,
    context_length: int,
    duration_scaled_default: bool = True,
    audio_encoder_service: Any | None = None,
) -> tuple[
    Callable[[StagePayload], MossTranscribeDiarizeRequestData],
    Callable[[Any], StagePayload],
]:
    audio_token_id = int(
        getattr(processor, "audio_token_id", None)
        or tokenizer.convert_tokens_to_ids(_AUDIO_PAD)
    )
    audio_start_id = int(tokenizer.convert_tokens_to_ids(_AUDIO_START))
    audio_end_id = int(tokenizer.convert_tokens_to_ids(_AUDIO_END))
    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = int(tokenizer.vocab_size)
    audio_token = str(processor.audio_token)
    default_prompt = _render_prompt(processor, DEFAULT_TRANSCRIBE_DIARIZE_PROMPT)
    default_prompt_parts = _prompt_token_parts(
        default_prompt,
        tokenizer,
        audio_token,
    )

    def request_builder(payload: StagePayload) -> MossTranscribeDiarizeRequestData:
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        explicit_fields = _explicit_generation_fields(metadata)
        prepared = prepare_audio(
            payload,
            source_name="MOSS-Transcribe-Diarize",
            target_sample_rate=_SAMPLE_RATE,
            source_resolver=_audio_source_from_payload,
        )
        audio = prepared.waveform
        audio_duration_s = prepared.duration_s
        fingerprint = prepared.fingerprint
        prompt = _prompt_from_payload(
            payload,
            processor,
            default_prompt=default_prompt,
        )

        # note (db-ol): cap the processor limit at the model context. The
        # processor rejects sequences past max_length rather than truncating,
        # and a request max_length above the context would skip that early
        # rejection and defer the failure to scheduler admission.
        max_length = min(
            int(params.get("max_length") or context_length), context_length
        )
        cached_embedding = None
        if audio_encoder_service is not None:
            audio_feature_lengths = _audio_feature_lengths_from_waveform(
                processor,
                len(audio),
            )
            cached_embedding = audio_encoder_service.lookup_cached_embedding(
                fingerprint,
                int(audio_feature_lengths.sum().item()),
            )

        if cached_embedding is None:
            (
                features,
                audio_feature_lengths,
                audio_chunk_mapping,
                audio_token_count,
            ) = _extract_audio_features(processor, audio)
        else:
            features = None
            audio_chunk_mapping = torch.zeros_like(audio_feature_lengths)
            audio_token_count = int(audio_feature_lengths.sum().item())

        if prompt == default_prompt:
            prefix_ids, suffix_ids = default_prompt_parts
        else:
            prefix_ids, suffix_ids = _prompt_token_parts(
                prompt,
                tokenizer,
                audio_token,
            )
        audio_span_ids = processor._audio_span_ids(audio_token_count)
        if len(prefix_ids) + len(audio_span_ids) + len(suffix_ids) > max_length:
            raise ValueError(f"Prompt/audio sequence exceeds max_length={max_length}")
        offsets = [
            (start + len(prefix_ids), end + len(prefix_ids))
            for start, end in _contiguous_offsets(
                audio_span_ids,
                audio_token_id,
            )
        ]
        if not offsets:
            raise ValueError("MOSS-Transcribe-Diarize prompt has no audio tokens")

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=prepared.fingerprint_int,
            feature=features,
            model_specific_data={
                "audio_feature_lengths": audio_feature_lengths,
                "audio_chunk_mapping": audio_chunk_mapping,
                "audio_fingerprint": fingerprint,
            },
        )
        audio_item.set_pad_value()
        audio_item.offsets = offsets

        if audio_encoder_service is not None:
            if cached_embedding is None:
                audio_encoder_service.encode_item(audio_item)
            else:
                audio_encoder_service.attach_embedding(audio_item, cached_embedding)

        padded_audio_span_ids = [
            audio_item.pad_value if token_id == audio_token_id else token_id
            for token_id in audio_span_ids
        ]
        padded_input_ids = [
            *prefix_ids,
            *padded_audio_span_ids,
            *suffix_ids,
        ]

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=audio_token_count,
            audio_token_id=audio_token_id,
            audio_start_id=audio_start_id,
            audio_end_id=audio_end_id,
        )

        temperature = _sampling_param(
            params, explicit_fields, "temperature", DEFAULT_TEMPERATURE, float
        )
        top_p = _sampling_param(params, explicit_fields, "top_p", DEFAULT_TOP_P, float)
        top_k = _sampling_param(params, explicit_fields, "top_k", DEFAULT_TOP_K, int)
        # Opt-in mitigation for repetition loops (#975): honoured only when
        # the caller sets it explicitly, so greedy defaults stay unchanged.
        repetition_penalty = _sampling_param(
            params, explicit_fields, "repetition_penalty", 1.0, float
        )
        # Same range SamplingParams enforces; failing here keeps the error
        # on the request path instead of inside the scheduler.
        if not 0.0 < repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in (0, 2]")
        # note (db-ol): the model default was sized for short clips and
        # silently cuts transcripts past about 20 minutes. Scale the default
        # budget with duration unless the operator configured a fixed one.
        requested_max_new_tokens = params.get("max_new_tokens")
        if requested_max_new_tokens is not None:
            request_max_new_tokens = int(requested_max_new_tokens)
            # note (db-ol): the API layer enforces ge=1 but internal callers
            # bypass it, and a zero here would otherwise silently fall
            # through to the duration scaled default.
            if request_max_new_tokens < 1:
                raise ValueError("max_new_tokens must be at least 1")
        elif duration_scaled_default:
            if audio_duration_s <= 0.0:
                # Empty audio has no legitimate long transcript; keep the
                # same floor as the scaled path so a loop cannot burn the
                # full fixed default here either.
                request_max_new_tokens = min(max_new_tokens, _EMPTY_AUDIO_OUTPUT_TOKENS)
                logger.warning(
                    "Request %s decoded to empty audio, the output budget "
                    "falls back to %d",
                    payload.request_id,
                    request_max_new_tokens,
                )
            else:
                # Note (Ruilin Gao): scale the budget with duration in both directions.
                # Raising it keeps dense transcripts past ~20 minutes from
                # being cut; capping it keeps greedy decoding from looping
                # until the fixed default on short non-speech audio
                # (laughter, noise), which inflated per-request latency
                # ~40x and cut CI batch throughput ~19% (#975). The floor
                # never exceeds the operator's fixed default, so a small
                # configured budget still bounds short requests.
                request_max_new_tokens = max(
                    min(max_new_tokens, _MIN_SCALED_OUTPUT_TOKENS),
                    math.ceil(audio_duration_s * _OUTPUT_TOKENS_PER_AUDIO_SECOND),
                )
        else:
            request_max_new_tokens = max_new_tokens
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[eos_token_id],
        )
        sampling_params.normalize(tokenizer=None)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=padded_input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            extra_key=fingerprint,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None

        logger.debug(
            f"[moss-td] prompt_tokens={len(padded_input_ids)} "
            f"audio_tokens={sum(end - start + 1 for start, end in offsets)} "
            f"chunks={int(audio_feature_lengths.numel())} "
            f"duration={audio_duration_s:.3f}s"
        )

        return MossTranscribeDiarizeRequestData(
            input_ids=torch.tensor(padded_input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=padded_input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            audio_duration_s=audio_duration_s,
            language=str(params.get("language") or "auto"),
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: MossTranscribeDiarizeRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        raw_text = _decode_token_ids(
            tokenizer,
            output_ids,
            skip_special_tokens=False,
        )
        text = postprocess_moss_transcribe_diarize_text(raw_text)
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "token_ids": output_ids,
                "language": data.language,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "prompt_tokens": len(data.prompt_token_ids or []),
                "completion_tokens": len(output_ids),
                "usage": {"engine_time_s": engine_time_s},
                "finish_reason": data.finish_reason,
                "weight_version": getattr(data, "weight_version", None),
                "modality": "text",
            },
        )

    return request_builder, result_adapter


def make_moss_transcribe_diarize_stream_output_builder(
    tokenizer: Any,
    eos_token_id: int | None = None,
    min_emit_interval_s: float = 0.0,
) -> Callable[[str, Any, Any], list[OutgoingMessage]]:
    tokenizer_eos = tokenizer.eos_token_id
    resolved_eos = (
        eos_token_id
        if eos_token_id is not None
        else (int(tokenizer_eos) if tokenizer_eos is not None else None)
    )
    return make_token_text_stream_output_builder(
        decode_fn=lambda ids: _decode_token_ids(
            tokenizer, ids, skip_special_tokens=True
        ),
        build_message_data=lambda delta: {
            "text": delta,
            "modality": "text",
            "stage_name": "asr",
        },
        build_message_metadata=lambda token_id: {
            "modality": "text",
            "token_id": token_id,
        },
        pending_ids_attr="_moss_stream_pending_ids",
        last_emit_attr="_moss_stream_last_emit_t",
        eos_token_id=resolved_eos,
        min_emit_interval_s=min_emit_interval_s,
        allow_terminal_flush=False,
        emit_trailing_replacement_on_terminal=False,
    )


__all__ = [
    "DEFAULT_TRANSCRIBE_DIARIZE_PROMPT",
    "MossTranscribeDiarizeRequestData",
    "make_moss_transcribe_diarize_scheduler_adapters",
    "make_moss_transcribe_diarize_stream_output_builder",
    "postprocess_moss_transcribe_diarize_text",
]
