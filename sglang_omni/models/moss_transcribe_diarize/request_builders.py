# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import logging
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

from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int, load_audio

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_AUDIO_PAD = "<|audio_pad|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END = "<|audio_end|>"
_SPECIAL_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
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
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file", "audio_data"):
            value = inputs.get(key)
            if value is not None:
                return value
        value = inputs.get("audios")
        if value is not None:
            return _only_audio(value)
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return value

    metadata = payload.request.metadata or {}
    value = metadata.get("audios")
    if value is not None:
        return _only_audio(value)
    for key in ("audio_data", "audio"):
        value = metadata.get(key)
        if value is not None:
            return value
    return inputs


def _has_metadata_audio_source(payload: StagePayload) -> bool:
    metadata = payload.request.metadata or {}
    return any(
        metadata.get(key) is not None for key in ("audios", "audio_data", "audio")
    )


def _load_audio(source: Any) -> np.ndarray:
    if isinstance(source, dict):
        if source.get("data") is not None:
            source = source["data"]
        elif source.get("path") is not None:
            source = source["path"]
        elif source.get("url") is not None:
            source = source["url"]
    return load_audio(
        source,
        source_name="MOSS-Transcribe-Diarize",
        target_sample_rate=_SAMPLE_RATE,
    )


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


def _prompt_from_payload(payload: StagePayload, processor: Any) -> str:
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
        input_text = DEFAULT_TRANSCRIBE_DIARIZE_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": ""},
                {"type": "text", "text": str(input_text)},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


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


def make_moss_transcribe_diarize_scheduler_adapters(
    processor: Any,
    tokenizer: Any,
    max_new_tokens: int,
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

    def request_builder(payload: StagePayload) -> MossTranscribeDiarizeRequestData:
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        explicit_fields = _explicit_generation_fields(metadata)
        audio = _load_audio(_audio_source_from_payload(payload))
        audio_duration_s = float(len(audio) / _SAMPLE_RATE)
        fingerprint = audio_fingerprint(audio)
        prompt = _prompt_from_payload(payload, processor)

        encoded = processor(
            text=prompt,
            audio=audio,
            return_tensors="pt",
            max_length=int(params.get("max_length") or 131072),
        )
        input_ids = encoded["input_ids"][0].tolist()
        features = encoded["input_features"]
        audio_feature_lengths = encoded["audio_feature_lengths"]
        audio_chunk_mapping = encoded["audio_chunk_mapping"]

        offsets = _contiguous_offsets(input_ids, audio_token_id)
        if not offsets:
            raise ValueError("MOSS-Transcribe-Diarize prompt has no audio tokens")

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=audio_fingerprint_int(fingerprint),
            feature=features,
            model_specific_data={
                "audio_feature_lengths": audio_feature_lengths,
                "audio_chunk_mapping": audio_chunk_mapping,
            },
        )
        audio_item.set_pad_value()
        audio_item.offsets = offsets

        padded_input_ids = [
            audio_item.pad_value if token_id == audio_token_id else token_id
            for token_id in input_ids
        ]

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=int(audio_feature_lengths.sum().item()),
            audio_token_id=audio_token_id,
            audio_start_id=audio_start_id,
            audio_end_id=audio_end_id,
        )

        temperature = _sampling_param(
            params, explicit_fields, "temperature", DEFAULT_TEMPERATURE, float
        )
        top_p = _sampling_param(params, explicit_fields, "top_p", DEFAULT_TOP_P, float)
        top_k = _sampling_param(params, explicit_fields, "top_k", DEFAULT_TOP_K, int)
        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
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
            "[moss-td] prompt_tokens=%d audio_tokens=%d chunks=%d duration=%.3fs",
            len(padded_input_ids),
            sum(end - start + 1 for start, end in offsets),
            int(audio_feature_lengths.numel()),
            audio_duration_s,
        )

        return MossTranscribeDiarizeRequestData(
            input_ids=torch.tensor(padded_input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=padded_input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
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

    def _build_stream_output(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        if req_data.req is None or req_output.data is None:
            return []
        req = req_data.req
        # note (guozhihao): while chunked prefill is still consuming prompt tokens, suppress
        # emission — prompt-side states would masquerade as output text.
        if req.is_chunked > 0:
            return []

        if req_data.stage_payload is None:
            return []
        stage_payload = req_data.stage_payload
        if not (stage_payload.request.params or {}).get("stream", False):
            return []

        try:
            token_id = int(req_output.data)
        except (TypeError, ValueError):
            return []

        try:
            pending = req._moss_stream_pending_ids
        except AttributeError:
            pending = []
            req._moss_stream_pending_ids = pending

        is_eos = resolved_eos is not None and token_id == resolved_eos
        if not is_eos:
            pending.append(token_id)
        if not pending:
            return []

        # note (guozhihao): rate-limit by holding tokens until the interval elapses;
        # last_emit == 0.0 means nothing emitted yet (first delta goes out immediately),
        # and EOS always flushes the remaining buffer.
        now = time.perf_counter()
        try:
            last_emit = req._moss_stream_last_emit_t
        except AttributeError:
            last_emit = 0.0
        if (
            not is_eos
            and min_emit_interval_s > 0.0
            and last_emit > 0.0
            and (now - last_emit) < min_emit_interval_s
        ):
            return []

        delta = _decode_token_ids(tokenizer, pending, skip_special_tokens=True)
        if delta.endswith("\ufffd"):
            return []
        pending.clear()
        if not delta:
            return []

        req._moss_stream_last_emit_t = now

        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data={
                    "text": delta,
                    "modality": "text",
                    "stage_name": "asr",
                },
                metadata={"modality": "text", "token_id": token_id},
            )
        ]

    return _build_stream_output


__all__ = [
    "DEFAULT_TRANSCRIBE_DIARIZE_PROMPT",
    "MossTranscribeDiarizeRequestData",
    "load_audio",
    "make_moss_transcribe_diarize_scheduler_adapters",
    "make_moss_transcribe_diarize_stream_output_builder",
    "postprocess_moss_transcribe_diarize_text",
]
