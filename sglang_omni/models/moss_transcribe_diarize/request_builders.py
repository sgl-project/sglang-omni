# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import (
    FINISH_MATCHED_STR,
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
_COMPLETE_SEGMENT_RE = re.compile(
    r"^\s*\[(?P<start>\d+(?:\.\d+)?)\]\s*"
    r"\[(?P<speaker>S\d{2,})\]"
    r"(?P<body>.*?)"
    r"\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL | re.IGNORECASE,
)
_WHISPER_ENCODER_STRIDE = 2
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
MOSS_TD_MARKER_LOOP_REASON = "moss_td_no_progress_marker_loop"
MOSS_TD_REPEATED_SEGMENT_REASON = "moss_td_no_progress_repeated_segment"
_NO_PROGRESS_MAX_PENDING_TOKEN_IDS = 256
_NO_PROGRESS_MAX_INCOMPLETE_CHARS = 65536
_NO_PROGRESS_MAX_INCOMPLETE_DECODE_STEPS = 256

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


@dataclass(frozen=True, slots=True)
class _NoProgressDecision:
    reason: str
    completed_segments: int
    marker_only_segments: int
    repeated_segments: int
    detected_completion_tokens: int


@dataclass(slots=True)
class _MossTDNoProgressState:
    marker_segment_limit: int
    repeat_segment_limit: int
    pending_token_ids: list[int] = field(default_factory=list)
    buffer: str = ""
    completed_segments: int = 0
    marker_only_segments: int = 0
    repeated_segments: int = 0
    observed_tokens: int = 0
    incomplete_decode_steps: int = 0
    last_content_signature: tuple[str, str, str, str] | None = None
    disabled: bool = False

    def observe_token_id(
        self,
        token_id: int,
        decode_fn: Callable[[list[int]], str],
    ) -> _NoProgressDecision | None:
        if self.disabled:
            return None
        self.observed_tokens += 1
        if len(self.pending_token_ids) >= _NO_PROGRESS_MAX_PENDING_TOKEN_IDS:
            # note (JiaxinD): some valid tokenizer IDs decode to a trailing
            # replacement forever. Bound full-prefix decode work and fail open.
            self._disable()
            return None
        self.pending_token_ids.append(token_id)
        try:
            decoded = decode_fn(self.pending_token_ids)
        except Exception:
            # note (JiaxinD): tokenizer plugins can surface different ordinary
            # exception types for malformed IDs. The optional detector must
            # fail open instead of escalating one request into a batch failure;
            # process-control BaseException subclasses are intentionally not caught.
            self._disable()
            return None
        if decoded.endswith("\ufffd"):
            return None
        if "\ufffd" in decoded:
            self._disable()
            return None
        self.pending_token_ids.clear()
        self.buffer += decoded
        pending_reason: str | None = None

        while match := _COMPLETE_SEGMENT_RE.match(self.buffer):
            self.buffer = self.buffer[match.end() :]
            self.completed_segments += 1
            pending_reason = None
            body = match.group("body")
            if not body.strip():
                self.marker_only_segments += 1
                self.repeated_segments = 0
                self.last_content_signature = None
                if (
                    self.marker_segment_limit > 0
                    and self.marker_only_segments >= self.marker_segment_limit
                ):
                    pending_reason = MOSS_TD_MARKER_LOOP_REASON
                continue

            self.marker_only_segments = 0
            signature = (
                match.group("start"),
                match.group("speaker"),
                body,
                match.group("end"),
            )
            if signature == self.last_content_signature:
                self.repeated_segments += 1
            else:
                self.repeated_segments = 0
                self.last_content_signature = signature
            if (
                self.repeat_segment_limit > 0
                and self.repeated_segments >= self.repeat_segment_limit
            ):
                pending_reason = MOSS_TD_REPEATED_SEGMENT_REASON

        # note (JiaxinD): a trigger is actionable only when the observed token
        # ends exactly on a segment boundary. Streaming output cannot retract
        # an incomplete suffix, so a partial next segment postpones the stop.
        if not self.buffer.strip():
            self.incomplete_decode_steps = 0
            return self._decision(pending_reason) if pending_reason else None
        self.incomplete_decode_steps += 1
        if (
            len(self.buffer) > _NO_PROGRESS_MAX_INCOMPLETE_CHARS
            or self.incomplete_decode_steps >= _NO_PROGRESS_MAX_INCOMPLETE_DECODE_STEPS
        ):
            # note (JiaxinD): unknown output is not loop evidence. Disable the
            # detector for this request instead of attempting a lossy resync
            # or repeatedly rescanning an ever-growing incomplete segment.
            self._disable()
        return None

    def _disable(self) -> None:
        self.disabled = True
        self.pending_token_ids.clear()
        self.buffer = ""
        self.marker_only_segments = 0
        self.repeated_segments = 0
        self.incomplete_decode_steps = 0
        self.last_content_signature = None

    def _decision(self, reason: str) -> _NoProgressDecision:
        return _NoProgressDecision(
            reason=reason,
            completed_segments=self.completed_segments,
            marker_only_segments=self.marker_only_segments,
            repeated_segments=self.repeated_segments,
            detected_completion_tokens=self.observed_tokens,
        )


@dataclass
class MossTranscribeDiarizeRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "auto"
    engine_start_s: float = 0.0
    no_progress_termination_reason: str | None = None
    no_progress_completed_segments: int = 0
    no_progress_marker_only_segments: int = 0
    no_progress_repeated_segments: int = 0
    no_progress_detected_completion_tokens: int = 0
    no_progress_marker_limit: int = 0
    no_progress_repeat_limit: int = 0
    no_progress_response_mode: str = ""
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
        result_data = {
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
        }
        if data.no_progress_termination_reason is not None:
            termination_record = {
                "schema_version": 1,
                "server_request_id_sha256": hashlib.sha256(
                    str(payload.request_id).encode("utf-8")
                ).hexdigest(),
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "reason": data.no_progress_termination_reason,
                "completed_segments": data.no_progress_completed_segments,
                "marker_only_segments": data.no_progress_marker_only_segments,
                "repeated_segments": data.no_progress_repeated_segments,
                "detected_completion_tokens": data.no_progress_detected_completion_tokens,
                "raw_completion_tokens": len(output_ids),
                "applied_max_new_tokens": data.max_new_tokens,
                "marker_limit": data.no_progress_marker_limit,
                "repeat_limit": data.no_progress_repeat_limit,
                "response_mode": data.no_progress_response_mode,
                "complete_boundary": True,
            }
            logger.warning(
                "MOSS_TD_TERMINATION_JSON %s",
                json.dumps(termination_record, separators=(",", ":"), sort_keys=True),
            )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=result_data,
        )

    return request_builder, result_adapter


def make_moss_transcribe_diarize_stream_output_builder(
    tokenizer: Any,
    eos_token_id: int | None = None,
    min_emit_interval_s: float = 0.0,
    buffered_no_progress_marker_segments: int = 0,
    buffered_no_progress_repeat_segments: int = 0,
) -> Callable[[str, Any, Any], list[OutgoingMessage]]:
    tokenizer_eos = tokenizer.eos_token_id
    resolved_eos = (
        eos_token_id
        if eos_token_id is not None
        else (int(tokenizer_eos) if tokenizer_eos is not None else None)
    )
    stream_output_builder = make_token_text_stream_output_builder(
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
    marker_limit = int(buffered_no_progress_marker_segments)
    repeat_limit = int(buffered_no_progress_repeat_segments)
    if marker_limit < 0 or repeat_limit < 0:
        raise ValueError("MOSS-TD buffered no-progress limits must be non-negative")
    if marker_limit == 0 and repeat_limit == 0:
        return stream_output_builder

    def decode_no_progress_ids(token_ids: list[int]) -> str:
        return _decode_token_ids(
            tokenizer,
            token_ids,
            skip_special_tokens=True,
        )

    def guarded_stream_output_builder(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        req = req_data.req
        stage_payload = getattr(req_data, "stage_payload", None)
        request = getattr(stage_payload, "request", None)
        params = getattr(request, "params", None) or {}
        token_data = getattr(req_output, "data", None)
        if (
            not bool(params.get("stream"))
            and req is not None
            and getattr(req, "inflight_middle_chunks", 0) == 0
            and token_data is not None
            and getattr(req, "finished_reason", None) is None
            and getattr(req, "to_finish", None) is None
        ):
            try:
                token_id = int(token_data)
            except (TypeError, ValueError):
                token_id = None
            if token_id is not None and token_id != resolved_eos:
                state = getattr(req, "_moss_no_progress_state", None)
                if state is None:
                    state = _MossTDNoProgressState(
                        marker_segment_limit=marker_limit,
                        repeat_segment_limit=repeat_limit,
                    )
                    req._moss_no_progress_state = state
                decision = state.observe_token_id(token_id, decode_no_progress_ids)
                if decision is not None and not getattr(req, "output_ids", None):
                    # note (JiaxinD): the scheduler callback precedes SGLang's
                    # first prefill-token commit. Finishing at this point makes
                    # the result processor drop the boundary token. Suppress
                    # only this decision; a later complete decode boundary can
                    # safely confirm the same no-progress sequence.
                    decision = None
                if decision is not None and getattr(req, "to_finish", None) is None:
                    req.to_finish = FINISH_MATCHED_STR(matched=decision.reason)
                    req_data.no_progress_termination_reason = decision.reason
                    req_data.no_progress_completed_segments = (
                        decision.completed_segments
                    )
                    req_data.no_progress_marker_only_segments = (
                        decision.marker_only_segments
                    )
                    req_data.no_progress_repeated_segments = decision.repeated_segments
                    req_data.no_progress_detected_completion_tokens = (
                        decision.detected_completion_tokens
                    )
                    req_data.no_progress_marker_limit = marker_limit
                    req_data.no_progress_repeat_limit = repeat_limit
                    req_data.no_progress_response_mode = "buffered"
                    logger.warning(
                        "MOSS-TD no-progress termination request_id_sha256=%s "
                        "reason=%s completion_tokens=%d completed_segments=%d "
                        "marker_only_segments=%d repeated_segments=%d",
                        hashlib.sha256(str(request_id).encode("utf-8")).hexdigest(),
                        decision.reason,
                        decision.detected_completion_tokens,
                        decision.completed_segments,
                        decision.marker_only_segments,
                        decision.repeated_segments,
                    )
        return stream_output_builder(request_id, req_data, req_output)

    return guarded_stream_output_builder


__all__ = [
    "DEFAULT_TRANSCRIBE_DIARIZE_PROMPT",
    "MossTranscribeDiarizeRequestData",
    "MOSS_TD_MARKER_LOOP_REASON",
    "MOSS_TD_REPEATED_SEGMENT_REASON",
    "make_moss_transcribe_diarize_scheduler_adapters",
    "make_moss_transcribe_diarize_stream_output_builder",
    "postprocess_moss_transcribe_diarize_text",
]
