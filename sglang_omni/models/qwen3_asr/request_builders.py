# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Qwen3-ASR.

Unlike Whisper (encoder-decoder, features consumed inside the model forward),
Qwen3-ASR is a Qwen3 causal LM that ingests audio as multimodal embeddings:
the prompt contains an ``<|audio_pad|>`` placeholder repeated once per audio
token, and the model's ``general_mm_embed_routine`` scatters the encoder output
into those positions. So request_builder must:
  * reuse a cached LM-ready embedding before mel extraction when available,
  * otherwise extract mel features (WhisperFeatureExtractor) + attention mask,
  * compute how many audio tokens the encoder will emit,
  * build the chat prompt with that many ``<|audio_pad|>`` tokens,
  * hand features or precomputed embeddings over as a ``MultimodalDataItem``,
  * on a cache miss, submit encode and wait or return deferred admission.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.token_text_streaming import (
    make_token_text_stream_output_builder,
)
from sglang_omni.scheduling.types import DeferredAdmission
from sglang_omni.utils.audio import AudioDecodeError

from . import mrope_fast_path
from .audio_lengths import (
    QWEN3_ASR_OUTPUT_TOKENS_PER_SECOND,
    qwen3_asr_num_audio_tokens,
)
from .languages import resolve_language

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000

_AUDIO_START = "<|audio_start|>"
_AUDIO_PAD = "<|audio_pad|>"
_AUDIO_END = "<|audio_end|>"
_ASR_TEXT = "<asr_text>"
# Note(Audrey): the checkpoint's chat template emits this system turn even when
# empty; caller biasing text goes inside it.
_SYSTEM_PROMPT = "<|im_start|>system\n<|im_end|>\n"


def _system_turn(context: str | None) -> str:
    if not context:
        return _SYSTEM_PROMPT
    return f"<|im_start|>system\n{context}<|im_end|>\n"


@dataclass
class Qwen3ASRRequestData(SGLangARRequestData):
    enforce_request_limits: bool = True
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str | None = None
    engine_start_s: float = 0.0


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


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern:
        return None
    limit = len(values) - len(pattern) + 1
    for start in range(max(limit, 0)):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def _encode_literal(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    else:
        input_ids = encoded["input_ids"]
    return list(input_ids)


def make_qwen3_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
    feature_extractor: Any = None,
    context_length: int | None = None,
    audio_encoder_service: Any = None,
    should_wait_for_encode: Callable[[], bool] | None = None,
    greedy_only: bool = False,
) -> tuple[
    Callable[[StagePayload], Qwen3ASRRequestData | DeferredAdmission],
    Callable[[Any], StagePayload],
]:
    if feature_extractor is None:
        raise ValueError("Qwen3-ASR processor is missing a feature_extractor")

    audio_pad_token_id = int(tokenizer.convert_tokens_to_ids(_AUDIO_PAD))
    eos_token_id = int(tokenizer.eos_token_id)
    # note (Xinyu): added tokens such as <asr_text> live above
    # tokenizer.vocab_size. Req uses this bound to reject invalid model outputs,
    # so include the added tokens.
    vocab_size = len(tokenizer)
    asr_text_token_ids = _encode_literal(tokenizer, _ASR_TEXT)

    @lru_cache(maxsize=None)
    def _prompt_parts(
        language: str | None, context: str | None = None
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        prompt = _system_turn(context) + (
            f"<|im_start|>user\n"
            f"{_AUDIO_START}{_AUDIO_PAD}{_AUDIO_END}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if language is not None:
            prompt += f"language {language}<asr_text>"
        template_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        pad_positions = [
            index
            for index, token_id in enumerate(template_ids)
            if token_id == audio_pad_token_id
        ]
        if len(pad_positions) != 1:
            raise ValueError(
                "Qwen3-ASR prompt template must contain exactly one audio pad token"
            )
        audio_pad_index = pad_positions[0]
        return (
            tuple(template_ids[:audio_pad_index]),
            tuple(template_ids[audio_pad_index + 1 :]),
        )

    def _build_prompt_ids(
        num_audio_tokens: int, language: str | None, context: str | None = None
    ) -> list[int]:
        prefix_ids, suffix_ids = _prompt_parts(language, context)
        return [*prefix_ids, *([audio_pad_token_id] * num_audio_tokens), *suffix_ids]

    def _validate_context_budget(
        input_ids: list[int], request_max_new_tokens: int
    ) -> None:
        if (
            context_length is not None
            and len(input_ids) + request_max_new_tokens > context_length - 1
        ):
            raise ValueError(
                "Qwen3-ASR request is longer than the model's context length "
                f"({len(input_ids)} prompt/audio tokens + "
                f"{request_max_new_tokens} max_new_tokens > "
                f"{context_length - 1} usable tokens); "
                "reduce max_new_tokens or split the audio"
            )

    def request_builder(
        payload: StagePayload,
    ) -> Qwen3ASRRequestData | DeferredAdmission:
        params = payload.request.params or {}
        temperature = float(params.get("temperature") or 0.0)
        if greedy_only and temperature != 0.0:
            raise ValueError(
                "Qwen3-ASR Apple backend currently supports only greedy decoding; "
                "set temperature=0"
            )
        language = params.get("language")
        requested_language = None if language is None else str(language)
        forced_language = (
            None if requested_language is None else resolve_language(requested_language)
        )
        # Note(Audrey): the checkpoint reads biasing text from the system turn.
        raw_context = params.get("prompt")
        bias_context = str(raw_context).strip() if raw_context else None
        bias_context = bias_context or None
        try:
            prepared = prepare_audio(
                payload, source_name="Qwen3-ASR", target_sample_rate=_SAMPLE_RATE
            )
        except AudioDecodeError as exc:
            raise ValueError(
                "Qwen3-ASR could not decode the uploaded audio; provide a valid "
                "audio file."
            ) from exc
        audio = prepared.waveform
        audio_duration_s = prepared.duration_s
        fingerprint = prepared.fingerprint

        explicit_max_new_tokens = params.get("max_new_tokens")
        if explicit_max_new_tokens is not None:
            request_max_new_tokens = int(explicit_max_new_tokens)
        else:
            request_max_new_tokens = max(
                int(max_new_tokens),
                math.ceil(audio_duration_s * QWEN3_ASR_OUTPUT_TOKENS_PER_SECOND),
            )

        estimated_audio_tokens = None
        if context_length is not None or audio_encoder_service is not None:
            try:
                hop_length = int(feature_extractor.hop_length)
            except AttributeError as exc:
                raise ValueError(
                    "Qwen3-ASR feature extractor is missing its hop length"
                ) from exc
            if hop_length <= 0:
                raise ValueError(
                    "Qwen3-ASR feature extractor has an invalid hop length"
                )
            estimated_mel_frames = len(audio) // hop_length
            estimated_audio_tokens = qwen3_asr_num_audio_tokens(estimated_mel_frames)

        cached_embedding = None
        if audio_encoder_service is not None:
            assert estimated_audio_tokens is not None
            cached_embedding = audio_encoder_service.lookup_cached_embedding(
                fingerprint, estimated_audio_tokens
            )

        estimated_input_ids = None
        if context_length is not None:
            assert estimated_audio_tokens is not None
            # WhisperFeatureExtractor emits floor(samples / hop_length) frames.
            # Check the resulting prompt before the mel FFT so requests that
            # cannot fit the configured context do not consume preprocessing
            # memory and CPU.
            estimated_input_ids = _build_prompt_ids(
                estimated_audio_tokens, forced_language, bias_context
            )

            if explicit_max_new_tokens is None:
                remaining = context_length - 1 - len(estimated_input_ids)
                if remaining >= int(max_new_tokens):
                    request_max_new_tokens = min(request_max_new_tokens, remaining)

            _validate_context_budget(estimated_input_ids, request_max_new_tokens)

        if cached_embedding is None:
            # note (Jeffro Qu): unlike Whisper's default 30s window, here we pad the mel to the clip's true length.
            # WhisperFeatureExtractor defaults to padding="max_length", padding every clip to nb_max_frames=3000 (~30s),
            # so a short clip pays the full 30s of mel FFT on silence.
            # This is safe for Qwen3-ASR because its encoder is variable-length and keeps only the
            # valid frames via feature_attention_mask; vanilla Whisper's fixed-length encoder would instead break on padding="longest" (see ref: transformers#26241).
            # refs:
            #  https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py
            #  https://github.com/huggingface/transformers/issues/26241
            extracted = feature_extractor(
                audio,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
                padding="longest",
                # note (Junnan Li): Qwen3-ASR's encoder accepts mel sequences beyond
                # Whisper's 30-second window, so preprocessing must not truncate them.
                truncation=False,
            )
            features = extracted.input_features
            feature_attention_mask = getattr(extracted, "attention_mask", None)
            if feature_attention_mask is None:
                # WhisperFeatureExtractor normally returns one; fall back to all-valid.
                feature_attention_mask = torch.ones(
                    (features.shape[0], features.shape[-1]), dtype=torch.long
                )
            # note (Jeffro Qu): get_audio_feature uses the mask to select valid
            # frames; its no-mask branch transposes wrong, so the mask path must be taken.
            num_mel_frames = int(feature_attention_mask.sum().item())
            num_audio_tokens = int(qwen3_asr_num_audio_tokens(num_mel_frames))
            logger.debug(
                f"[qwen3-asr] mel_frames={num_mel_frames} "
                f"num_audio_tokens={num_audio_tokens} feat_shape={tuple(features.shape)}"
            )
        else:
            features = None
            feature_attention_mask = None
            assert estimated_audio_tokens is not None
            num_audio_tokens = estimated_audio_tokens

        input_ids = (
            estimated_input_ids
            if estimated_input_ids is not None
            and num_audio_tokens == estimated_audio_tokens
            else _build_prompt_ids(num_audio_tokens, forced_language, bias_context)
        )
        _validate_context_budget(input_ids, request_max_new_tokens)

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=prepared.fingerprint_int,
            feature=features,
            model_specific_data={
                "feature_attention_mask": feature_attention_mask,
                # note (luojiaxuan): the pre-LM encoder service reads these to
                # split batched encoder output and key its embedding cache.
                "num_audio_tokens": num_audio_tokens,
                "audio_fingerprint": fingerprint,
            },
        )
        # general_mm_embed_routine locates audio positions by matching each
        # item's pad_value against input_ids. The omni scheduler does not run
        # pad_input_ids for us, so compute the pad_value, replace the
        # <|audio_pad|> placeholders with it, and record the placeholder span as
        # item.offsets. SGLang treats offsets as inclusive.
        audio_item.set_pad_value()
        audio_start = input_ids.index(audio_pad_token_id)
        input_ids = [
            audio_item.pad_value if tok == audio_pad_token_id else tok
            for tok in input_ids
        ]
        audio_item.offsets = [(audio_start, audio_start + num_audio_tokens - 1)]

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=num_audio_tokens,
        )
        mm_inputs.audio_token_id = audio_pad_token_id
        # sglang indexes mm_input.mrope_positions[:, start:end] during prefill and
        # does not compute a default, so we must supply it. Qwen3-ASR's MRoPE is
        # degenerate for ASR (all 3 sections share the text position), i.e. plain
        # 1-D positions broadcast to shape [3, seq_len].
        seq_len = len(input_ids)
        positions = torch.arange(seq_len, dtype=torch.long)
        mm_inputs.mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()
        mm_inputs.mrope_position_delta = torch.tensor([0], dtype=torch.long)
        # Mark the input as degenerate-mrope so the vectorized decode fast path
        # (mrope_fast_path.py) can skip the per-request position loop for it.
        setattr(mm_inputs, mrope_fast_path.DEGENERATE_MROPE_FLAG, True)

        logger.debug(
            f"[qwen3-asr] sampling temp={temperature} "
            f"max_new_tokens={request_max_new_tokens} params={dict(params)}"
        )
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
        )
        sampling_params.normalize(tokenizer=None)

        if audio_encoder_service is not None and cached_embedding is not None:
            audio_encoder_service.attach_embedding(audio_item, cached_embedding)

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

        req_data = Qwen3ASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=audio_duration_s,
            language=requested_language,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )
        if audio_encoder_service is None or cached_embedding is not None:
            return req_data
        # note (guozhihao-224): a request is only admitted with its complete
        # LM-ready embedding. Submit after validation so a failed encode
        # never reaches the waiting queue. Wait in this worker when the
        # build queue still fits the pool so encode_item keeps its timeout
        # and failed counting; otherwise return deferred admission so
        # other builds can overlap encode.
        if should_wait_for_encode is not None and should_wait_for_encode():
            audio_encoder_service.encode_item(audio_item)
            return req_data
        return DeferredAdmission(
            value=req_data,
            ready=audio_encoder_service.submit_item(audio_item),
        )

    def result_adapter(data: Qwen3ASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        # Keep the marker handling at token level. Byte-level BPE decode->encode
        # is not an identity transform for all whitespace/Unicode transcripts.
        if logger.isEnabledFor(logging.DEBUG):
            raw = _decode_token_ids(tokenizer, output_ids, skip_special_tokens=False)
            logger.debug(
                f"[qwen3-asr] n_out={len(output_ids)} "
                f"ids={output_ids[:40]} raw={raw!r}"
            )
        asr_text_idx = _find_subsequence(output_ids, asr_text_token_ids)
        detected_language = None
        if data.language is None and asr_text_idx is not None:
            prefix = _decode_token_ids(
                tokenizer,
                output_ids[:asr_text_idx],
                skip_special_tokens=True,
            ).strip()
            label, separator, value = prefix.partition(" ")
            if separator and label.casefold() == "language":
                detected_language = value.strip() or None
                # The official parser treats this sentinel as unknown
                # language (and leaves any transcript tail untouched).
                if (
                    detected_language is not None
                    and detected_language.casefold() == "none"
                ):
                    detected_language = None
            elif prefix:
                detected_language = prefix
        transcript_ids = (
            output_ids[asr_text_idx + len(asr_text_token_ids) :]
            if asr_text_idx is not None
            else output_ids
        )
        text = _decode_token_ids(tokenizer, transcript_ids, skip_special_tokens=True)
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language or detected_language,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "usage": {"engine_time_s": engine_time_s},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


def make_qwen3_asr_stream_output_builder(
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
    asr_text_token_ids = _encode_literal(tokenizer, _ASR_TEXT)
    if not asr_text_token_ids:
        raise ValueError("Qwen3-ASR tokenizer produced no <asr_text> token IDs")

    token_stream_builder = make_token_text_stream_output_builder(
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
        pending_ids_attr="_qwen3_asr_stream_pending_ids",
        last_emit_attr="_qwen3_asr_stream_last_emit_t",
        eos_token_id=resolved_eos,
        min_emit_interval_s=min_emit_interval_s,
        allow_terminal_flush=True,
        emit_trailing_replacement_on_terminal=True,
    )

    def _build_stream_output(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        req = req_data.req
        if req is None or req.inflight_middle_chunks > 0:
            return token_stream_builder(request_id, req_data, req_output)
        stage_payload = req_data.stage_payload
        if stage_payload is None or not (stage_payload.request.params or {}).get(
            "stream", False
        ):
            return token_stream_builder(request_id, req_data, req_output)

        # note (Xinyu): Auto-language output starts with
        # language <name><asr_text>; suppress it until the transcript marker.
        if req_data.language is not None:
            return token_stream_builder(request_id, req_data, req_output)
        try:
            transcript_started = req._qwen3_asr_stream_transcript_started
        except AttributeError:
            transcript_started = False
        if transcript_started:
            return token_stream_builder(request_id, req_data, req_output)

        token_data = req_output.data
        if token_data is None:
            return token_stream_builder(request_id, req_data, req_output)
        try:
            token_id = int(token_data)
        except (TypeError, ValueError):
            return token_stream_builder(request_id, req_data, req_output)
        if resolved_eos is not None and token_id == resolved_eos:
            return token_stream_builder(request_id, req_data, req_output)

        try:
            matched = int(req._qwen3_asr_stream_marker_match_len)
        except AttributeError:
            matched = 0
        candidate = [*asr_text_token_ids[:matched], token_id]
        matched = 0
        for prefix_len in range(min(len(asr_text_token_ids), len(candidate)), 0, -1):
            if candidate[-prefix_len:] == asr_text_token_ids[:prefix_len]:
                matched = prefix_len
                break

        if matched == len(asr_text_token_ids):
            req._qwen3_asr_stream_transcript_started = True
            req._qwen3_asr_stream_marker_match_len = 0
        else:
            req._qwen3_asr_stream_marker_match_len = matched
        return []

    def _flush_stream_output(request_id: str, req_data: Any) -> list[OutgoingMessage]:
        return token_stream_builder.flush(request_id, req_data)  # type: ignore[attr-defined]

    _build_stream_output.flush = _flush_stream_output  # type: ignore[attr-defined]
    return _build_stream_output


__all__ = [
    "Qwen3ASRRequestData",
    "make_qwen3_asr_scheduler_adapters",
    "make_qwen3_asr_stream_output_builder",
]
