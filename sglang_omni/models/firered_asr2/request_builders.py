# SPDX-License-Identifier: Apache-2.0
"""Transcription requests lowered into native SGLang multimodal requests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_asr2.audio import (
    FireRedFeatureExtractor,
    num_audio_tokens,
)
from sglang_omni.models.firered_asr2.config import FireRedASR2PipelineConfig
from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.token_text_streaming import (
    make_token_text_stream_output_builder,
)
from sglang_omni.scheduling.types import StreamOutputBuilder

SPEECH_TOKEN = "<speech>"
PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_SUFFIX = "请转写音频为文字<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class FireRedRequestData(SGLangARRequestData):
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    engine_start_s: float = 0.0


class FireRedRequestBuilder(Protocol):
    def __call__(self, payload: StagePayload) -> FireRedRequestData: ...


class FireRedResultAdapter(Protocol):
    def __call__(self, request_data: FireRedRequestData) -> StagePayload: ...


def build_prompt_ids(
    tokenizer: PreTrainedTokenizerBase, audio_token_count: int
) -> tuple[list[int], int, int]:
    if audio_token_count < 1:
        raise ValueError("Audio must produce at least one token")
    else:
        pass
    prefix_token_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
    suffix_token_ids = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)
    speech_token_id = tokenizer.convert_tokens_to_ids(SPEECH_TOKEN)
    if speech_token_id is None or speech_token_id == tokenizer.unk_token_id:
        raise ValueError("Checkpoint tokenizer is missing the <speech> token")
    else:
        pass
    return (
        prefix_token_ids + [speech_token_id] * audio_token_count + suffix_token_ids,
        len(prefix_token_ids),
        speech_token_id,
    )


def make_firered_asr2_scheduler_adapters(
    *,
    tokenizer: PreTrainedTokenizerBase,
    feature_extractor: FireRedFeatureExtractor,
    max_new_tokens: int,
    downsample_rate: int,
) -> tuple[FireRedRequestBuilder, FireRedResultAdapter]:
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def request_builder(payload: StagePayload) -> FireRedRequestData:
        params = payload.request.params or {}
        audio = prepare_audio(
            payload,
            source_name="FireRedASR2-LLM",
            max_duration_s=FireRedASR2PipelineConfig.max_native_clip_s,
        )
        feature = feature_extractor(audio.waveform)
        audio_token_count = num_audio_tokens(feature.shape[0], downsample_rate)
        input_ids, audio_token_offset, speech_token_id = build_prompt_ids(
            tokenizer, audio_token_count
        )
        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=audio.fingerprint_int,
            feature=feature,
            model_specific_data={"num_audio_tokens": audio_token_count},
        )
        audio_item.set_pad_value()
        input_ids = [
            audio_item.pad_value if token == speech_token_id else token
            for token in input_ids
        ]
        audio_item.offsets = [
            (audio_token_offset, audio_token_offset + audio_token_count - 1)
        ]
        multimodal_inputs = MultimodalInputs(
            mm_items=[audio_item], num_image_tokens=audio_token_count
        )
        multimodal_inputs.audio_token_id = speech_token_id
        requested_max_new_tokens = params.get("max_new_tokens")
        generation_token_limit = int(
            max_new_tokens
            if requested_max_new_tokens is None
            else requested_max_new_tokens
        )
        if generation_token_limit < 1:
            raise ValueError("max_new_tokens must be positive")
        else:
            pass
        temperature = float(params.get("temperature") or 0.0)
        sampling_params = SamplingParams(
            max_new_tokens=generation_token_limit,
            temperature=temperature,
            top_p=float(params.get("top_p", 1.0)),
            repetition_penalty=float(params.get("repetition_penalty", 1.0)),
            stop_token_ids=[int(eos_token_id)],
        )
        sampling_params.normalize(tokenizer=None)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=len(tokenizer),
            extra_key=audio.fingerprint,
        )
        req.multimodal_inputs = multimodal_inputs
        req._codec_suppress_tokens = (
            None  # noqa: leading-underscore  # Shared runner contract.
        )
        return FireRedRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            max_new_tokens=generation_token_limit,
            temperature=temperature,
            audio_duration_s=audio.duration_s,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(request_data: FireRedRequestData) -> StagePayload:
        payload = request_data.stage_payload
        text = tokenizer.decode(
            request_data.output_ids or [], skip_special_tokens=True
        ).strip()
        elapsed_seconds = time.perf_counter() - request_data.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "modality": "text",
                "duration_s": request_data.audio_duration_s,
                "asr_latency_s": elapsed_seconds,
                "usage": {"engine_time_s": elapsed_seconds},
            },
        )

    return request_builder, result_adapter


def make_firered_asr2_stream_output_builder(
    tokenizer: PreTrainedTokenizerBase,
) -> StreamOutputBuilder[FireRedRequestData]:
    return make_token_text_stream_output_builder(
        decode_fn=lambda input_ids: tokenizer.decode(
            input_ids, skip_special_tokens=True
        ),
        build_message_data=lambda delta: {"text": delta, "modality": "text"},
        build_message_metadata=lambda token: {"token_id": token},
        pending_ids_attr="firered_pending_text_ids",
        last_emit_attr="firered_last_text_emit",
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        allow_terminal_flush=True,
    )
