# SPDX-License-Identifier: Apache-2.0
"""Build Nemotron Transcribe requests and decoded text responses."""

from __future__ import annotations

import time
from collections.abc import Callable
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

from sglang_omni.models.nemotron_transcribe.encoder_service import (
    NemotronTranscribePreLMEncoderService,
)
from sglang_omni.models.nemotron_transcribe.hf_config import (
    AUDIO_TOKEN_ID,
    EOS_TOKEN_ID,
    num_audio_frames,
    num_audio_tokens,
)
from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.token_text_streaming import (
    make_token_text_stream_output_builder,
)
from sglang_omni.scheduling.types import DeferredAdmission, StreamOutputBuilder

SAMPLE_RATE = 16000
AUDIO_TOKEN = "<|audio|>"
DEFAULT_PROMPT = (
    "Produce a verbatim transcript of the audio. Preserve named entities, "
    "abbreviations, numbers, dates, measurements, acronyms, and technical terms "
    "as clearly as possible. Keep the same language as spoken and do not translate."
)


@dataclass
class NemotronTranscribeRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str | None = None
    engine_start_s: float = 0.0


class RequestBuilder(Protocol):
    def __call__(
        self, payload: StagePayload
    ) -> (
        NemotronTranscribeRequestData | DeferredAdmission[NemotronTranscribeRequestData]
    ): ...


class ResultAdapter(Protocol):
    def __call__(self, request_data: NemotronTranscribeRequestData) -> StagePayload: ...


def render_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    system_prompt: str = "",
) -> list[int]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        pass
    messages.append({"role": "user", "content": f"{AUDIO_TOKEN}\n{instruction}"})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def decode_tokens(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> str:
    return tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def make_scheduler_adapters(
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int,
    audio_token_estimator: dict[str, object] | None = None,
    audio_token_id: int = AUDIO_TOKEN_ID,
    eos_token_id: int = EOS_TOKEN_ID,
    max_audio_seconds: float | None = None,
    vocab_size: int = AUDIO_TOKEN_ID,
    audio_encoder_service: NemotronTranscribePreLMEncoderService | None = None,
    should_wait_for_encode: Callable[[], bool] | None = None,
) -> tuple[
    RequestBuilder,
    ResultAdapter,
]:
    def request_builder(
        payload: StagePayload,
    ) -> (
        NemotronTranscribeRequestData | DeferredAdmission[NemotronTranscribeRequestData]
    ):
        params = payload.request.params or {}
        prepared = prepare_audio(
            payload,
            source_name="Nemotron Transcribe",
            target_sample_rate=SAMPLE_RATE,
            max_duration_s=max_audio_seconds,
        )
        waveform = torch.from_numpy(prepared.waveform).float()
        if num_audio_frames(waveform.numel(), audio_token_estimator) < 2:
            raise ValueError(
                "Nemotron Transcribe requires at least two valid mel frames"
            )
        else:
            n_audio = num_audio_tokens(waveform.numel(), audio_token_estimator)
        cached_embedding = None
        if audio_encoder_service is not None:
            cached_embedding = audio_encoder_service.lookup_cached_embedding(
                prepared.fingerprint, n_audio
            )
        else:
            pass

        instruction = str(params.get("prompt") or DEFAULT_PROMPT)
        prompt_ids = render_prompt_ids(
            tokenizer, instruction, str(params.get("system_prompt") or "")
        )
        if prompt_ids.count(audio_token_id) != 1:
            raise ValueError(
                "Nemotron Transcribe prompt must contain exactly one <|audio|> token"
            )
        else:
            pass
        audio_start = prompt_ids.index(audio_token_id)

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=prepared.fingerprint_int,
            feature=None if cached_embedding is not None else waveform,
            model_specific_data={
                "num_audio_tokens": n_audio,
                "audio_fingerprint": prepared.fingerprint,
            },
        )
        audio_item.set_pad_value()
        input_ids = (
            prompt_ids[:audio_start]
            + [audio_item.pad_value] * n_audio
            + prompt_ids[audio_start + 1 :]
        )
        audio_item.offsets = [(audio_start, audio_start + n_audio - 1)]
        mm_inputs = MultimodalInputs(mm_items=[audio_item], num_image_tokens=n_audio)
        mm_inputs.audio_token_id = audio_token_id

        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        temperature = float(params.get("temperature") or 0.0)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=float(params.get("top_p") or 1.0),
            stop_token_ids=[int(eos_token_id)],
        )
        sampling_params.normalize(tokenizer=None)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            extra_key=prepared.fingerprint,
        )
        req.multimodal_inputs = mm_inputs

        if cached_embedding is not None:
            audio_encoder_service.attach_embedding(audio_item, cached_embedding)
        else:
            pass
        req_data = NemotronTranscribeRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=prepared.duration_s,
            language=params.get("language"),
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )
        if audio_encoder_service is None or cached_embedding is not None:
            return req_data
        else:
            pass
        # note (yhayaran): Waiting while a build worker is free keeps a failed encode's timeout.
        if should_wait_for_encode is not None and should_wait_for_encode():
            audio_encoder_service.encode_item(audio_item)
            return req_data
        else:
            pass
        return DeferredAdmission(
            value=req_data,
            ready=audio_encoder_service.submit_item(audio_item),
        )

    def result_adapter(data: NemotronTranscribeRequestData) -> StagePayload:
        payload = data.stage_payload
        assert payload is not None
        output_ids = [t for t in (data.output_ids or []) if t != eos_token_id]
        text = decode_tokens(tokenizer, output_ids).strip()
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
                "finish_reason": data.finish_reason,
                "usage": {
                    "engine_time_s": engine_time_s,
                    "prompt_tokens": len(data.prompt_token_ids or []),
                    "completion_tokens": len(output_ids),
                },
                "modality": "text",
            },
        )

    return request_builder, result_adapter


def make_stream_output_builder(
    tokenizer: PreTrainedTokenizerBase,
    min_emit_interval_s: float,
    eos_token_id: int,
) -> StreamOutputBuilder[SGLangARRequestData]:
    return make_token_text_stream_output_builder(
        decode_fn=lambda ids: decode_tokens(
            tokenizer, [t for t in ids if t != eos_token_id]
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
        pending_ids_attr="nemotron_transcribe_stream_pending_ids",
        last_emit_attr="nemotron_transcribe_stream_last_emit_seconds",
        eos_token_id=eos_token_id,
        min_emit_interval_s=min_emit_interval_s,
        allow_terminal_flush=True,
        emit_trailing_replacement_on_terminal=True,
    )


__all__ = [
    "DEFAULT_PROMPT",
    "NemotronTranscribeRequestData",
    "make_scheduler_adapters",
    "make_stream_output_builder",
    "render_prompt_ids",
]
