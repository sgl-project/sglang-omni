# SPDX-License-Identifier: Apache-2.0
"""Request/result helpers for Chatterbox-Turbo TTS."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.chatterbox.payload_types import ChatterboxState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

# Speech-side vocabulary boundaries (s3tokenizer): valid speech tokens are
# 0..6560, with SOS=6561 and EOS=6562; T3's speech_tokens_dict_size is 6563.
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
TEXT_VOCAB_SIZE = 50276
SPEECH_VOCAB_SIZE = 6563


@dataclass
class ChatterboxSGLangRequestData(SGLangARRequestData):
    """Per-request AR engine state for the T3 backbone."""

    speaker_embedding: torch.Tensor | None = None
    cond_prompt_speech_tokens: list[int] = field(default_factory=list)
    text_tokens: list[int] = field(default_factory=list)
    engine_start_s: float = 0.0
    seed: int | None = None


def build_sglang_tts_request(
    state: ChatterboxState,
    tokenizer: Any,
    request_id: str = "",
    *,
    vocab_size: int | None = None,
) -> ChatterboxSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    if vocab_size is None:
        vocab_size = TEXT_VOCAB_SIZE + SPEECH_VOCAB_SIZE

    # The T3 input sequence is [speaker, cond_speech, text, start_speech]. The
    # speaker slot is a continuous embedding injected by the model runner, so its
    # input_ids entry is a placeholder; the remaining slots are the real tokens.
    input_ids = (
        [0]
        + list(state.cond_prompt_speech_tokens)
        + list(state.text_tokens)
        + [START_SPEECH_TOKEN]
    )

    sampling_params = SamplingParams(
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        repetition_penalty=state.repetition_penalty,
        stop_token_ids=[STOP_SPEECH_TOKEN],
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        eos_token_ids={STOP_SPEECH_TOKEN},
    )
    req.tokenizer = tokenizer
    # The MLX runner reads the conditioning prompt from the Req.
    req._chatterbox_speaker_emb = state.speaker_embedding
    req._chatterbox_cond_speech_tokens = list(state.cond_prompt_speech_tokens)
    req._chatterbox_text_tokens = list(state.text_tokens)

    return ChatterboxSGLangRequestData(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        req=req,
        speaker_embedding=state.speaker_embedding,
        cond_prompt_speech_tokens=list(state.cond_prompt_speech_tokens),
        text_tokens=list(state.text_tokens),
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        repetition_penalty=state.repetition_penalty,
        seed=state.seed,
    )


def apply_tts_result(state: ChatterboxState, result: ChatterboxSGLangRequestData) -> None:
    output_ids = result.req.output_ids if result.req is not None else []
    speech_tokens = [int(token) for token in output_ids if int(token) < START_SPEECH_TOKEN]
    if not speech_tokens:
        raise ValueError(
            f"Request {result.req.rid if result.req is not None else '?'}: "
            "Chatterbox generated no speech tokens"
        )
    state.speech_tokens = speech_tokens
    state.completion_tokens = len(state.speech_tokens)
    state.finish_reason = result.finish_reason or "stop"


def make_tts_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens_cap: int | None = None,
):
    from sglang.srt.utils.hf_transformers_utils import (
        attach_additional_stop_token_ids,
    )

    if not hasattr(tokenizer, "additional_stop_token_ids"):
        attach_additional_stop_token_ids(tokenizer)

    vocab_size = TEXT_VOCAB_SIZE + SPEECH_VOCAB_SIZE

    def request_builder(payload: StagePayload) -> ChatterboxSGLangRequestData:
        state = ChatterboxState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_new_tokens = min(int(state.max_new_tokens), int(max_new_tokens_cap))
        req_data = build_sglang_tts_request(
            state,
            tokenizer=tokenizer,
            request_id=payload.request_id,
            vocab_size=vocab_size,
        )
        req_data.engine_start_s = time.perf_counter()
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: ChatterboxSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = ChatterboxState.from_dict(payload.data)
        apply_tts_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = time.perf_counter() - data.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    def stream_output_builder(
        _request_id: str,
        _data: ChatterboxSGLangRequestData,
        _req_output: Any,
    ) -> list[OutgoingMessage]:
        return []

    return request_builder, result_adapter, stream_output_builder
