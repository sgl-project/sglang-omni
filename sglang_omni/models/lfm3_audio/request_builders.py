# SPDX-License-Identifier: Apache-2.0
"""Scheduler request and result adapters of the LFM3-Audio AR stage."""

from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from sglang_omni.models.lfm3_audio.depth import FrameSampling
from sglang_omni.models.lfm3_audio.generation import SpeechState
from sglang_omni.models.lfm3_audio.hf_config import Lfm3AudioTokens
from sglang_omni.models.lfm3_audio.io import Lfm3AudioState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

if TYPE_CHECKING:
    from sglang_omni.models.lfm3_audio.sglang_model import Lfm3AudioSGLangModel
else:
    pass

# The checkpoint's generation_config.json speech defaults.
DEFAULT_AUDIO_TEMPERATURE = 1.0
DEFAULT_AUDIO_TOP_K = 128
DEFAULT_AUDIO_REPETITION_PENALTY = 1.1
DEFAULT_AUDIO_REPETITION_WINDOW = 4096


@dataclass
class Lfm3AudioRequestData(SGLangARRequestData):
    speech: SpeechState | None = None
    input_features: torch.Tensor | None = None
    feature_attention_mask: torch.Tensor | None = None
    pending_feedback_queue: collections.deque[torch.Tensor] = field(
        default_factory=collections.deque
    )


def speech_sampling(state: Lfm3AudioState, device: str) -> FrameSampling:
    generator = None
    if state.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(state.seed))
    else:
        generator = torch.Generator(device=device)
        generator.seed()
    return FrameSampling(
        temperature=(
            DEFAULT_AUDIO_TEMPERATURE
            if state.audio_temperature is None
            else float(state.audio_temperature)
        ),
        top_k=(
            DEFAULT_AUDIO_TOP_K if state.audio_top_k is None else int(state.audio_top_k)
        ),
        repetition_penalty=(
            DEFAULT_AUDIO_REPETITION_PENALTY
            if state.audio_repetition_penalty is None
            else float(state.audio_repetition_penalty)
        ),
        repetition_window=DEFAULT_AUDIO_REPETITION_WINDOW,
        generator=generator,
    )


def build_request(
    payload: StagePayload,
    *,
    model: "Lfm3AudioSGLangModel",
    tokens: Lfm3AudioTokens,
    device: str,
) -> Lfm3AudioRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = Lfm3AudioState.from_dict(payload.data)
    prompt = [int(token) for token in state.input_ids or []]
    if not prompt:
        raise ValueError("LFM3-Audio received an empty prompt")
    else:
        pass
    vocab_size = model.config.vocab_size
    # The turn ends on <|im_end|>; <|text_end|> only ends a text-mode turn, which the runner maps to <|im_end|>.
    stop_id = tokens.eos_token_ids[0]
    sampling_params = SamplingParams(
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p),
        top_k=int(state.top_k),
        stop_token_ids=[stop_id],
    )
    sampling_params.normalize(None)
    sampling_params.verify(vocab_size)
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prompt,
        sampling_params=sampling_params,
        eos_token_ids={stop_id},
        vocab_size=vocab_size,
    )
    req.tokenizer = None
    req._codec_suppress_tokens = None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    speech = SpeechState.from_prompt(
        tokens,
        prompt[-1],
        text_only=bool(state.text_only),
        sampling=speech_sampling(state, device),
    )
    data = Lfm3AudioRequestData(
        input_ids=torch.tensor(prompt, dtype=torch.long),
        max_new_tokens=int(state.max_new_tokens),
        output_ids=req.output_ids,
        req=req,
        speech=speech,
        input_features=state.input_features,
        feature_attention_mask=state.feature_attention_mask,
    )
    data.stage_payload = payload
    return data


def apply_result(payload: StagePayload, data: Lfm3AudioRequestData) -> StagePayload:
    state = Lfm3AudioState.from_dict(payload.data)
    speech = data.speech
    state.generated_ids = list(speech.text_out) if speech is not None else []
    if speech is not None and speech.frames:
        state.audio_codes = torch.stack([frame.cpu() for frame in speech.frames], dim=0)
    else:
        state.audio_codes = torch.empty((0, 0), dtype=torch.long)
    state.input_features = None
    state.feature_attention_mask = None
    state.prompt_tokens = len(data.input_ids) if data.input_ids is not None else 0
    state.completion_tokens = len(state.generated_ids)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_adapters(
    *,
    model: "Lfm3AudioSGLangModel",
    tokens: Lfm3AudioTokens,
    device: str,
) -> tuple[
    Callable[[StagePayload], Lfm3AudioRequestData],
    Callable[[Lfm3AudioRequestData], StagePayload],
]:
    def request_builder(payload: StagePayload) -> Lfm3AudioRequestData:
        return build_request(payload, model=model, tokens=tokens, device=device)

    def result_adapter(data: Lfm3AudioRequestData) -> StagePayload:
        return apply_result(data.stage_payload, data)

    return request_builder, result_adapter
