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
from sglang_omni.models.lfm3_audio.hf_config import (
    Lfm3AudioTokens,
    SpeechSamplingDefaults,
)
from sglang_omni.models.lfm3_audio.io import Lfm3AudioState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

if TYPE_CHECKING:
    from sglang_omni.models.lfm3_audio.sglang_model import Lfm3AudioSGLangModel
else:
    pass


@dataclass
class Lfm3AudioRequestData(SGLangARRequestData):
    speech: SpeechState | None = None
    input_features: torch.Tensor | None = None
    feature_attention_mask: torch.Tensor | None = None
    # LFM3-Omni vision inputs: flattened patches and (t, h, w) patch grids.
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    pending_feedback_queue: collections.deque[torch.Tensor] = field(
        default_factory=collections.deque
    )


def speech_sampling(
    state: Lfm3AudioState, device: str, defaults: SpeechSamplingDefaults
) -> FrameSampling:
    """A request's speech-frame sampling; unset fields take the checkpoint's defaults."""
    if state.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(state.seed))
    else:
        generator = torch.Generator(device=device)
        generator.seed()
    return FrameSampling(
        temperature=(
            defaults.temperature
            if state.audio_temperature is None
            else float(state.audio_temperature)
        ),
        top_k=defaults.top_k if state.audio_top_k is None else int(state.audio_top_k),
        repetition_penalty=(
            defaults.repetition_penalty
            if state.audio_repetition_penalty is None
            else float(state.audio_repetition_penalty)
        ),
        repetition_window=defaults.repetition_window,
        generator=generator,
    )


def build_request(
    payload: StagePayload,
    *,
    model: "Lfm3AudioSGLangModel",
    tokens: Lfm3AudioTokens,
    speech_defaults: SpeechSamplingDefaults,
    device: str,
) -> Lfm3AudioRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = Lfm3AudioState.from_dict(payload.data)
    prompt = [int(token) for token in state.input_ids or []]
    if not prompt:
        raise ValueError("LFM3 received an empty prompt")
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
        sampling=speech_sampling(state, device, speech_defaults),
    )
    data = Lfm3AudioRequestData(
        input_ids=torch.tensor(prompt, dtype=torch.long),
        max_new_tokens=int(state.max_new_tokens),
        output_ids=req.output_ids,
        req=req,
        speech=speech,
        input_features=state.input_features,
        feature_attention_mask=state.feature_attention_mask,
        pixel_values=state.pixel_values,
        image_grid_thw=state.image_grid_thw,
        pixel_values_videos=state.pixel_values_videos,
        video_grid_thw=state.video_grid_thw,
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
    state.pixel_values = None
    state.image_grid_thw = None
    state.pixel_values_videos = None
    state.video_grid_thw = None
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
    speech_defaults: SpeechSamplingDefaults,
    device: str,
) -> tuple[
    Callable[[StagePayload], Lfm3AudioRequestData],
    Callable[[Lfm3AudioRequestData], StagePayload],
]:
    def request_builder(payload: StagePayload) -> Lfm3AudioRequestData:
        return build_request(
            payload,
            model=model,
            tokens=tokens,
            speech_defaults=speech_defaults,
            device=device,
        )

    def result_adapter(data: Lfm3AudioRequestData) -> StagePayload:
        return apply_result(data.stage_payload, data)

    return request_builder, result_adapter
