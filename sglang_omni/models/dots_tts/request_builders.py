# SPDX-License-Identifier: Apache-2.0
"""Map dots.tts pipeline state onto SGLang's native request lifecycle."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.models.dots_tts.payload_types import (
    DotsTTSState,
    load_dots_tts_state,
    store_dots_tts_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

if TYPE_CHECKING:
    from sglang_omni.models.dots_tts.flow_head import DotsFlowState


@dataclass(frozen=True)
class DotsFlowResume:
    rng_state: torch.Tensor | None


@dataclass
class DotsTTSSGLangRequestData(SGLangARRequestData):
    state: DotsTTSState = field(default_factory=DotsTTSState)
    generation_schedule: torch.Tensor | None = None
    span_positions: torch.Tensor | None = None
    prompt_span_positions: torch.Tensor | None = None
    prefill_end: int = 0
    flow_state: DotsFlowState | DotsFlowResume | None = None
    latest_latent_patch: torch.Tensor | None = None
    latent_patches: list[torch.Tensor] = field(default_factory=list)
    decoded_latent_patches: list[torch.Tensor] = field(default_factory=list)
    stream_metadata: dict[str, Any] | None = None
    chunk_id: int = 0
    control_token_id: int = 0
    engine_start_s: float = 0.0


def build_sglang_dots_tts_request(
    payload: StagePayload,
) -> DotsTTSSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = load_dots_tts_state(payload)
    schedule = state.generation_schedule
    if schedule is None:
        raise RuntimeError("dots.tts preprocessing did not build a generation schedule")
    if schedule.ndim == 1:
        schedule = schedule.unsqueeze(0)
    audio_ids = {int(value) for value in state.audio_span_token_ids}
    if not audio_ids:
        raise RuntimeError("dots.tts preprocessing did not resolve audio span ids")
    schedule_row = schedule[0]
    span_positions = torch.nonzero(
        torch.isin(schedule_row, torch.tensor(sorted(audio_ids))), as_tuple=False
    ).squeeze(-1)
    prompt_frames = 0 if state.prompt_latents is None else state.prompt_latents.shape[1]
    prompt_patch_count = 0
    if state.prompt_latents is not None:
        configured_patch_size = int(state.latent_patch_size)
        if configured_patch_size <= 0:
            raise RuntimeError("dots.tts state is missing latent_patch_size")
        if prompt_frames % configured_patch_size:
            raise ValueError("dots.tts prompt latents are not patch aligned")
        prompt_patch_count = prompt_frames // configured_patch_size
    if span_positions.numel() <= prompt_patch_count:
        raise ValueError(
            "dots.tts generation schedule has no audio span after prompt prefill"
        )
    prefill_end = int(span_positions[prompt_patch_count].item())
    remaining_spans = int(span_positions.numel()) - prompt_patch_count
    discarded_patch_count = int(prompt_patch_count > 0)
    if remaining_spans <= discarded_patch_count:
        raise ValueError(
            "dots.tts prompt prefill requires one regenerated prompt patch "
            "and at least one payload patch"
        )
    generation_budget = remaining_spans
    if state.max_new_tokens is not None:
        generation_budget = min(
            generation_budget,
            int(state.max_new_tokens) + discarded_patch_count,
        )
    input_ids = schedule[0, :prefill_end].to(dtype=torch.long).tolist()
    control_token_id = int(schedule[0, span_positions[prompt_patch_count]])

    sampling_params = SamplingParams(
        max_new_tokens=generation_budget,
        temperature=0.0,
        stop_token_ids=[],
    )
    sampling_params.normalize(None)
    sampling_params.verify(int(state.vocab_size))
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        eos_token_ids=set(),
        vocab_size=int(state.vocab_size),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None
    return DotsTTSSGLangRequestData(
        state=state,
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        input_ids=schedule[:, :prefill_end],
        generation_schedule=schedule,
        span_positions=span_positions,
        prompt_span_positions=span_positions[:prompt_patch_count],
        prefill_end=prefill_end,
        max_new_tokens=generation_budget,
        input_embeds_are_projected=True,
        control_token_id=control_token_id,
        engine_start_s=time.perf_counter(),
        stream_metadata={
            "modality": "audio_latents",
            "stream": bool(state.stream),
        },
    )


def build_stream_output(
    request_id: str,
    data: DotsTTSSGLangRequestData,
    req_output: Any,
) -> Iterator[OutgoingMessage]:
    del req_output
    latent = data.latest_latent_patch
    data.latest_latent_patch = None
    if latent is None or not data.state.stream:
        return
    metadata = dict(data.stream_metadata or {})
    metadata["chunk_id"] = data.chunk_id
    data.chunk_id += 1
    yield OutgoingMessage(
        request_id=request_id,
        type="stream",
        target="vocoder",
        data=latent,
        metadata=metadata,
    )


def apply_latent_result(data: DotsTTSSGLangRequestData) -> StagePayload:
    state = data.state
    if not data.latent_patches:
        raise RuntimeError("dots.tts generated no payload latent patches")
    state.generated_latents = torch.cat(data.latent_patches, dim=1)
    state.prompt_tokens = int(data.input_ids.numel())
    state.completion_tokens = len(data.latent_patches)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    state.finish_reason = data.finish_reason
    return store_dots_tts_state(data.stage_payload, state)


__all__ = [
    "DotsTTSSGLangRequestData",
    "apply_latent_result",
    "build_sglang_dots_tts_request",
    "build_stream_output",
]
