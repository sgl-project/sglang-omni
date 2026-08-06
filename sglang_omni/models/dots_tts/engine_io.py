# SPDX-License-Identifier: Apache-2.0
"""SGLang engine I/O adapters for dots.tts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang_omni.models.dots_tts.payload_types import (
    DotsTtsState,
    load_dots_tts_state,
    store_dots_tts_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

# note (chenyang): the first decode step regenerates the prompt tail patch and
# is dropped, so a request needs at least two decode spans to emit audio.
MIN_DECODE_SPANS = 2


@dataclass
class DotsTtsSGLangRequestData(SGLangARRequestData):
    """CPU scheduler state; recurrent GPU tensors live in the model runner."""

    enforce_request_limits: bool = True
    state: DotsTtsState | None = None
    audio_span_token_id: int = 0
    audio_eos_token_id: int = 0
    prompt_span_start: int = 0
    prompt_patch_count: int = 0
    engine_start_s: float = 0.0
    generated_latents: torch.Tensor | None = None
    stop_step: int | None = None


@dataclass(frozen=True)
class DotsTtsSpecialTokens:
    """Audio placeholder ids the AR stage needs from the artifact tokenizer."""

    audio_gen_span: int
    audio_gen_end: int


def load_dots_tts_special_tokens(tokenizer: Any) -> DotsTtsSpecialTokens:
    from sglang_omni.models.dots_tts.components.utils.tokenizer import (
        AUDIO_GEN_END_TOKEN,
        AUDIO_GEN_SPAN_TOKEN,
        require_token_id,
    )

    return DotsTtsSpecialTokens(
        audio_gen_span=require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN),
        audio_gen_end=require_token_id(tokenizer, AUDIO_GEN_END_TOKEN),
    )


def split_prefill_schedule(
    schedule_ids: list[int],
    *,
    audio_span_token_id: int,
    prompt_patch_count: int,
) -> tuple[list[int], int, int]:
    """Split a generation schedule into its prefill prefix and decode budget.

    Returns ``(prefill_ids, prompt_span_start, max_new_tokens)``. The prefill
    prefix stops right after the prompt audio spans; every remaining span is a
    decode step whose input embedding comes from the patch encoder.
    """
    if audio_span_token_id not in schedule_ids:
        raise ValueError("dots.tts generation schedule has no audio spans")
    prompt_span_start = schedule_ids.index(audio_span_token_id)
    total_spans = sum(1 for token_id in schedule_ids if token_id == audio_span_token_id)
    if prompt_span_start == 0:
        raise ValueError(
            "dots.tts generation schedule must start with text before audio spans"
        )
    if prompt_patch_count <= 0 or prompt_patch_count >= total_spans:
        raise ValueError(
            "dots.tts prompt_patch_count must fall inside the audio span budget: "
            f"prompt_patch_count={prompt_patch_count} audio_spans={total_spans}"
        )
    max_new_tokens = total_spans - prompt_patch_count
    if max_new_tokens < MIN_DECODE_SPANS:
        raise ValueError(
            "dots.tts needs at least "
            f"{MIN_DECODE_SPANS} decode spans, got {max_new_tokens}; "
            "raise max_audio_patches or shorten the reference audio"
        )
    return (
        schedule_ids[: prompt_span_start + prompt_patch_count],
        prompt_span_start,
        max_new_tokens,
    )


def make_dots_tts_scheduler_adapters(
    *,
    model: Any,
    special_tokens: DotsTtsSpecialTokens,
    reset_request: Callable[[str], None],
    num_steps: int,
    max_audio_patches: int,
):
    """Build StagePayload <-> SGLang request adapters for dots.tts."""

    vocab_size = int(model.llm_config.vocab_size)
    latent_patch_size = int(model.latent_patch_size)
    latent_dim = int(model.latent_dim)
    num_steps = int(num_steps)
    max_audio_patches = int(max_audio_patches)

    def request_builder(payload: StagePayload) -> DotsTtsSGLangRequestData:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        state = load_dots_tts_state(payload)
        # note (chenyang): the MeanFlow tail batches one shared ODE grid across
        # the running batch, and the DiT KV pools are sized once at startup, so
        # both knobs are engine-wide rather than per request.
        if int(state.num_steps) != num_steps:
            raise ValueError(
                "dots.tts num_steps is fixed by the tts_engine stage: "
                f"requested={int(state.num_steps)} engine={num_steps}"
            )
        if int(state.max_audio_patches) > max_audio_patches:
            raise ValueError(
                "dots.tts max_audio_patches exceeds the tts_engine capacity: "
                f"requested={int(state.max_audio_patches)} "
                f"engine={max_audio_patches}"
            )
        schedule_ids = [
            int(token_id) for token_id in (state.generation_schedule_ids or [])
        ]
        if not schedule_ids:
            raise ValueError(
                "dots.tts tts_engine requires a generation schedule from "
                "reference_encode"
            )
        prefill_ids, prompt_span_start, max_new_tokens = split_prefill_schedule(
            schedule_ids,
            audio_span_token_id=special_tokens.audio_gen_span,
            prompt_patch_count=int(state.prompt_patch_count),
        )

        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            stop_token_ids=[special_tokens.audio_gen_end],
        )
        sampling_params.normalize(None)
        sampling_params.verify(vocab_size)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=prefill_ids,
            sampling_params=sampling_params,
            eos_token_ids={special_tokens.audio_gen_end},
            vocab_size=vocab_size,
            extra_key=f"dots_tts:{payload.request_id}",
        )
        req.tokenizer = None
        req._input_embeds_are_projected = True

        data = DotsTtsSGLangRequestData(
            input_ids=torch.tensor(prefill_ids, dtype=torch.long),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            output_ids=req.output_ids,
            req=req,
            state=state,
            input_embeds_are_projected=True,
            audio_span_token_id=special_tokens.audio_gen_span,
            audio_eos_token_id=special_tokens.audio_gen_end,
            prompt_span_start=prompt_span_start,
            prompt_patch_count=int(state.prompt_patch_count),
            engine_start_s=time.perf_counter(),
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: DotsTtsSGLangRequestData) -> StagePayload:
        request_id = data.stage_payload.request_id
        try:
            state = data.state
            patches = data.generated_latents
            if patches is None or patches.numel() == 0:
                raise RuntimeError(
                    "dots.tts generated no payload latents; EOS fired immediately "
                    "after the prompt continuation"
                )
            state.generated_latents = patches.reshape(1, -1, latent_dim)
            state.completion_tokens = int(patches.numel()) // (
                latent_patch_size * latent_dim
            )
            state.finish_reason = _resolve_finish_reason(data)
            state.engine_time_s = time.perf_counter() - data.engine_start_s
            state.prompt_latent = None
            state.spk_emb = None
            state.generation_schedule_ids = None
            return store_dots_tts_state(data.stage_payload, state)
        finally:
            reset_request(request_id)

    return request_builder, result_adapter


def _resolve_finish_reason(data: DotsTtsSGLangRequestData) -> str:
    if data.stop_step is not None:
        return "stop"
    raw = data.finish_reason
    if raw is None and data.req is not None:
        finished_reason = getattr(data.req, "finished_reason", None)
        if finished_reason is not None and hasattr(finished_reason, "to_json"):
            raw = finished_reason.to_json().get("type")
        elif finished_reason is not None:
            raw = str(finished_reason)
    if raw is None:
        return "length"
    normalized = str(raw).lower()
    for marker in ("length", "abort", "error"):
        if marker in normalized:
            return marker
    return str(raw)


__all__ = [
    "MIN_DECODE_SPANS",
    "DotsTtsSGLangRequestData",
    "DotsTtsSpecialTokens",
    "load_dots_tts_special_tokens",
    "make_dots_tts_scheduler_adapters",
    "split_prefill_schedule",
]
