# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 request adapters — StagePayload <-> SGLang Req."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

from .constants import MAX_PROMPT_TOKENS
from .payload_types import MiniMaxMusic3State
from .prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS
from .request_builders import build_ttm_state

_C0_VOCAB_SIZE = 16384
_CFG_UNCOND_RID_SUFFIX = "-cfg"


def cfg_uncond_rid(request_id: str) -> str:
    """Request id of the unconditioned twin that guides this request."""
    return f"{request_id}{_CFG_UNCOND_RID_SUFFIX}"


def is_cfg_uncond_rid(rid: str) -> bool:
    return rid.endswith(_CFG_UNCOND_RID_SUFFIX)


@dataclass
class MiniMaxMusic3SGLangRequestData(SGLangARRequestData):
    enforce_request_limits: bool = True
    minimax_state: MiniMaxMusic3State | None = None
    cfg_uncond: "MiniMaxMusic3SGLangRequestData | None" = None
    is_cfg_uncond: bool = False
    prompt_token_ids: torch.Tensor | None = None
    prompt_tokens: int = 0
    ar_state: Any = None
    engine_start_s: float = 0.0


def build_sglang_minimax_request(
    payload: StagePayload, tokenizer: Any
) -> MiniMaxMusic3SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = build_ttm_state(payload)
    if state.prompt is None:
        raise RuntimeError("MiniMax Music 3 preprocessing did not build a prompt")
    # The twin's id is derived, so the scheduler can pair and abort by id alone.
    if is_cfg_uncond_rid(payload.request_id):
        raise ValueError(
            f"MiniMax Music 3 cannot serve request id {payload.request_id!r}: "
            f"the {_CFG_UNCOND_RID_SUFFIX} suffix is reserved for the CFG twin"
        )
    prompt_token_ids = tokenizer(state.prompt, return_tensors="pt")["input_ids"][0]
    prompt_tokens = int(prompt_token_ids.numel())
    if prompt_tokens > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"MiniMax Music 3 prompt has {prompt_tokens} tokens; "
            f"maximum is {MAX_PROMPT_TOKENS}"
        )

    max_new_tokens = int(state.max_audio_frames) + 1
    vocab_size = AUDIO_CODE_OFFSET + _C0_VOCAB_SIZE
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        stop_token_ids=[SPECIAL_TOKEN_IDS["<|audio_end|>"]],
    )
    sampling_params.normalize(None)
    sampling_params.verify(vocab_size)

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prompt_token_ids.tolist(),
        sampling_params=sampling_params,
        eos_token_ids={SPECIAL_TOKEN_IDS["<|audio_end|>"]},
        vocab_size=vocab_size,
    )
    req.tokenizer = None
    req._input_embeds_are_projected = True
    req._codec_suppress_tokens = None

    uncond_ids = prompt_token_ids.clone()
    uncond_ids[1:-2] = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
    uncond_req = Req(
        rid=cfg_uncond_rid(payload.request_id),
        origin_input_text="",
        origin_input_ids=uncond_ids.tolist(),
        sampling_params=sampling_params,
        eos_token_ids={SPECIAL_TOKEN_IDS["<|audio_end|>"]},
        vocab_size=vocab_size,
    )
    uncond_req.tokenizer = None
    uncond_req._input_embeds_are_projected = True
    uncond_req._codec_suppress_tokens = None
    uncond = MiniMaxMusic3SGLangRequestData(
        minimax_state=state,
        stage_payload=payload,
        req=uncond_req,
        output_ids=uncond_req.output_ids,
        input_ids=uncond_ids.unsqueeze(0),
        prompt_token_ids=uncond_ids,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        input_embeds_are_projected=True,
        is_cfg_uncond=True,
        engine_start_s=time.perf_counter(),
    )
    return MiniMaxMusic3SGLangRequestData(
        cfg_uncond=uncond,
        minimax_state=state,
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        input_ids=prompt_token_ids.unsqueeze(0),
        prompt_token_ids=prompt_token_ids,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        input_embeds_are_projected=True,
        engine_start_s=time.perf_counter(),
    )


def build_stream_output(
    request_id: str, data: MiniMaxMusic3SGLangRequestData, req_output: Any
) -> Iterator[OutgoingMessage]:
    del req_output
    yield from _drain_pending_chunks(request_id, data)


def flush_stream_output(
    request_id: str, data: MiniMaxMusic3SGLangRequestData
) -> Iterator[OutgoingMessage]:
    """Terminal drain — the runner's final windows are queued after the last
    decode step, so they need a flush pass of their own."""
    yield from _drain_pending_chunks(request_id, data)


build_stream_output.flush = flush_stream_output


def apply_minimax_result(data: MiniMaxMusic3SGLangRequestData) -> StagePayload:
    state = data.minimax_state
    if state is None:
        raise RuntimeError("MiniMax Music 3 result has no request state")
    state.caption = ""
    state.lyrics = ""
    state.prompt = None
    return store_state(data.stage_payload, state)


def _drain_pending_chunks(
    request_id: str, data: MiniMaxMusic3SGLangRequestData
) -> Iterator[OutgoingMessage]:
    ar_state = data.ar_state
    if ar_state is None:
        return
    pending = ar_state.pending_chunks
    while pending:
        chunk, metadata = pending.pop(0)
        yield OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=chunk,
            metadata=metadata,
        )


__all__ = [
    "MiniMaxMusic3SGLangRequestData",
    "apply_minimax_result",
    "build_sglang_minimax_request",
    "build_stream_output",
    "cfg_uncond_rid",
    "is_cfg_uncond_rid",
]
