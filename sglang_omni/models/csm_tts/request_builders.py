# SPDX-License-Identifier: Apache-2.0
"""SGLang request construction + scheduler adapters for CSM TTS."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.csm_tts.utils import (
    BACKBONE_CTX,
    CODEBOOK_VOCAB,
    DEFAULT_MAX_FRAMES,
    NUM_CODEBOOKS,
    stack_frames,
    to_codes_F32,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM

_TEXT_VOCAB_SIZE = 128_256


@dataclass
class CsmSGLangRequestData(SGLangARRequestData):
    """Per-request state for the CSM TTS scheduler."""

    context_codes: list[Any] | None = None  # CPU int32 [F, 32] per segment
    num_ctx_codes_consumed: int = 0  # chunked-prefill overlay cursor
    output_codes: list[torch.Tensor] = field(default_factory=list)
    generation_done: bool = False
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    engine_start_s: float = 0.0
    stream_metadata: dict[str, Any] | None = None


_CsmRequestBuilder = Callable[[StagePayload], CsmSGLangRequestData]
_CsmResultAdapter = Callable[[CsmSGLangRequestData], StagePayload]


def perf_counter() -> float:
    return time.perf_counter()


def coerce_context_codes(state: CsmTtsState) -> list[torch.Tensor] | None:
    """Coerce per-segment context codes to CPU [F, 32]."""
    if state.context_codes is None:
        return None
    coerced: list[torch.Tensor] = []
    for i, codes in enumerate(state.context_codes):
        t = to_codes_F32(codes)
        if t is None:
            raise ValueError(
                f"context segment {i} reached the tts_engine without encoded "
                "codes; the audio_encoder stage must run first"
            )
        coerced.append(t.cpu())
    return coerced


def context_audio_fingerprint(state: CsmTtsState) -> str | None:
    """Stable hash of context codes + speaker ids for Req.extra_key."""
    context_codes = coerce_context_codes(state)
    if not context_codes:
        return None
    h = hashlib.blake2b(digest_size=16)
    for i, codes in enumerate(context_codes):
        entry = state.context[i] if i < len(state.context) else {}
        speaker = int(entry.get("speaker_id", entry.get("speaker", 0)) or 0)
        h.update(f"|seg:{i}|spk:{speaker}|F:{int(codes.shape[0])}|".encode())
        flat = codes.reshape(-1).tolist()
        buf = bytearray(2 * len(flat))
        for j, c in enumerate(flat):
            buf[2 * j] = c & 0xFF
            buf[2 * j + 1] = (c >> 8) & 0xFF
        h.update(bytes(buf))
    return h.hexdigest()


def build_sglang_csm_request(
    state: CsmTtsState, *, request_id: str = ""
) -> CsmSGLangRequestData:
    input_ids_list = [int(t) for t in state.prompt_token_ids]
    if not input_ids_list:
        raise ValueError(
            "CSM request reached the tts_engine without prompt_token_ids; "
            "preprocessing/audio_encoder must assemble the prompt first"
        )
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    sp_kwargs: dict[str, Any] = {
        "max_new_tokens": int(state.max_new_tokens),
        "temperature": float(state.temperature),
    }
    if state.top_p is not None:
        sp_kwargs["top_p"] = float(state.top_p)
    if state.top_k is not None:
        sp_kwargs["top_k"] = int(state.top_k)
    if state.seed is not None:
        sp_kwargs["sampling_seed"] = int(state.seed)
    sampling_params = SamplingParams(**sp_kwargs)
    # Custom pipeline skips tokenizer_manager.normalize(); stop lists stay None.
    sampling_params.normalize(tokenizer=None)

    context_codes = coerce_context_codes(state)

    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=_TEXT_VOCAB_SIZE,
        extra_key=context_audio_fingerprint(state),
    )
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    return CsmSGLangRequestData(
        input_ids=input_ids,
        req=req,
        context_codes=context_codes,
        num_ctx_codes_consumed=int(state.num_ctx_codes_consumed),
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p) if state.top_p is not None else 1.0,
        top_k=int(state.top_k) if state.top_k is not None else -1,
        depth_temperature=float(state.depth_temperature),
        depth_top_k=int(state.depth_top_k),
    )


def build_csm_stream_metadata(
    payload: StagePayload, data: CsmSGLangRequestData
) -> dict[str, Any] | None:
    params = payload.request.params
    if not isinstance(params, dict):
        raise TypeError(
            f"CSM request params must be a dict, got {type(params).__name__}"
        )
    if not bool(params.get("stream", False)):
        return None

    metadata: dict[str, Any] = {
        "modality": "audio_codes",
        "stream": True,
        "num_codebooks": NUM_CODEBOOKS,
        "codebook_size": CODEBOOK_VOCAB,
    }
    if params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) is not None:
        metadata[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = params[
            INITIAL_CODEC_CHUNK_FRAMES_PARAM
        ]
    return metadata


def apply_csm_result(state: CsmTtsState, data: CsmSGLangRequestData) -> None:
    if data.output_codes:
        codes = stack_frames([torch.as_tensor(f) for f in data.output_codes])
        state.output_codes = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    else:
        state.output_codes = None
        state.completion_tokens = 0
    state.prompt_tokens = len(data.input_ids)


def make_csm_scheduler_adapters(
    *,
    max_new_tokens_cap: int | None = DEFAULT_MAX_FRAMES,
) -> tuple[_CsmRequestBuilder, _CsmResultAdapter]:
    """Build scheduler request/result adapters."""

    def request_builder(payload: StagePayload) -> CsmSGLangRequestData:
        state = CsmTtsState.from_dict(payload.data)
        frame_budget = BACKBONE_CTX - 1 - len(state.prompt_token_ids)
        if frame_budget < 1:
            raise ValueError(
                f"prompt of {len(state.prompt_token_ids)} tokens leaves no frame "
                f"budget (max_req_len is {BACKBONE_CTX - 1}); shorten the "
                "text or context audio"
            )
        limit = (
            frame_budget
            if max_new_tokens_cap is None
            else min(int(max_new_tokens_cap), frame_budget)
        )
        state.max_new_tokens = max(1, min(int(state.max_new_tokens), limit))
        data = build_sglang_csm_request(state, request_id=payload.request_id)
        data.engine_start_s = perf_counter()
        data.stage_payload = payload
        data.stream_metadata = build_csm_stream_metadata(payload, data)
        return data

    def result_adapter(data: CsmSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = CsmTtsState.from_dict(payload.data)
        apply_csm_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = perf_counter() - data.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


__all__ = [
    "CsmSGLangRequestData",
    "apply_csm_result",
    "build_csm_stream_metadata",
    "build_sglang_csm_request",
    "context_audio_fingerprint",
    "make_csm_scheduler_adapters",
]
