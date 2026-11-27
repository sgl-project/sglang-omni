# SPDX-License-Identifier: Apache-2.0
"""SGLang request construction + scheduler adapters for CSM TTS."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

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

TEXT_VOCAB_SIZE = 128_256


class CsmSamplingOptions(TypedDict, total=False):
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    sampling_seed: int


@dataclass
class CsmSGLangRequestData(SGLangARRequestData):
    """Per-request state for the CSM TTS scheduler."""

    context_codes: list[torch.Tensor] | None = None  # CPU int32 [F, 32] per segment
    # advances across chunked-prefill overlays of the same prompt.
    num_context_codes_consumed: int = 0
    output_codes: list[torch.Tensor] = field(default_factory=list)
    generation_done: bool = False
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    engine_start_s: float = 0.0
    stream_metadata: dict[str, str | int | bool] | None = None


class CsmRequestBuilder(Protocol):
    def __call__(self, payload: StagePayload) -> CsmSGLangRequestData: ...


class CsmResultAdapter(Protocol):
    def __call__(self, request_data: CsmSGLangRequestData) -> StagePayload: ...


def perf_counter() -> float:
    return time.perf_counter()


def coerce_context_codes(state: CsmTtsState) -> list[torch.Tensor] | None:
    """Coerce per-segment context codes to CPU [F, 32]."""
    if state.context_codes is None:
        return None
    else:
        pass
    coerced: list[torch.Tensor] = []
    for index, codes in enumerate(state.context_codes):
        codes_F32 = to_codes_F32(codes)
        if codes_F32 is None:
            raise ValueError(
                f"context segment {index} reached the tts_engine without encoded "
                "codes; the audio_encoder stage must run first"
            )
        else:
            pass
        coerced.append(codes_F32.cpu())
    return coerced


def context_audio_fingerprint(state: CsmTtsState) -> str | None:
    """Stable hash of context codes + speaker ids for Req.extra_key."""
    context_codes = coerce_context_codes(state)
    if not context_codes:
        return None
    else:
        pass
    digest = hashlib.blake2b(digest_size=16)
    for index, codes in enumerate(context_codes):
        entry = state.context[index] if index < len(state.context) else {}
        speaker = int(entry.get("speaker_id", entry.get("speaker", 0)) or 0)
        digest.update(f"|seg:{index}|spk:{speaker}|F:{int(codes.shape[0])}|".encode())
        flat_codes = codes.reshape(-1).tolist()
        packed = bytearray(2 * len(flat_codes))
        for position, code in enumerate(flat_codes):
            packed[2 * position] = code & 0xFF
            packed[2 * position + 1] = (code >> 8) & 0xFF
        digest.update(bytes(packed))
    return digest.hexdigest()


def build_sglang_csm_request(
    state: CsmTtsState, *, request_id: str = ""
) -> CsmSGLangRequestData:
    input_ids_list = [int(token_id) for token_id in state.prompt_token_ids]
    if not input_ids_list:
        raise ValueError(
            "CSM request reached the tts_engine without prompt_token_ids; "
            "preprocessing/audio_encoder must assemble the prompt first"
        )
    else:
        pass
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    sampling_options: CsmSamplingOptions = {
        "max_new_tokens": int(state.max_new_tokens),
        "temperature": float(state.temperature),
    }
    if state.top_p is not None:
        sampling_options["top_p"] = float(state.top_p)
    else:
        pass
    if state.top_k is not None:
        sampling_options["top_k"] = int(state.top_k)
    else:
        pass
    if state.seed is not None:
        sampling_options["sampling_seed"] = int(state.seed)
    else:
        pass
    sampling_params = SamplingParams(**sampling_options)
    # custom pipeline skips tokenizer_manager.normalize; stop lists stay None.
    sampling_params.normalize(tokenizer=None)

    context_codes = coerce_context_codes(state)

    engine_request = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=TEXT_VOCAB_SIZE,
        extra_key=context_audio_fingerprint(state),
    )
    engine_request._codec_suppress_tokens = (
        None  # noqa: leading-underscore  # SGLang Req field spelling
    )
    engine_request._input_embeds_are_projected = (
        False  # noqa: leading-underscore  # SGLang Req field spelling
    )

    return CsmSGLangRequestData(
        input_ids=input_ids,
        req=engine_request,
        context_codes=context_codes,
        num_context_codes_consumed=int(state.num_context_codes_consumed),
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p) if state.top_p is not None else 1.0,
        top_k=int(state.top_k) if state.top_k is not None else -1,
        depth_temperature=float(state.depth_temperature),
        depth_top_k=int(state.depth_top_k),
    )


def build_csm_stream_metadata(
    payload: StagePayload, request_data: CsmSGLangRequestData
) -> dict[str, str | int | bool] | None:
    params = payload.request.params
    if not isinstance(params, dict):
        raise TypeError(
            f"CSM request params must be a dict, got {type(params).__name__}"
        )
    else:
        pass
    if not bool(params.get("stream", False)):
        return None
    else:
        pass

    metadata: dict[str, str | int | bool] = {
        "modality": "audio_codes",
        "stream": True,
        "num_codebooks": NUM_CODEBOOKS,
        "codebook_size": CODEBOOK_VOCAB,
    }
    if params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) is not None:
        metadata[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = params[
            INITIAL_CODEC_CHUNK_FRAMES_PARAM
        ]
    else:
        pass
    return metadata


def apply_csm_result(state: CsmTtsState, request_data: CsmSGLangRequestData) -> None:
    if request_data.output_codes:
        codes = stack_frames(
            [torch.as_tensor(frame) for frame in request_data.output_codes]
        )
        state.output_codes = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    else:
        state.output_codes = None
        state.completion_tokens = 0
    state.prompt_tokens = len(request_data.input_ids)


def make_csm_scheduler_adapters(
    *,
    max_new_tokens_cap: int | None = DEFAULT_MAX_FRAMES,
) -> tuple[CsmRequestBuilder, CsmResultAdapter]:
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
        else:
            pass
        limit = (
            frame_budget
            if max_new_tokens_cap is None
            else min(int(max_new_tokens_cap), frame_budget)
        )
        state.max_new_tokens = max(1, min(int(state.max_new_tokens), limit))
        request_data = build_sglang_csm_request(state, request_id=payload.request_id)
        request_data.engine_start_s = perf_counter()
        request_data.stage_payload = payload
        request_data.stream_metadata = build_csm_stream_metadata(payload, request_data)
        return request_data

    def result_adapter(request_data: CsmSGLangRequestData) -> StagePayload:
        payload = request_data.stage_payload
        state = CsmTtsState.from_dict(payload.data)
        apply_csm_result(state, request_data)
        if request_data.engine_start_s:
            state.engine_time_s = perf_counter() - request_data.engine_start_s
        else:
            pass
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


__all__ = [
    "CsmRequestBuilder",
    "CsmResultAdapter",
    "CsmSGLangRequestData",
    "apply_csm_result",
    "build_csm_stream_metadata",
    "build_sglang_csm_request",
    "context_audio_fingerprint",
    "make_csm_scheduler_adapters",
]
