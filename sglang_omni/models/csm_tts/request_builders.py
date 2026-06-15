# SPDX-License-Identifier: Apache-2.0
# Request construction + scheduler adapters follow the sglang-omni TTS pattern (cf. Higgs);
# EOS-frame and radix-key semantics mirror HuggingFace Transformers CSM (Apache-2.0).
"""SGLang request construction + scheduler adapters for CSM TTS.


"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

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
from sglang_omni.models.tts_streaming import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

_TEXT_VOCAB_SIZE = 128_256


@dataclass
class CsmSGLangRequestData(SGLangARRequestData):
    """Per-request state for the CSM TTS scheduler (engine-side mirror of
    :class:`CsmTtsState` plus runtime bookkeeping)."""

    context_codes: list[Any] | None = None  # CPU int32 [F, 32] per segment
    num_ctx_codes_consumed: int = 0  # chunked-prefill overlay cursor
    output_frames: list[torch.Tensor] = field(default_factory=list)
    generation_done: bool = False
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    engine_start_s: float = 0.0
    stream_metadata: dict[str, Any] | None = None


class _ResettableCsmModel(Protocol):
    def reset_request(self, req_id: str) -> None: ...


_CsmRequestBuilder = Callable[[StagePayload], CsmSGLangRequestData]
_CsmResultAdapter = Callable[[CsmSGLangRequestData], StagePayload]


def _coerce_context_codes(state: CsmTtsState) -> list[torch.Tensor] | None:
    """Engine-side coercion of per-segment context codes to CPU ``[F, 32]``
    long tensors. By the time a request reaches the tts_engine every segment
    MUST be encoded (the audio_encoder stage owns the raw-waveform path)."""
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


def _context_fingerprint(state: CsmTtsState) -> str | None:
    """blake2b-16 hex over ALL context-code matrices (2 bytes/code, range
    0..2050) + speaker ids; ``None`` for context-free requests so they share
    the radix subtree.

    Required because different context audios share identical ``128002×F``
    token prefixes — without ``Req.extra_key`` namespacing the radix tree
    would share KV across different voices (higgs lesson, verbatim).

    Returns:
        Short hex digest or ``None``.
    """
    context_codes = _coerce_context_codes(state)
    if not context_codes:
        return None
    h = hashlib.blake2b(digest_size=16)
    for i, codes in enumerate(context_codes):
        entry = state.context[i] if i < len(state.context) else {}
        speaker = int(entry.get("speaker_id", entry.get("speaker", 0)) or 0)
        h.update(f"|seg:{i}|spk:{speaker}|F:{int(codes.shape[0])}|".encode())
        # 2 bytes/code (range 0..2050) so the hash is sensitive to every
        # codebook of every frame, not just cb0 (higgs pattern).
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
    """Build the SGLang ``Req`` + request data for one CSM TTS request.

    Returns:
        :class:`CsmSGLangRequestData`.
    """
    input_ids_list = [int(t) for t in state.prompt_ids]
    if not input_ids_list:
        raise ValueError(
            "CSM request reached the tts_engine without prompt_ids; "
            "preprocessing/audio_encoder must assemble the prompt first"
        )
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    # max_new_tokens counts FRAMES (1 frame = 1 backbone position, like HF).
    sp_kwargs: dict[str, Any] = {
        "max_new_tokens": int(state.max_new_tokens),
        "temperature": float(state.temperature),
    }
    if state.top_p is not None:
        sp_kwargs["top_p"] = float(state.top_p)
    if state.top_k is not None:
        sp_kwargs["top_k"] = int(state.top_k)
    if state.seed is not None:
        # The kwarg is ``sampling_seed``, NOT ``seed``.
        sp_kwargs["sampling_seed"] = int(state.seed)
    sampling_params = SamplingParams(**sp_kwargs)
    # tokenizer_manager.normalize() is bypassed in our custom pipeline;
    # without it stop_strs / stop_regex_strs stay None and the upstream
    # scheduler's check_finished trips on ``len(None)``.
    sampling_params.normalize(tokenizer=None)

    context_codes = _coerce_context_codes(state)

    # vocab_size = backbone TEXT vocab so cb0 (0..2050) rides sglang's
    # standard token bookkeeping. extra_key namespaces the radix cache per
    # context-audio fingerprint so prompts sharing the same 128002×F
    # placeholder prefix can never cross-contaminate KV.
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=_TEXT_VOCAB_SIZE,
        extra_key=_context_fingerprint(state),
    )
    # V1's prefill manager probes these attrs; absence triggers AttributeError.
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
    """Stream metadata latched by the vocoder scheduler:
    ``{modality: "audio_codes", stream: True, num_codebooks: 32,
    codebook_size: 2051, [initial_codec_chunk_frames]}``.

    Returns:
        dict for streaming requests, ``None`` for non-streaming.
    """
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
    """Write ``output_frames`` + usage (prompt_tokens, completion_frames,
    engine_time_s) back into ``state``."""
    if data.output_frames:
        # [F, 32] incl. the EOS frame (HF ``sequences`` parity; the vocoder
        # never received it).
        codes = stack_frames([torch.as_tensor(f) for f in data.output_frames])
        state.output_frames = codes.tolist()
        state.completion_frames = int(codes.shape[0])
    else:
        state.output_frames = None
        state.completion_frames = 0
    state.prompt_tokens = len(data.input_ids)


def make_csm_scheduler_adapters(
    model: _ResettableCsmModel,
    max_new_tokens_cap: int | None = DEFAULT_MAX_FRAMES,
) -> tuple[_CsmRequestBuilder, _CsmResultAdapter]:
    """Build the (request_builder, result_adapter) pair for ``OmniScheduler``.

    request_builder: stamps ``engine_start_s`` / ``stage_payload`` /
    ``stream_metadata`` and clamps ``max_new_tokens`` to
    ``min(cap, BACKBONE_CTX - 1 - len(prompt_ids))`` — the scheduler's budget
    is ``max_req_len = min(context_length - 1, max_total_num_tokens - 1)``
    = 2047, NOT 2048 (omni_scheduler.py:150-153; clamping against 2048 admits
    requests the admission check then rejects).

    result_adapter: writes ``output_frames`` + usage into the state, calls
    ``model.reset_request(rid)`` (frees the sampler row), returns a fresh
    ``StagePayload``.

    Returns:
        ``(request_builder, result_adapter)``.
    """

    def request_builder(payload: StagePayload) -> CsmSGLangRequestData:
        state = CsmTtsState.from_dict(payload.data)
        frame_budget = BACKBONE_CTX - 1 - len(state.prompt_ids)
        if frame_budget < 1:
            raise ValueError(
                f"prompt of {len(state.prompt_ids)} tokens leaves no frame "
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
        data.engine_start_s = time.perf_counter()
        data.stage_payload = payload
        data.stream_metadata = build_csm_stream_metadata(payload, data)
        return data

    def result_adapter(data: CsmSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = CsmTtsState.from_dict(payload.data)
        apply_csm_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = time.perf_counter() - data.engine_start_s
        model.reset_request(payload.request_id)
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
    "make_csm_scheduler_adapters",
]
