# SPDX-License-Identifier: Apache-2.0
"""Per-request data + StagePayload <-> scheduler adapters for Higgs TTS (V1)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.tts_streaming import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class HiggsSGLangRequestData(SGLangARRequestData):
    """Per-request state for the Higgs TTS scheduler."""

    reference_codes_delayed: list[list[int]] | None = None
    num_ref_codes_consumed: int = 0
    num_codebooks: int = 8
    codebook_size: int = 1026
    output_codes: list[torch.Tensor] = field(default_factory=list)
    generation_done: bool = False
    engine_start_s: float = 0.0
    stream_metadata: dict[str, Any] | None = None


class _ResettableHiggsModel(Protocol):
    def reset_request(self, req_id: str) -> None: ...


_HiggsRequestBuilder = Callable[[StagePayload], HiggsSGLangRequestData]
_HiggsResultAdapter = Callable[[HiggsSGLangRequestData], StagePayload]


def _ref_audio_fingerprint(codes: list[list[int]] | None) -> str | None:
    """Stable hash of the full N-codebook ref-audio sequence.

    Returned as a short hex string used as ``Req.extra_key``. ``None`` for
    zero-shot (no ref audio) so all zero-shot requests share the radix subtree.
    Each codec value packs into 2 bytes (range 0..1025) so the hash is
    sensitive to every codebook, not just cb0.
    """
    if not codes:
        return None
    buf = bytearray(2 * sum(len(row) for row in codes))
    i = 0
    for row in codes:
        for c in row:
            buf[i] = c & 0xFF
            buf[i + 1] = (c >> 8) & 0xFF
            i += 2
    return hashlib.blake2b(bytes(buf), digest_size=16).hexdigest()


def build_sglang_higgs_request(
    state: HiggsTtsState, *, request_id: str = ""
) -> HiggsSGLangRequestData:
    input_ids_list = list(state.prompt_token_ids)
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
    # tokenizer_manager.normalize() is bypassed in our custom pipeline;
    # without it stop_strs / stop_regex_strs stay None and the upstream
    # scheduler's check_finished trips on ``len(None)``.
    sampling_params.normalize(tokenizer=None)

    # vocab_size = backbone text vocab so cb0 rides sglang's standard sampler path.
    # extra_key namespaces the radix cache per ref-audio fingerprint so prompts
    # sharing the -100 placeholder prefix can never cross-contaminate KV.
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=151_936,
        extra_key=_ref_audio_fingerprint(state.reference_codes_delayed),
    )
    # V1's prefill manager probes these attrs; absence triggers AttributeError.
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    return HiggsSGLangRequestData(
        input_ids=input_ids,
        req=req,
        reference_codes_delayed=state.reference_codes_delayed,
        num_codebooks=int(state.num_codebooks),
        codebook_size=int(state.codebook_size),
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p) if state.top_p is not None else 1.0,
        top_k=int(state.top_k) if state.top_k is not None else -1,
    )


def build_higgs_stream_metadata(
    payload: StagePayload, data: HiggsSGLangRequestData
) -> dict[str, Any] | None:
    params = payload.request.params
    if not isinstance(params, dict):
        raise TypeError(
            f"Higgs request params must be a dict, got {type(params).__name__}"
        )
    if not bool(params.get("stream", False)):
        return None

    num_codebooks = int(data.num_codebooks)
    codebook_size = int(data.codebook_size)
    if num_codebooks <= 0 or codebook_size <= 2:
        raise ValueError(
            f"Invalid Higgs stream codec contract: "
            f"num_codebooks={num_codebooks}, codebook_size={codebook_size}"
        )
    metadata: dict[str, Any] = {
        "modality": "audio_codes",
        "stream": True,
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
    }
    if params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) is not None:
        metadata[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = params[
            INITIAL_CODEC_CHUNK_FRAMES_PARAM
        ]
    return metadata


def apply_higgs_result(state: HiggsTtsState, data: HiggsSGLangRequestData) -> None:
    if data.output_codes:
        codes = torch.stack(data.output_codes, dim=0).to(torch.long)
        state.output_codes_delayed = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    else:
        state.output_codes_delayed = None
    state.prompt_tokens = len(data.input_ids)


def make_higgs_scheduler_adapters(
    model: _ResettableHiggsModel,
    *,
    max_new_tokens_cap: int | None = None,
) -> tuple[_HiggsRequestBuilder, _HiggsResultAdapter]:
    """Build (request_builder, result_adapter) closures bound to a
    :class:`HiggsTTSModel` instance.

    The result adapter drops the model's per-request slot (sampler state +
    accumulated codes) once a result is emitted so a long-running server
    doesn't accumulate dead slots.
    """

    def request_builder(payload: StagePayload) -> HiggsSGLangRequestData:
        state = HiggsTtsState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_new_tokens = min(
                int(state.max_new_tokens),
                int(max_new_tokens_cap),
            )
        data = build_sglang_higgs_request(state, request_id=payload.request_id)
        data.engine_start_s = time.perf_counter()
        data.stage_payload = payload
        data.stream_metadata = build_higgs_stream_metadata(payload, data)
        return data

    def result_adapter(data: HiggsSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = HiggsTtsState.from_dict(payload.data)
        apply_higgs_result(state, data)
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
    "HiggsSGLangRequestData",
    "apply_higgs_result",
    "build_higgs_stream_metadata",
    "build_sglang_higgs_request",
    "make_higgs_scheduler_adapters",
]
