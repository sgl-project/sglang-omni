# SPDX-License-Identifier: Apache-2.0
"""Per-request data + StagePayload <-> scheduler adapters for Higgs TTS (V1)."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Callable, TypedDict

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.rollout_trace import build_omni_rollout_trace
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
    DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_HIGGS_STREAM_STRIDE,
    HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA,
    HIGGS_STREAM_STRIDE_METADATA,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)


HIGGS_CODEC_FRAMES_PER_S = 25.0
# note (JiaxinD / #1116): Higgs TTS 4B AR generation runs at 25 fps (~40 ms per token,
# 2048 completion tokens = 81.6 s of speech). Natural stops take ~1.5 s base (lead-in,
# delay winddown) plus ~0.4 s per target text token. A budget margin of 2.0x allows twice
# the slowest observed duration, so slow speech and pauses end naturally at codec EOC while
# unseeded runaway stragglers stop within seconds instead of hitting the global 2048 cap.
HIGGS_TTS_SLOWEST_STOP_S = 1.5
HIGGS_TTS_SLOWEST_STOP_S_PER_TOKEN = 0.4
HIGGS_TTS_BUDGET_MARGIN = 2.0
HIGGS_TTS_MIN_NEW_TOKENS = 50
HIGGS_DEFAULT_MAX_NEW_TOKENS = 2048


def estimate_text_tokens(text: str) -> int:
    """Heuristic token count estimate when a tokenizer is not available.

    Treats CJK / wide characters as 1 token each and Latin / non-CJK words as
    ~0.28 tokens per character (~3.5 characters per subword token).
    """
    if not text or not text.strip():
        return 0
    cjk_count = 0
    non_cjk_chars = 0
    for ch in text:
        if ord(ch) >= 0x2E80:
            cjk_count += 1
        elif not ch.isspace():
            non_cjk_chars += 1
    return max(1, cjk_count + math.ceil(non_cjk_chars * 0.28))


def text_length_budget(
    text_tokens: int,
    *,
    safety_factor: float = HIGGS_TTS_BUDGET_MARGIN,
    max_cap: int = HIGGS_DEFAULT_MAX_NEW_TOKENS,
    min_tokens: int = HIGGS_TTS_MIN_NEW_TOKENS,
) -> int:
    """The codec frames a target text of `text_tokens` tokens may take without an explicit max_new_tokens.

    Caps runaway unseeded AR sampling stragglers (Issue #1116).
    """
    if safety_factor <= 0:
        raise ValueError(f"safety_factor must be > 0, got {safety_factor}")
    budget = math.ceil(
        safety_factor
        * (
            HIGGS_TTS_SLOWEST_STOP_S
            + HIGGS_TTS_SLOWEST_STOP_S_PER_TOKEN * max(0, text_tokens)
        )
        * HIGGS_CODEC_FRAMES_PER_S
    )
    budget = max(min_tokens, budget)
    if max_cap is not None and max_cap > 0:
        budget = min(max_cap, budget)
    return budget


def calculate_higgs_max_new_tokens(
    text_or_tokens: str | int,
    *,
    tokenizer: object | None = None,
    safety_factor: float = HIGGS_TTS_BUDGET_MARGIN,
    max_cap: int = HIGGS_DEFAULT_MAX_NEW_TOKENS,
    min_tokens: int = HIGGS_TTS_MIN_NEW_TOKENS,
) -> int:
    """Derive dynamic max_new_tokens from text or token count."""
    if isinstance(text_or_tokens, (int, float)) and not isinstance(
        text_or_tokens, bool
    ):
        tokens = int(text_or_tokens)
    elif isinstance(text_or_tokens, str):
        if not text_or_tokens.strip():
            tokens = 0
        elif tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                tokens = len(tokenizer.encode(text_or_tokens, add_special_tokens=False))
            except Exception:
                tokens = estimate_text_tokens(text_or_tokens)
        else:
            tokens = estimate_text_tokens(text_or_tokens)
    else:
        tokens = 0
    return text_length_budget(
        tokens,
        safety_factor=safety_factor,
        max_cap=max_cap,
        min_tokens=min_tokens,
    )


class HiggsSamplingOptions(TypedDict, total=False):
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    sampling_seed: int


@dataclass
class HiggsSGLangRequestData(SGLangARRequestData):
    """Per-request state for the Higgs TTS scheduler."""

    reference_codes_delayed: list[list[int]] | None = None
    num_codebooks: int = 8
    codebook_size: int = 1026
    output_codes: list[torch.Tensor] = field(default_factory=list)
    output_code_buffer: torch.Tensor | None = None
    output_code_count: int = 0
    output_logprobs: list[torch.Tensor] = field(default_factory=list)
    return_omni_rollout: bool = False
    generation_done: bool = False
    engine_start_s: float = 0.0
    stream_metadata: dict[str, str | int | bool] | None = None
    stream_code_buffer: list[torch.Tensor] = field(default_factory=list)
    stream_code_seen_rows: int = 0
    stream_code_next_flush_rows: int = 0


_HiggsRequestBuilder = Callable[[StagePayload], HiggsSGLangRequestData]
_HiggsResultAdapter = Callable[[HiggsSGLangRequestData], StagePayload]


def perf_counter() -> float:
    return time.perf_counter()


def ref_audio_fingerprint(codes: list[list[int]] | None) -> str | None:
    """Stable hash of the full N-codebook ref-audio sequence.

    Returned as a short hex string used as ``Req.extra_key``. ``None`` for
    zero-shot (no ref audio) so all zero-shot requests share the radix subtree.
    Each codec value packs into 2 bytes (range 0..1025) so the hash is
    sensitive to every codebook, not just cb0.
    """
    if not codes:
        return None
    else:
        pass
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

    sp_kwargs: HiggsSamplingOptions = {
        "max_new_tokens": int(state.max_new_tokens),
        "temperature": float(state.temperature),
    }
    if state.top_p is not None:
        sp_kwargs["top_p"] = float(state.top_p)
    else:
        pass
    if state.top_k is not None:
        sp_kwargs["top_k"] = int(state.top_k)
    else:
        pass
    if state.seed is not None:
        sp_kwargs["sampling_seed"] = int(state.seed)
    else:
        pass
    sampling_params = SamplingParams(**sp_kwargs)
    # tokenizer_manager.normalize() is bypassed in our custom pipeline;
    # without it stop_strs / stop_regex_strs stay None and the upstream
    # scheduler's update_finish_state trips on ``len(None)``.
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
        extra_key=ref_audio_fingerprint(state.reference_codes_delayed),
    )
    # V1's prefill manager probes these attrs; absence triggers AttributeError.
    req._codec_suppress_tokens = None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    req._input_embeds_are_projected = False  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

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
        return_logprob=bool(state.return_logprob),
        return_omni_rollout=bool(state.return_omni_rollout),
    )


def build_higgs_stream_metadata(
    payload: StagePayload,
    data: HiggsSGLangRequestData,
    *,
    stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
    stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    initial_chunk_frames: int = DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
) -> dict[str, str | int | bool] | None:
    params = payload.request.params
    if not isinstance(params, dict):
        raise TypeError(
            f"Higgs request params must be a dict, got {type(params).__name__}"
        )
    else:
        pass
    if not bool(params.get("stream", False)):
        return None
    else:
        pass

    num_codebooks = int(data.num_codebooks)
    codebook_size = int(data.codebook_size)
    if num_codebooks <= 0 or codebook_size <= 2:
        raise ValueError(
            f"Invalid Higgs stream codec contract: "
            f"num_codebooks={num_codebooks}, codebook_size={codebook_size}"
        )
    else:
        pass
    metadata: dict[str, str | int | bool] = {
        "modality": "audio_codes",
        "stream": True,
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        HIGGS_STREAM_STRIDE_METADATA: stream_stride,
        HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA: stream_followup_stride,
        INITIAL_CODEC_CHUNK_FRAMES_PARAM: resolve_initial_codec_chunk_frames(
            params,
            steady_chunk_frames=max(1, stream_stride - num_codebooks + 1),
            default_frames=initial_chunk_frames,
        ),
    }
    return metadata


def apply_higgs_result(state: HiggsTtsState, data: HiggsSGLangRequestData) -> None:
    num_codebooks = int(data.num_codebooks)
    if data.output_code_buffer is not None and data.output_code_count > 0:
        codes = data.output_code_buffer[: data.output_code_count].to(torch.long)
        state.output_codes_delayed = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    elif data.output_codes:
        codes = torch.stack(data.output_codes, dim=0).to(torch.long)
        state.output_codes_delayed = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    else:
        codes = torch.empty((0, num_codebooks), dtype=torch.long)
        state.output_codes_delayed = None

    if data.return_omni_rollout:
        logprobs = (
            torch.stack(data.output_logprobs, dim=0).to(torch.float32)
            if (data.return_logprob and data.output_logprobs)
            else None
        )
        state.omni_rollout = build_omni_rollout_trace(
            codes,
            num_codebooks=num_codebooks,
            codebook_vocab_size=int(data.codebook_size),
            delayed_logprobs=logprobs,
        )
    else:
        pass
    state.prompt_tokens = len(data.input_ids)


def make_higgs_scheduler_adapters(
    *,
    max_new_tokens_cap: int | None = None,
    stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
    stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    initial_chunk_frames: int = DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES,
) -> tuple[_HiggsRequestBuilder, _HiggsResultAdapter]:
    """Build scheduler request/result adapters for :class:`HiggsTTSModel`."""

    def request_builder(payload: StagePayload) -> HiggsSGLangRequestData:
        state = HiggsTtsState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_new_tokens = min(
                int(state.max_new_tokens),
                int(max_new_tokens_cap),
            )
        else:
            pass
        data = build_sglang_higgs_request(state, request_id=payload.request_id)
        data.engine_start_s = perf_counter()
        data.stage_payload = payload
        data.stream_metadata = build_higgs_stream_metadata(
            payload,
            data,
            stream_stride=stream_stride,
            stream_followup_stride=stream_followup_stride,
            initial_chunk_frames=initial_chunk_frames,
        )
        return data

    def result_adapter(data: HiggsSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = HiggsTtsState.from_dict(payload.data)
        apply_higgs_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = perf_counter() - data.engine_start_s
        else:
            pass
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


__all__ = [
    "DEFAULT_HIGGS_INITIAL_CHUNK_FRAMES",
    "DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE",
    "DEFAULT_HIGGS_STREAM_STRIDE",
    "HIGGS_CODEC_FRAMES_PER_S",
    "HIGGS_DEFAULT_MAX_NEW_TOKENS",
    "HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA",
    "HIGGS_STREAM_STRIDE_METADATA",
    "HIGGS_TTS_BUDGET_MARGIN",
    "HIGGS_TTS_MIN_NEW_TOKENS",
    "HIGGS_TTS_SLOWEST_STOP_S",
    "HIGGS_TTS_SLOWEST_STOP_S_PER_TOKEN",
    "HiggsSGLangRequestData",
    "HiggsSamplingOptions",
    "INITIAL_CODEC_CHUNK_FRAMES_PARAM",
    "apply_higgs_result",
    "build_higgs_stream_metadata",
    "build_sglang_higgs_request",
    "calculate_higgs_max_new_tokens",
    "estimate_text_tokens",
    "make_higgs_scheduler_adapters",
    "text_length_budget",
]
