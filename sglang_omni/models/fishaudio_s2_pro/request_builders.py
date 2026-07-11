# SPDX-License-Identifier: Apache-2.0
"""Request/result helpers for Fish Audio S2-Pro TTS."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

_S2PRO_GRAPH_TOP_K = 30


@dataclass
class S2ProSGLangRequestData(SGLangARRequestData):
    """S2-Pro per-request state."""

    vq_mask_tokens: Any = None
    vq_parts: list[torch.Tensor] | None = None
    num_codebooks: int = 10
    codebook_size: int = 4096
    output_codes: list[torch.Tensor] = field(default_factory=list)
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 1.1
    ras_window: int = 16
    ras_temperature: float = 1.0
    ras_top_p: float = 0.9
    seed: int | None = None
    previous_semantic_tokens: list[int] = field(default_factory=list)
    semantic_history_tokens: torch.Tensor | None = None
    semantic_history_count: int = 0
    last_codebook_values: Any = None
    latest_stream_code_chunk: torch.Tensor | None = None
    finish_reason: str | None = None
    engine_start_s: float = 0.0

    def __post_init__(self) -> None:
        validate_s2pro_top_k(self.top_k)


def validate_s2pro_top_k(top_k: int) -> None:
    if top_k == -1:
        return
    if not 1 <= top_k <= _S2PRO_GRAPH_TOP_K:
        raise ValueError(
            f"S2-Pro top_k must be -1 or between 1 and {_S2PRO_GRAPH_TOP_K}; got {top_k}"
        )


def _ref_vq_fingerprint(vq_parts: list[torch.Tensor] | None) -> str | None:
    # note (Gaokai): only cb0 of the ref VQ codes becomes prompt token ids;
    # cb1..N ride in as embeddings, so extra_key must hash all codebooks to keep
    # same-cb0 prompts from sharing radix KV across different reference audio.
    if not vq_parts:
        return None
    digest = hashlib.blake2b(digest_size=16)
    for part in vq_parts:
        codes = part.detach().to(device="cpu", dtype=torch.int32).contiguous()
        digest.update(str(tuple(codes.shape)).encode())
        digest.update(codes.numpy().tobytes())
    return digest.hexdigest()


def build_sglang_tts_request(
    state: S2ProState, tokenizer: Any, request_id: str = ""
) -> S2ProSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.utils.hf_transformers_utils import attach_additional_stop_token_ids

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import S2ProTokenizerAdapter

    input_ids_list = list(state.input_ids)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    vq_mask_tokens = state.vq_mask_tokens
    if vq_mask_tokens is not None:
        if isinstance(vq_mask_tokens, torch.Tensor):
            vq_mask_tokens = vq_mask_tokens.detach().clone().to(dtype=torch.bool)
        else:
            vq_mask_tokens = torch.as_tensor(vq_mask_tokens, dtype=torch.bool)

    vq_parts = state.vq_parts
    if vq_parts is not None:
        vq_parts = [
            p.detach().clone() if isinstance(p, torch.Tensor) else torch.as_tensor(p)
            for p in vq_parts
        ]

    if not hasattr(tokenizer, "additional_stop_token_ids"):
        attach_additional_stop_token_ids(tokenizer)

    adapter = S2ProTokenizerAdapter(tokenizer)
    im_end_token_id = int(adapter.eos_token_ids[0])
    # note (Gaokai): the semantic tokens live in the added vocab
    # (151678..155773 > tokenizer.vocab_size); Req must carry the full width or
    # upstream check_finished's vocab-boundary guard kills every request on its
    # first sampled code.
    vocab_size = len(tokenizer)

    sampling_params = SamplingParams(
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        repetition_penalty=state.repetition_penalty,
        stop_token_ids=[im_end_token_id],
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        eos_token_ids={im_end_token_id},
        extra_key=_ref_vq_fingerprint(vq_parts),
    )
    req.tokenizer = tokenizer
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    return S2ProSGLangRequestData(
        input_ids=input_ids,
        req=req,
        vq_mask_tokens=vq_mask_tokens,
        vq_parts=vq_parts,
        num_codebooks=state.num_codebooks,
        codebook_size=state.codebook_size,
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        repetition_penalty=state.repetition_penalty,
        ras_window=state.ras_window,
        ras_temperature=state.ras_temperature,
        ras_top_p=state.ras_top_p,
        seed=state.seed,
    )


def apply_tts_result(state: S2ProState, result: S2ProSGLangRequestData) -> None:
    if not result.output_codes:
        raise ValueError(
            f"Request {result.req.rid}: S2-Pro generated no audio codec tokens"
        )
    state.output_codes = torch.cat(result.output_codes, dim=1)
    state.completion_tokens = state.output_codes.shape[1]
    state.prompt_tokens = len(result.input_ids) if result.input_ids is not None else 0
    state.finish_reason = result.finish_reason or "stop"


def make_tts_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens_cap: int | None = None,
    context_length: int | None = None,
):
    """Build model-specific StagePayload <-> scheduler adapters for Fish TTS."""

    def request_builder(payload: StagePayload) -> S2ProSGLangRequestData:
        state = S2ProState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_new_tokens = min(
                int(state.max_new_tokens), int(max_new_tokens_cap)
            )
        if context_length is not None:
            # note (Gaokai): clamp instead of letting the scheduler's
            # KV-capacity check reject: long-prompt requests kept being served
            # under FishScheduler because generation stops at im_end early.
            state.max_new_tokens = min(
                int(state.max_new_tokens),
                max(int(context_length) - 1 - len(state.input_ids), 1),
            )
        req_data = build_sglang_tts_request(
            state,
            tokenizer=tokenizer,
            request_id=payload.request_id,
        )
        req_data.engine_start_s = time.perf_counter()
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: S2ProSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = S2ProState.from_dict(payload.data)
        apply_tts_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = time.perf_counter() - data.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    def stream_output_builder(
        request_id: str, data: S2ProSGLangRequestData, req_output: Any
    ) -> list[OutgoingMessage]:
        del req_output
        if not data.stage_payload.request.params.get("stream"):
            return []
        codes = data.latest_stream_code_chunk
        if codes is None:
            return []
        data.latest_stream_code_chunk = None
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=codes,
                target="vocoder",
                metadata={"modality": "audio_codes"},
            )
        ]

    return request_builder, result_adapter, stream_output_builder
