# SPDX-License-Identifier: Apache-2.0
"""Request/result helpers for Fish Audio S2-Pro TTS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni_v1.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni_v1.proto import StagePayload
from sglang_omni_v1.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class S2ProSGLangRequestData(SGLangARRequestData):
    """S2-Pro per-request state."""

    vq_mask_tokens: Any = None
    vq_parts: list | None = None
    num_codebooks: int = 10
    codebook_size: int = 4096
    output_codes: list = field(default_factory=list)
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 1.1
    previous_semantic_tokens: list = field(default_factory=list)
    last_codebook_values: Any = None


def build_sglang_tts_request(
    state: S2ProState, tokenizer: Any, request_id: str = ""
) -> S2ProSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.utils.hf_transformers_utils import attach_additional_stop_token_ids

    from sglang_omni_v1.models.fishaudio_s2_pro.tokenizer import S2ProTokenizerAdapter

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

    sampling_params = SamplingParams(
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        repetition_penalty=state.repetition_penalty,
        stop_token_ids=[im_end_token_id],
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(tokenizer.vocab_size)

    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=tokenizer.vocab_size,
        eos_token_ids={im_end_token_id},
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
    )


def apply_tts_result(state: S2ProState, result: S2ProSGLangRequestData) -> None:
    assert result.output_codes, (
        "apply_tts_result expects non-empty output_codes; "
        "FishScheduler.emit_finished must filter immediate-EOS cases"
    )
    state.output_codes = torch.cat(result.output_codes, dim=1)
    state.completion_tokens = state.output_codes.shape[1]
    state.prompt_tokens = len(result.input_ids) if result.input_ids is not None else 0


def make_tts_scheduler_adapters(*, tokenizer: Any):
    """Build model-specific StagePayload <-> scheduler adapters for Fish TTS."""

    def request_builder(payload: StagePayload) -> S2ProSGLangRequestData:
        state = S2ProState.from_dict(payload.data)
        req_data = build_sglang_tts_request(
            state,
            tokenizer=tokenizer,
            request_id=payload.request_id,
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: S2ProSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = S2ProState.from_dict(payload.data)
        apply_tts_result(state, data)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter
