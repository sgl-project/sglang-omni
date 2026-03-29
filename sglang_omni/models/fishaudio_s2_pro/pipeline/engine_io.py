# SPDX-License-Identifier: Apache-2.0
"""Engine request/result helpers for the S2-Pro TTS stage."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_sglang_ar import (
    S2ProSGLangRequestData,
)


def build_sglang_tts_request(
    state: S2ProState, tokenizer: Any, request_id: str
) -> S2ProSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    input_ids_list = list(state.input_ids)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    vq_mask_tokens = state.vq_mask_tokens
    if vq_mask_tokens is not None:
        vq_mask_tokens = torch.tensor(vq_mask_tokens, dtype=torch.bool)

    vq_parts = state.vq_parts
    if vq_parts is not None:
        vq_parts = [torch.tensor(p) for p in vq_parts]

    sampling_params = SamplingParams(
        max_new_tokens=state.max_new_tokens, temperature=state.temperature
    )

    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=tokenizer.vocab_size,
    )

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
    if result.output_codes:
        state.output_codes = torch.cat(result.output_codes, dim=1)
        state.completion_tokens = state.output_codes.shape[1]
    else:
        state.output_codes = None
    state.prompt_tokens = len(result.input_ids) if result.input_ids is not None else 0
