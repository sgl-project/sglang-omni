# SPDX-License-Identifier: Apache-2.0
"""Engine request/result helpers for the S2-Pro TTS stage."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_ar import S2ProRequestData


def build_tts_request(state: S2ProState) -> S2ProRequestData:
    """Convert pipeline state into an S2ProRequestData for the engine."""
    input_ids = state.input_ids
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)

    vq_mask_tokens = state.vq_mask_tokens
    if vq_mask_tokens is not None and not isinstance(vq_mask_tokens, torch.Tensor):
        vq_mask_tokens = torch.tensor(vq_mask_tokens, dtype=torch.bool)

    vq_parts = state.vq_parts
    if vq_parts is not None:
        vq_parts = [
            p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in vq_parts
        ]

    return S2ProRequestData(
        input_ids=input_ids,
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


def apply_tts_result(state: S2ProState, result: Any) -> None:
    """Extract output codes from engine result and store in pipeline state."""
    if isinstance(result, S2ProRequestData):
        if result.output_codes:
            all_codes = torch.cat(result.output_codes, dim=1)
            state.output_codes = all_codes
        else:
            state.output_codes = None
    else:
        state.output_codes = result
