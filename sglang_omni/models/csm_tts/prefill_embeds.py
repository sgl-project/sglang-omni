# SPDX-License-Identifier: Apache-2.0
"""Prefill input-embed overlay for CSM context audio frames."""

from __future__ import annotations

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.csm_tts.model import CsmTTSModel
from sglang_omni.models.csm_tts.request_builders import CsmSGLangRequestData
from sglang_omni.models.csm_tts.utils import (
    AUDIO_EOS_TOKEN_ID,
    AUDIO_TOKEN_ID,
    NUM_CODEBOOKS,
)
from sglang_omni.scheduling.types import SchedulerRequest


def concat_context_codes(
    request_data: CsmSGLangRequestData, device: torch.device
) -> torch.Tensor | None:
    """Concatenate per-segment context codes in prompt order."""
    segments = request_data.context_codes or []
    code_frames = [
        torch.as_tensor(segment).reshape(-1, NUM_CODEBOOKS).to(torch.long)
        for segment in segments
    ]
    if not code_frames:
        return None
    else:
        pass
    return torch.cat(code_frames, dim=0).to(device)


def build_prefill_input_embeds(
    model: CsmTTSModel,
    forward_batch: ForwardBatch,
    requests: list[SchedulerRequest],
) -> torch.Tensor | None:
    """Overlay context-frame embeds at 128002 and the zeros frame at 128003."""
    if not any(scheduler_request.data.context_codes for scheduler_request in requests):
        return None
    else:
        pass

    input_ids = forward_batch.input_ids
    device = input_ids.device
    embed_tokens = model.backbone.model.embed_tokens
    frame_embedding = model.frame_embedding

    text_embeds = embed_tokens(input_ids)
    eos_frame_embed: torch.Tensor | None = None

    offset = 0
    for scheduler_request in requests:
        request_data = scheduler_request.data
        end = offset + int(request_data.req.extend_len)
        chunk_ids = input_ids[offset:end]

        audio_indices = (chunk_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
        num_placeholders = int(audio_indices.numel())
        if num_placeholders > 0:
            codes = concat_context_codes(request_data, device)
            if codes is None:
                raise ValueError(
                    f"request {scheduler_request.request_id}: prompt has "
                    f"{num_placeholders} <|AUDIO|> placeholders but no "
                    f"context_codes (prompt/codes mismatch)"
                )
            else:
                pass
            consumed = int(request_data.num_context_codes_consumed)
            if consumed + num_placeholders > codes.shape[0]:
                raise ValueError(
                    f"request {scheduler_request.request_id}: placeholder count "
                    f"overruns context codes ({consumed} consumed + "
                    f"{num_placeholders} needed > {codes.shape[0]} frames)"
                )
            else:
                pass
            with torch.no_grad():
                paste = frame_embedding(codes[consumed : consumed + num_placeholders])
            text_embeds[audio_indices + offset] = paste.to(text_embeds.dtype)
            request_data.num_context_codes_consumed = consumed + num_placeholders
        else:
            pass

        eos_indices = (chunk_ids == AUDIO_EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
        if eos_indices.numel() > 0:
            if eos_frame_embed is None:
                with torch.no_grad():
                    eos_frame_embed = frame_embedding(
                        torch.zeros(1, NUM_CODEBOOKS, dtype=torch.long, device=device)
                    )
            else:
                pass
            text_embeds[eos_indices + offset] = eos_frame_embed.to(text_embeds.dtype)
        else:
            pass

        offset = end

    return text_embeds


__all__ = ["build_prefill_input_embeds"]
