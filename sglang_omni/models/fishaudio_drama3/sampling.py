# SPDX-License-Identifier: Apache-2.0
"""Token sampling for the Drama 3 audio heads: top-k / top-p and semantic RAS."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    FIRST_DRAW_STREAM,
    RAS_DRAW_STREAM,
    CodebookSampler,
    sample_codebook_head,
    sample_semantic_candidates,
)

# note (SunskyXH): RAS settings are fixed for Drama 3, not request parameters:
# when a semantic first draw repeats a main token from the last 16 frames, the
# frame takes a second draw with these parameters instead.
RAS_WINDOW_FRAMES = 16
# Pads the windows of requests with fewer than 16 frames; matches no token id.
EMPTY_RAS_SLOT = -1
RAS_TEMPERATURE = 1.0
RAS_TOP_P = 0.95
RAS_TOP_K = 30
MINIMUM_TEMPERATURE = 1e-5


def sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    sampler: CodebookSampler | None,
    stream_id: int,
    *,
    keep_top_p_crossing_token: bool = False,
) -> torch.Tensor:
    """Sample one token per row in fp32.

    Acoustic heads exclude the top-p crossing token. Semantic draws retain it;
    both policies respect top-k and always keep rank 0.
    """
    if sampler is not None:
        return sample_codebook_head(
            logits,
            temperature,
            top_k,
            top_p,
            sampler,
            stream_id,
            MINIMUM_TEMPERATURE,
            keep_top_p_crossing_token=keep_top_p_crossing_token,
        )
    else:
        pass
    scaled_logits = logits.float() / temperature.float().clamp(
        min=MINIMUM_TEMPERATURE
    ).unsqueeze(-1)
    sorted_logits, sorted_token_ids = torch.sort(scaled_logits, dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )
    ranks = torch.arange(sorted_logits.shape[-1], device=logits.device).unsqueeze(0)
    is_dropped = cumulative_probabilities > top_p.float().unsqueeze(-1)
    if keep_top_p_crossing_token:
        is_dropped[:, 1:] = is_dropped[:, :-1].clone()
    else:
        pass
    is_dropped |= ranks >= top_k.unsqueeze(-1)
    is_dropped[:, 0] = False
    probabilities = torch.softmax(
        sorted_logits.masked_fill(is_dropped, float("-inf")), dim=-1
    )
    exponential_noise = torch.empty_like(probabilities).exponential_(1)
    sorted_choice = torch.argmax(
        probabilities / exponential_noise, dim=-1, keepdim=True
    )
    return torch.gather(sorted_token_ids, -1, sorted_choice).squeeze(-1)


def sample_semantic_with_ras(
    semantic_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    previous_main_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    semantic_begin_id: int,
    codebook_size: int,
    sampler: CodebookSampler | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample each frame's main token and codebook-0 code.

    Both draws always run, so the random stream does not depend on the RAS window.
    """
    if previous_main_tokens.shape[-1] != RAS_WINDOW_FRAMES:
        raise ValueError(
            f"previous_main_tokens must hold the last {RAS_WINDOW_FRAMES} frames; "
            f"got shape {tuple(previous_main_tokens.shape)}"
        )
    elif sampler is not None:
        return sample_semantic_candidates(
            semantic_logits,
            candidate_token_ids,
            previous_main_tokens,
            temperature,
            top_k,
            top_p,
            semantic_begin_id,
            codebook_size,
            sampler,
            minimum_temperature=MINIMUM_TEMPERATURE,
            ras_temperature=RAS_TEMPERATURE,
            ras_top_k=RAS_TOP_K,
            ras_top_p=RAS_TOP_P,
        )
    else:
        pass
    candidate_logits = semantic_logits.index_select(-1, candidate_token_ids).float()
    first_draw = candidate_token_ids[
        sample_top_k_top_p(
            candidate_logits,
            temperature,
            top_k,
            top_p,
            None,
            FIRST_DRAW_STREAM,
            keep_top_p_crossing_token=True,
        )
    ]
    row_count = candidate_logits.shape[0]
    device = candidate_logits.device
    ras_draw = candidate_token_ids[
        sample_top_k_top_p(
            candidate_logits,
            torch.full(
                (row_count,), RAS_TEMPERATURE, dtype=torch.float32, device=device
            ),
            torch.full((row_count,), RAS_TOP_K, dtype=torch.int64, device=device),
            torch.full((row_count,), RAS_TOP_P, dtype=torch.float32, device=device),
            None,
            RAS_DRAW_STREAM,
            keep_top_p_crossing_token=True,
        )
    ]
    is_repeat = (previous_main_tokens == first_draw.unsqueeze(-1)).any(dim=-1)
    is_semantic = (first_draw >= semantic_begin_id) & (
        first_draw < semantic_begin_id + codebook_size
    )
    main_tokens = torch.where(is_repeat & is_semantic, ras_draw, first_draw)
    codebook_zero = (main_tokens - semantic_begin_id).clamp(0, codebook_size - 1)
    return main_tokens, codebook_zero
