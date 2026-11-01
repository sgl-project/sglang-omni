# SPDX-License-Identifier: Apache-2.0
"""Drama 3 reference sampling in plain PyTorch, with epsilon sampling (Hewitt et al.,
2022) on the main token."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_drama3.config import ReferenceSamplingFallback
from sglang_omni.models.fishaudio_drama3.sampling import (
    MINIMUM_TEMPERATURE,
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
)

# Codes per residual codebook of the codec (quantizer codebook_size in the S2-Pro
# modded_dac_vq.yaml); the codebook heads are wider.
RESIDUAL_CODEBOOK_SIZE = 1024


def row_values(
    value: float | torch.Tensor,
    row_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """One value per row: a per-row tensor in dtype, or a number filled on the device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    else:
        return torch.full((row_count,), value, dtype=dtype, device=device)


def truncate_candidates(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    top_k: int | torch.Tensor,
    top_p: float | torch.Tensor,
    epsilon_cutoff: float | torch.Tensor,
    legal_column_count: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scaled logits sorted in descending order, their columns, and the kept mask in
    that order."""
    row_count, column_count = logits.shape
    device = logits.device
    temperature_rows = row_values(temperature, row_count, torch.float32, device)
    top_k_limits = row_values(top_k, row_count, torch.int64, device).unsqueeze(-1)
    top_p_limits = row_values(top_p, row_count, torch.float32, device).unsqueeze(-1)
    epsilon_cutoffs = row_values(
        epsilon_cutoff, row_count, torch.float32, device
    ).unsqueeze(-1)

    scaled_logits = logits.float() / temperature_rows.clamp(
        min=MINIMUM_TEMPERATURE
    ).unsqueeze(-1)
    sorted_logits, sorted_columns = torch.sort(
        scaled_logits, dim=-1, descending=True, stable=True
    )
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(probabilities, dim=-1)
    ranks = torch.arange(column_count, device=device).unsqueeze(0)

    is_within_top_k = (ranks < top_k_limits) | (top_k_limits < 1)
    is_within_top_p = (cumulative_probabilities <= top_p_limits) | (top_p_limits >= 1.0)
    is_above_cutoff = probabilities >= epsilon_cutoffs
    is_kept = is_within_top_k & is_within_top_p & is_above_cutoff
    if legal_column_count is None:
        is_kept[:, 0] = True
    else:
        is_legal = sorted_columns < legal_column_count
        is_kept &= is_legal
        is_kept.scatter_(-1, is_legal.int().argmax(dim=-1, keepdim=True), True)
    return sorted_logits, sorted_columns, is_kept


def truncated_sample(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    top_k: int | torch.Tensor,
    top_p: float | torch.Tensor,
    epsilon_cutoff: float | torch.Tensor,
    *,
    legal_column_count: int | None = None,
) -> torch.Tensor:
    """One column per row. Rank r stays when r < top_k and (r == 0 or (cum[r] <= top_p
    and p[r] >= epsilon_cutoff)); top_k < 1, top_p >= 1 and epsilon_cutoff <= 0 turn
    their rule off. legal_column_count then drops later columns, keeping the best legal
    one."""
    sorted_logits, sorted_columns, is_kept = truncate_candidates(
        logits, temperature, top_k, top_p, epsilon_cutoff, legal_column_count
    )
    kept_probabilities = torch.softmax(
        sorted_logits.masked_fill(~is_kept, float("-inf")), dim=-1
    )
    # The argmax of p / E with E ~ Exp(1) per column is a draw from p.
    exponential_noise = torch.empty_like(kept_probabilities).exponential_(1.0)
    sorted_choice = torch.argmax(
        kept_probabilities / exponential_noise, dim=-1, keepdim=True
    )
    return torch.gather(sorted_columns, -1, sorted_choice).squeeze(-1)


def sample_main_tokens(
    semantic_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    previous_main_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    epsilon_cutoff: torch.Tensor,
    semantic_begin_id: int,
    codebook_size: int,
    fallback: ReferenceSamplingFallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each frame's main token and codebook-0 code. A first draw found among
    previous_main_tokens takes the fallback, the most probable candidate (argmax) or a
    draw at 1.0 / 0.95 / 30 (ras); neither applies the epsilon cutoff."""
    if semantic_logits.shape[-1] == candidate_token_ids.shape[0]:
        candidate_logits = semantic_logits
    else:
        candidate_logits = semantic_logits.index_select(-1, candidate_token_ids)
    first_draw = candidate_token_ids[
        truncated_sample(candidate_logits, temperature, top_k, top_p, epsilon_cutoff)
    ]
    if fallback == "argmax":
        fallback_tokens = candidate_token_ids[candidate_logits.float().argmax(dim=-1)]
    elif fallback == "ras":
        fallback_tokens = candidate_token_ids[
            truncated_sample(
                candidate_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, 0.0
            )
        ]
    else:
        raise ValueError(f"fallback must be 'argmax' or 'ras'; got {fallback!r}")
    is_repeat = (previous_main_tokens == first_draw.unsqueeze(-1)).any(dim=-1)
    main_tokens = torch.where(is_repeat, fallback_tokens, first_draw)
    codebook_zero = (main_tokens - semantic_begin_id).clamp(0, codebook_size - 1)
    return main_tokens, codebook_zero


def sample_residual_codes(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    """One code per row of a codebook head: the request's temperature, top-k and top-p,
    no epsilon cutoff, and only codes the codec's residual codebooks hold."""
    return truncated_sample(
        logits,
        temperature,
        top_k,
        top_p,
        0.0,
        legal_column_count=RESIDUAL_CODEBOOK_SIZE,
    )
