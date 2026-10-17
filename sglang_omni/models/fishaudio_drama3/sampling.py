# SPDX-License-Identifier: Apache-2.0
"""Token sampling for the Drama 3 audio heads: top-k / top-p and semantic RAS."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    FIRST_DRAW_STREAM,
    RAS_DRAW_STREAM,
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
PHILOX_ROUNDS = 10
PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85
PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57
UINT32_MASK = 0xFFFFFFFF
UNIFORM_MANTISSA_BITS = 23


def multiply_high_and_low(
    multiplier: int, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """High and low 32 bits of multiplier * values for uint32 values held in int64."""
    # note (cipheraxat): the product can pass 2**63; int64 multiply wraps, so its
    # low 64 bits, which are all that is read, stay exact.
    product = values * multiplier
    return (product >> 32) & UINT32_MASK, product & UINT32_MASK


def philox_exponential_noise(
    seeds: torch.Tensor,
    positions: torch.Tensor,
    stream_id: int,
    candidate_count: int,
) -> torch.Tensor:
    """Exp(1) noise [rows, candidate_count] in fp32 from Philox-4x32-10 keyed on each
    row's seed and position, so a draw does not depend on the batch around it.

    The counters and the uniform mapping follow the Triton kernels, which draw from
    the same stream.
    """
    row_count = seeds.shape[0]
    shape = (row_count, candidate_count)
    counter_0 = torch.arange(candidate_count, device=seeds.device).expand(shape)
    counter_1 = torch.zeros(shape, dtype=torch.int64, device=seeds.device)
    counter_2 = torch.full(shape, stream_id, dtype=torch.int64, device=seeds.device)
    counter_3 = (positions & UINT32_MASK).unsqueeze(-1).expand(shape)
    key_0 = (seeds & UINT32_MASK).unsqueeze(-1)
    key_1 = ((seeds >> 32) & UINT32_MASK).unsqueeze(-1)
    for _ in range(PHILOX_ROUNDS):
        high_b, low_b = multiply_high_and_low(PHILOX_ROUND_B, counter_2)
        high_a, low_a = multiply_high_and_low(PHILOX_ROUND_A, counter_0)
        counter_0 = high_b ^ counter_1 ^ key_0
        counter_2 = high_a ^ counter_3 ^ key_1
        counter_1 = low_b
        counter_3 = low_a
        key_0 = (key_0 + PHILOX_KEY_A) & UINT32_MASK
        key_1 = (key_1 + PHILOX_KEY_B) & UINT32_MASK
    uniform = ((counter_0 >> (32 - UNIFORM_MANTISSA_BITS)).float() + 0.5) * (
        1.0 / (1 << UNIFORM_MANTISSA_BITS)
    )
    return -torch.log(uniform)


def sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    use_triton_kernels: bool,
    stream_id: int,
) -> torch.Tensor:
    """Sample one token per row in fp32, drawing from the row's seed and position.

    Drop ranks >= top_k and tokens whose inclusive cumulative probability exceeds
    top_p; always keep rank 0.
    """
    if use_triton_kernels:
        return sample_codebook_head(
            logits,
            temperature,
            top_k,
            top_p,
            seeds,
            positions,
            stream_id,
            MINIMUM_TEMPERATURE,
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
    is_dropped = (cumulative_probabilities > top_p.float().unsqueeze(-1)) | (
        ranks >= top_k.unsqueeze(-1)
    )
    is_dropped[:, 0] = False
    probabilities = torch.softmax(
        sorted_logits.masked_fill(is_dropped, float("-inf")), dim=-1
    )
    sorted_noise = torch.gather(
        philox_exponential_noise(seeds, positions, stream_id, logits.shape[-1]),
        -1,
        sorted_token_ids,
    )
    sorted_choice = torch.argmax(probabilities / sorted_noise, dim=-1, keepdim=True)
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
    seeds: torch.Tensor,
    positions: torch.Tensor,
    use_triton_kernels: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample each frame's main token and codebook-0 code.

    Both draws always run, so the random stream does not depend on the RAS window.
    """
    if previous_main_tokens.shape[-1] != RAS_WINDOW_FRAMES:
        raise ValueError(
            f"previous_main_tokens must hold the last {RAS_WINDOW_FRAMES} frames; "
            f"got shape {tuple(previous_main_tokens.shape)}"
        )
    elif use_triton_kernels:
        return sample_semantic_candidates(
            semantic_logits,
            candidate_token_ids,
            previous_main_tokens,
            temperature,
            top_k,
            top_p,
            semantic_begin_id,
            codebook_size,
            seeds,
            positions,
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
            seeds,
            positions,
            False,
            FIRST_DRAW_STREAM,
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
            seeds,
            positions,
            False,
            RAS_DRAW_STREAM,
        )
    ]
    is_repeat = (previous_main_tokens == first_draw.unsqueeze(-1)).any(dim=-1)
    is_semantic = (first_draw >= semantic_begin_id) & (
        first_draw < semantic_begin_id + codebook_size
    )
    main_tokens = torch.where(is_repeat & is_semantic, ras_draw, first_draw)
    codebook_zero = (main_tokens - semantic_begin_id).clamp(0, codebook_size - 1)
    return main_tokens, codebook_zero
