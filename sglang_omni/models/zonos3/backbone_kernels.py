# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the Zonos3 backbone: the block-attnres residual's store and mix of sublayer outputs, and the MoE expert selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The values of one token, and the query logits of one value, fit one power-of-two block.
QUERY_BLOCK = 32
# note (zhaochenyang20): with these, the residual of a 14-token forward took 297 us against 700 us for the torch ops it replaces (both with the 22 sublayer matmuls, CUDA graph, one H100).
HIDDEN_CHUNK = 256
RESIDUAL_WARPS = 4
# note (zhaochenyang20): with 16 tokens per program, a 14-token router call took 13 us against 30 us with the torch top-k and softmax (CUDA graph, one H100).
ROUTER_TOKEN_BLOCK = 16


# note (zhaochenyang20): every stored sublayer output is mixed for its own query next, so one launch does both; the 23 pairs of a 13-token step took 156 us against 193 us as separate store and mix kernels, with bit-identical outputs at 1 to 1024 tokens (CUDA graph, one H100).
@triton.jit
def store_and_mix_kernel(
    value_pointer,
    values_pointer,
    queries_pointer,
    value_logits_pointer,
    norm_weight_pointer,
    output_pointer,
    value_index,
    value_row_stride,
    store_epsilon,
    mix_epsilon,
    HIDDEN_SIZE: tl.constexpr,
    QUERY_COUNT: tl.constexpr,
    BLOCK_QUERIES: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    """One token: values[token, value_index] = value in float32 and value_logits[token, value_index] = (value @ queries.T) / rms(value); then the softmax of query value_index's logits over the values stored so far weighs their sum, which leaves RMS-normed and scaled by norm_weight in the output's dtype."""
    token = tl.program_id(0).to(tl.int64)
    query_index = tl.arange(0, BLOCK_QUERIES)
    query_mask = query_index < QUERY_COUNT
    stored_row_pointer = (
        values_pointer + (token * QUERY_COUNT + value_index) * HIDDEN_SIZE
    )
    square_sums = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
    dot_products = tl.zeros((BLOCK_QUERIES,), dtype=tl.float32)
    for chunk_start in tl.static_range(0, HIDDEN_SIZE, BLOCK_HIDDEN):
        hidden_index = chunk_start + tl.arange(0, BLOCK_HIDDEN)
        value = tl.load(value_pointer + token * value_row_stride + hidden_index).to(
            tl.float32
        )
        tl.store(stored_row_pointer + hidden_index, value)
        square_sums += value * value
        queries = tl.load(
            queries_pointer
            + query_index[:, None] * HIDDEN_SIZE
            + hidden_index[None, :],
            mask=query_mask[:, None],
            other=0.0,
        )
        dot_products += tl.sum(queries * value[None, :], axis=1)
    inverse_rms = 1.0 / tl.sqrt_rn(
        tl.sum(square_sums, axis=0) / HIDDEN_SIZE + store_epsilon
    )
    tl.store(
        value_logits_pointer
        + (token * QUERY_COUNT + value_index) * QUERY_COUNT
        + query_index,
        dot_products * inverse_rms,
        mask=query_mask,
    )
    # The mix reads back through global memory the row and logits other threads of this program just stored.
    tl.debug_barrier()
    available = value_index + 1
    query_logits_pointer = (
        value_logits_pointer + token * QUERY_COUNT * QUERY_COUNT + value_index
    )
    logits = tl.load(
        query_logits_pointer + query_index * QUERY_COUNT,
        mask=query_index < available,
        other=float("-inf"),
    )
    largest_logit = tl.max(logits, axis=0)
    weight_sum = tl.sum(tl.exp(logits - largest_logit), axis=0)
    mixed_index = tl.arange(0, HIDDEN_SIZE)
    token_values_pointer = values_pointer + token * QUERY_COUNT * HIDDEN_SIZE
    mixed = tl.zeros((HIDDEN_SIZE,), dtype=tl.float32)
    # note (zhaochenyang20): a runtime-length loop waited out each row's load in turn; unrolled over every query with the unstored rows masked to zero weight, the loads overlap and the sum keeps its order: the 23 calls of a 12-token step took 126 us against 154 us (CUDA graph, one H100), bit-identical at 1 to 2048 tokens.
    for stored_index in tl.static_range(QUERY_COUNT):
        is_stored = stored_index < available
        stored_logit = tl.load(
            query_logits_pointer + stored_index * QUERY_COUNT,
            mask=is_stored,
            other=float("-inf"),
        )
        weight = tl.exp(stored_logit - largest_logit) / weight_sum
        mixed += weight * tl.load(
            token_values_pointer + stored_index * HIDDEN_SIZE + mixed_index,
            mask=is_stored,
            other=0.0,
        )
    mixed_inverse_rms = 1.0 / tl.sqrt_rn(
        tl.sum(mixed * mixed, axis=0) / HIDDEN_SIZE + mix_epsilon
    )
    normed = mixed * mixed_inverse_rms * tl.load(norm_weight_pointer + mixed_index)
    tl.store(
        output_pointer + token * HIDDEN_SIZE + mixed_index,
        normed.to(output_pointer.dtype.element_ty),
    )


@triton.jit
def select_experts_kernel(
    router_logits_pointer,
    balancing_biases_pointer,
    expert_weights_pointer,
    expert_ids_pointer,
    token_count,
    EXPERT_COUNT: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """BLOCK_TOKENS tokens: the TOP_K experts of the largest bias-corrected logits, in descending order, weighted by their softmax probabilities renormalized over the chosen experts."""
    token_index = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    token_mask = token_index < token_count
    expert_index = tl.arange(0, EXPERT_COUNT)
    logits = tl.load(
        router_logits_pointer
        + token_index.to(tl.int64)[:, None] * EXPERT_COUNT
        + expert_index[None, :],
        mask=token_mask[:, None],
        other=0.0,
    )
    probabilities = tl.exp(logits - tl.max(logits, axis=1)[:, None])
    probabilities = probabilities / tl.sum(probabilities, axis=1)[:, None]
    biased_logits = logits - tl.load(balancing_biases_pointer + expert_index)[None, :]
    expert_ranks = tl.full((BLOCK_TOKENS, EXPERT_COUNT), -1, dtype=tl.int32)
    for rank in tl.static_range(TOP_K):
        best_logits = tl.max(biased_logits, axis=1)
        best_experts = tl.min(
            tl.where(
                biased_logits == best_logits[:, None],
                expert_index[None, :],
                EXPERT_COUNT,
            ),
            axis=1,
        )
        is_best = expert_index[None, :] == best_experts[:, None]
        expert_ranks = tl.where(is_best, rank, expert_ranks)
        biased_logits = tl.where(is_best, float("-inf"), biased_logits)
    is_chosen = expert_ranks >= 0
    chosen_mass = tl.sum(tl.where(is_chosen, probabilities, 0.0), axis=1)
    chosen_offsets = token_index.to(tl.int64)[:, None] * TOP_K + expert_ranks
    chosen_mask = is_chosen & token_mask[:, None]
    tl.store(
        expert_weights_pointer + chosen_offsets,
        probabilities / chosen_mass[:, None],
        mask=chosen_mask,
    )
    tl.store(
        expert_ids_pointer + chosen_offsets,
        expert_index[None, :] + tl.zeros_like(expert_ranks),
        mask=chosen_mask,
    )


def select_experts(
    router_logits: torch.Tensor, balancing_biases: torch.Tensor, *, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 expert weights and int32 expert ids [tokens, top_k] of float32 router_logits [tokens, experts]."""
    token_count, expert_count = router_logits.shape
    assert router_logits.is_contiguous()
    assert expert_count == triton.next_power_of_2(expert_count)
    expert_weights = torch.empty(
        (token_count, top_k), dtype=torch.float32, device=router_logits.device
    )
    expert_ids = torch.empty(
        (token_count, top_k), dtype=torch.int32, device=router_logits.device
    )
    select_experts_kernel[(triton.cdiv(token_count, ROUTER_TOKEN_BLOCK),)](
        router_logits,
        balancing_biases,
        expert_weights,
        expert_ids,
        token_count,
        EXPERT_COUNT=expert_count,
        TOP_K=top_k,
        BLOCK_TOKENS=ROUTER_TOKEN_BLOCK,
        num_warps=RESIDUAL_WARPS,
    )
    return expert_weights, expert_ids


def store_and_mix_values(
    value: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    value_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    value_index: int,
    store_epsilon: float,
    mix_epsilon: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Stores value [tokens, hidden] in float32 at values[:, value_index] and its logits against queries [queries, hidden] at value_logits[:, value_index], then returns the RMS-normed softmax mix [tokens, hidden] of values[:, : value_index + 1] for query value_index; values [tokens, queries, hidden] and value_logits [tokens, queries, queries] are float32."""
    token_count, query_count, hidden_size = values.shape
    assert value.stride(1) == 1 and values.is_contiguous() and queries.is_contiguous()
    assert query_count <= QUERY_BLOCK and hidden_size % HIDDEN_CHUNK == 0
    assert hidden_size == triton.next_power_of_2(hidden_size)
    output = torch.empty(
        (token_count, hidden_size), dtype=output_dtype, device=values.device
    )
    store_and_mix_kernel[(token_count,)](
        value,
        values,
        queries,
        value_logits,
        norm_weight,
        output,
        value_index,
        value.stride(0),
        store_epsilon,
        mix_epsilon,
        HIDDEN_SIZE=hidden_size,
        QUERY_COUNT=query_count,
        BLOCK_QUERIES=QUERY_BLOCK,
        BLOCK_HIDDEN=HIDDEN_CHUNK,
        num_warps=RESIDUAL_WARPS,
    )
    return output
