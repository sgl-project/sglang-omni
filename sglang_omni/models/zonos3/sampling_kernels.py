# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the Zonos3 sampler tail: the code draw per (request, codebook) row and the frame embedding sum."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Candidates a row keeps before top-k and top-p cut it: the power of two at or above the largest top-k it serves, at least this many.
MIN_CANDIDATE_COUNT = 128
SAMPLE_WARPS = 8
EMBEDDING_BLOCK = 256
# A repetition-window entry that holds no code: the window has not filled yet, or the row ended audio there.
NO_CODE = -1
# The smallest normal float32; libdevice tanh flushes a subnormal input to zero, where torch's tanh returns it.
SMALLEST_NORMAL_FLOAT32 = tl.constexpr(2.0**-126)


@triton.jit
def orderable_bits(values):
    """Int32 keys that order like the float32 values."""
    bits = values.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def float_from_orderable_bits(keys):
    bits = tl.where(keys >= 0, keys, keys ^ 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def soft_capped(logits, softcap, inverse_softcap):
    """softcap * tanh(logits / softcap) in float32, rounded like torch's multiply by the reciprocal, tanh and multiply."""
    scaled = logits * inverse_softcap
    tanh = tl.where(
        tl.abs(scaled) < SMALLEST_NORMAL_FLOAT32, scaled, libdevice.tanh(scaled)
    )
    return softcap * tanh


@triton.jit
def sample_code_rows_kernel(
    logits_pointer,
    slots_pointer,
    knobs_pointer,
    top_k_pointer,
    seeds_pointer,
    steps_pointer,
    windows_pointer,
    window_counts_pointer,
    codes_pointer,
    vocabulary_size,
    codebook_count,
    slot_count,
    pad_id,
    codebook_size,
    softcap,
    inverse_softcap,
    NO_CODE_ID: tl.constexpr,
    WINDOW_ROWS: tl.constexpr,
    PENALIZED_CODEBOOKS: tl.constexpr,
    BLOCK_VOCABULARY: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    SPECIAL_IDS: tl.constexpr,
    CANDIDATES: tl.constexpr,
):
    """One (request, codebook) row, with the request's knobs read from its slot: the logits soft-capped, the pad code masked, the repetition penalty on the codes the slot's window holds, temperature (0 takes the best code), top-k, top-p, and the draw argmax(probability / noise) over the candidates in rank order, with Exp(1) noise from the slot's seeded Philox stream at (step, codebook, rank); then the drawn code replaces the window's oldest row.

    The window is a ring of the last WINDOW_ROWS codes plus a count per code, so the penalty reads one row of counts. The leading BLOCK_HEAD ids rank in one block and the SPECIAL_IDS ids after them merge into its top keys one by one; the candidates keep the rank order, and so the draws, of ranking every id at once.
    """
    row = tl.program_id(0)
    request = row // codebook_count
    codebook = row % codebook_count
    slot = tl.load(slots_pointer + request)
    temperature = tl.load(knobs_pointer + slot)
    top_p = tl.load(knobs_pointer + slot_count + slot)
    repetition_penalty = tl.load(knobs_pointer + 2 * slot_count + slot)
    top_k = tl.load(top_k_pointer + slot)
    seed = tl.load(seeds_pointer + slot)
    step = tl.load(steps_pointer + slot)
    row_logits_pointer = logits_pointer + row.to(tl.int64) * vocabulary_size
    head_index = tl.arange(0, BLOCK_HEAD)
    head_logits = tl.load(
        row_logits_pointer + head_index, mask=head_index < vocabulary_size, other=0.0
    ).to(tl.float32)
    head_logits = soft_capped(head_logits, softcap, inverse_softcap)
    head_logits = tl.where(
        (head_index < vocabulary_size) & (head_index != pad_id),
        head_logits,
        float("-inf"),
    )
    window_index = slot * PENALIZED_CODEBOOKS + codebook
    window_counts_row_pointer = window_counts_pointer + window_index * codebook_size
    is_penalized_codebook = codebook < PENALIZED_CODEBOOKS
    if is_penalized_codebook & (repetition_penalty != 1.0):
        window_counts = tl.load(
            window_counts_row_pointer + head_index,
            mask=head_index < codebook_size,
            other=0,
        )
        head_logits = tl.where(
            window_counts > 0,
            tl.where(
                head_logits > 0,
                head_logits / repetition_penalty,
                head_logits * repetition_penalty,
            ),
            head_logits,
        )
    else:
        pass
    head_scaled = tl.where(temperature > 0, head_logits / temperature, head_logits)
    head_keys = (orderable_bits(head_scaled).to(tl.int64) << 32) | (
        BLOCK_VOCABULARY - 1 - head_index
    ).to(tl.int64)
    candidate_keys = tl.topk(head_keys, CANDIDATES)
    candidate_index = tl.arange(0, CANDIDATES)
    # Each id past the head (end-of-audio and pad, never penalized) enters the ranked keys at its rank.
    for special_offset in tl.static_range(SPECIAL_IDS):
        special_id = BLOCK_HEAD + special_offset
        special_logit = tl.load(row_logits_pointer + special_id).to(tl.float32)
        special_logit = soft_capped(special_logit, softcap, inverse_softcap)
        special_logit = tl.where(special_id == pad_id, float("-inf"), special_logit)
        special_scaled = tl.where(
            temperature > 0, special_logit / temperature, special_logit
        )
        special_key = (orderable_bits(special_scaled).to(tl.int64) << 32) | (
            BLOCK_VOCABULARY - 1 - special_id
        )
        higher_keys = tl.gather(
            candidate_keys, tl.maximum(candidate_index - 1, 0), axis=0
        )
        displaced_keys = tl.where(
            candidate_index == 0, special_key, tl.minimum(special_key, higher_keys)
        )
        candidate_keys = tl.maximum(candidate_keys, displaced_keys)
    candidate_codes = BLOCK_VOCABULARY - 1 - (candidate_keys & 0xFFFFFFFF)
    candidate_values = float_from_orderable_bits((candidate_keys >> 32).to(tl.int32))
    candidate_values = tl.where(
        candidate_index < top_k, candidate_values, float("-inf")
    )
    probabilities = tl.exp(candidate_values - tl.max(candidate_values, axis=0))
    probabilities = probabilities / tl.sum(probabilities, axis=0)
    mass_before = tl.cumsum(probabilities, axis=0) - probabilities
    probabilities = tl.where(mass_before > top_p, 0.0, probabilities)
    # The rank stride is the vocabulary block, so a slot's stream does not depend on the batch's candidate count.
    noise = -tl.log(
        1.0
        - tl.rand(
            seed,
            (step * codebook_count + codebook) * BLOCK_VOCABULARY
            + candidate_index.to(tl.int64),
        )
    )
    scores = tl.where(probabilities > 0, probabilities / noise, -1.0)
    choice = tl.where(temperature > 0, tl.argmax(scores, axis=0), 0)
    code = tl.sum(tl.where(candidate_index == choice, candidate_codes, 0), axis=0)
    tl.store(codes_pointer + row, code)
    if is_penalized_codebook:
        ring_pointer = windows_pointer + window_index * WINDOW_ROWS + step % WINDOW_ROWS
        evicted_code = tl.load(ring_pointer)
        entered_code = tl.where(code < codebook_size, code, NO_CODE_ID)
        tl.atomic_add(
            window_counts_row_pointer + tl.maximum(evicted_code, 0),
            -1,
            mask=evicted_code != NO_CODE_ID,
        )
        tl.atomic_add(
            window_counts_row_pointer + tl.maximum(entered_code, 0),
            1,
            mask=entered_code != NO_CODE_ID,
        )
        tl.store(ring_pointer, entered_code)
    else:
        pass


@triton.jit
def frame_embedding_kernel(
    codes_pointer,
    table_addresses_pointer,
    text_row_pointer,
    slots_pointer,
    steps_pointer,
    output_pointer,
    hidden_size,
    codebook_count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum a sampled row's codebook embeddings and the text pad embedding in float32, in the codebooks' order, and store the bfloat16 sum in the request slot's row of the output table; the slot's generated row count advances."""
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < hidden_size
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for codebook in tl.static_range(codebook_count):
        table = tl.load(table_addresses_pointer + codebook).to(
            tl.pointer_type(tl.bfloat16)
        )
        code = tl.load(codes_pointer + request * codebook_count + codebook)
        embedding = tl.load(table + code * hidden_size + offsets, mask=mask).to(
            tl.float32
        )
        if codebook == 0:
            total = embedding
        else:
            total = total + embedding
    total = total + tl.load(text_row_pointer + offsets, mask=mask).to(tl.float32)
    slot = tl.load(slots_pointer + request)
    tl.store(
        output_pointer + slot * hidden_size + offsets,
        total.to(tl.bfloat16),
        mask=mask,
    )
    if tl.program_id(1) == 0:
        tl.store(steps_pointer + slot, tl.load(steps_pointer + slot) + 1)
    else:
        pass


def sample_code_rows(
    logits: torch.Tensor,
    slots: torch.Tensor,
    knobs: torch.Tensor,
    top_k: torch.Tensor,
    seeds: torch.Tensor,
    steps: torch.Tensor,
    windows: torch.Tensor,
    window_counts: torch.Tensor,
    *,
    max_top_k: int,
    pad_id: int,
    codebook_size: int,
    softcap: float,
) -> torch.Tensor:
    """Codes [requests, codebooks] drawn from logits [requests, codebooks, vocabulary], soft-capped first.

    Per-slot state, indexed by slots [requests]: knobs [3, slots] holds temperature, top-p and repetition penalty, top_k [slots] the top-k, seeds [slots] the Philox seed, steps [slots] the generated row count, windows [slots, penalized codebooks, window rows] the latest codes of the leading codebooks as a ring (NO_CODE where none), and window_counts [slots, penalized codebooks, codebook size] int32 how often the ring holds each code; the draw advances both.
    """
    request_count, codebook_count, vocabulary_size = logits.shape
    candidate_count = max(MIN_CANDIDATE_COUNT, triton.next_power_of_2(max_top_k))
    vocabulary_block = triton.next_power_of_2(vocabulary_size)
    # A power-of-two codebook ranks alone and the special ids past it merge in, which halves the ranked block.
    is_split = (
        codebook_size == triton.next_power_of_2(codebook_size)
        and candidate_count <= codebook_size
    )
    head_block = codebook_size if is_split else vocabulary_block
    codes = torch.empty(
        (request_count, codebook_count), dtype=torch.int64, device=logits.device
    )
    sample_code_rows_kernel[(request_count * codebook_count,)](
        logits,
        slots,
        knobs,
        top_k,
        seeds,
        steps,
        windows,
        window_counts,
        codes,
        vocabulary_size,
        codebook_count,
        knobs.shape[1],
        pad_id,
        codebook_size,
        # note (zhaochenyang20): the reciprocal mirrors torch's division by a scalar; every bfloat16 logit caps bit-identically to the float, divide, tanh and multiply ops this replaces.
        softcap,
        1.0 / softcap,
        NO_CODE_ID=NO_CODE,
        WINDOW_ROWS=windows.shape[2],
        PENALIZED_CODEBOOKS=windows.shape[1],
        BLOCK_VOCABULARY=vocabulary_block,
        BLOCK_HEAD=head_block,
        SPECIAL_IDS=vocabulary_size - head_block if is_split else 0,
        CANDIDATES=candidate_count,
        num_warps=SAMPLE_WARPS,
    )
    return codes


def store_frame_embeddings(
    codes: torch.Tensor,
    table_addresses: torch.Tensor,
    text_row: torch.Tensor,
    slots: torch.Tensor,
    steps: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Write each request's frame embedding (its codes' embeddings plus text_row, summed in float32) to its slot's row of output, and advance its slot's steps."""
    request_count, codebook_count = codes.shape
    hidden_size = output.shape[1]
    frame_embedding_kernel[(request_count, triton.cdiv(hidden_size, EMBEDDING_BLOCK))](
        codes,
        table_addresses,
        text_row,
        slots,
        steps,
        output,
        hidden_size,
        codebook_count=codebook_count,
        BLOCK=EMBEDDING_BLOCK,
    )
