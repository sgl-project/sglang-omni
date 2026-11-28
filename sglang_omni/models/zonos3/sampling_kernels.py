# SPDX-License-Identifier: Apache-2.0
"""Sampler tail: the code draw per (request, codebook) row — a triton kernel on CUDA, its torch twin elsewhere — and the frame embedding sum."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang_omni.platforms import current_platform

# Candidates a row keeps before top-k and top-p cut it: the power of two at or above the largest top-k it serves, at least this many.
MIN_CANDIDATE_COUNT = 128
SAMPLE_WARPS = 8
EMBEDDING_BLOCK = 256
# A repetition-window entry that holds no code: the window has not filled yet, or the row ended audio there.
NO_CODE = -1
# The smallest normal float32; libdevice tanh flushes a subnormal input to zero, where torch's tanh returns it.
SMALLEST_NORMAL_FLOAT32 = tl.constexpr(2.0**-126)


_STATIC_CACHE: dict[tuple, torch.Tensor] = {}


def _cached(key, factory):  # noqa: leading-underscore
    tensor = _STATIC_CACHE.get(key)
    if tensor is None:
        tensor = factory()
        _STATIC_CACHE[key] = tensor
    else:
        pass
    return tensor


def _pad_bias(device, vocabulary_size, pad_id):  # noqa: leading-underscore
    """0 except -inf at the pad id: one add masks the pad code where an index_put_ costs an AiCpu task per step."""

    def build():
        bias = torch.zeros(vocabulary_size, dtype=torch.float32, device=device)
        bias[pad_id] = float("-inf")
        return bias

    return _cached(("pad", device, vocabulary_size, pad_id), build)


def _rank_row(device, keep):  # noqa: leading-underscore
    return _cached(
        ("rank", device, keep),
        lambda: torch.arange(keep, device=device).view(1, 1, keep),
    )


def _negative_inf_row(device, keep):  # noqa: leading-underscore
    return _cached(
        ("negative_inf", device, keep),
        lambda: torch.full((keep,), float("-inf"), device=device),
    )


def zero_scalar(device):
    return _cached(("zero", device), lambda: torch.zeros((1,), device=device))


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
    # note: triton-ascend has neither tl.topk nor tl.sort nor libdevice.tanh, so there the ranked block runs as an argmax-loop kernel that draws exactly what the torch fallback draws.
    if current_platform.is_cuda():
        return sample_code_rows_triton(
            logits,
            slots,
            knobs,
            top_k,
            seeds,
            steps,
            windows,
            window_counts,
            max_top_k=max_top_k,
            pad_id=pad_id,
            codebook_size=codebook_size,
            softcap=softcap,
        )
    else:
        pass
    if current_platform.is_npu():
        return sample_code_rows_npu(
            logits,
            slots,
            knobs,
            top_k,
            seeds,
            steps,
            windows,
            window_counts,
            max_top_k=max_top_k,
            pad_id=pad_id,
            codebook_size=codebook_size,
            softcap=softcap,
        )
    else:
        pass
    return sample_code_rows_torch(
        logits,
        slots,
        knobs,
        top_k,
        seeds,
        steps,
        windows,
        window_counts,
        max_top_k=max_top_k,
        pad_id=pad_id,
        codebook_size=codebook_size,
        softcap=softcap,
    )


def sample_code_rows_triton(
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
    """Ranked-block draw: every row keeps the leading candidates, then top-k, top-p and an exponential race pick one."""
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


@triton.jit
def sample_code_rows_npu_kernel(
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
    BLOCK_HEAD: tl.constexpr,
    BLOCK_VOCABULARY: tl.constexpr,
    SPECIAL_IDS: tl.constexpr,
    CANDIDATES: tl.constexpr,
):
    """The ranked block of the CUDA kernel without tl.topk, tl.sort or libdevice.tanh: the leading ids rank by an argmax loop over float32 values (int32 argmax is far slower on this backend), and the special ids merge in as the CUDA kernel does.

    The draw, the hash noise and the ring update match sample_code_rows_torch bit for bit, so the NPU path keeps the same codes the torch fallback drew.
    """
    row = tl.program_id(0)
    request = row // codebook_count
    codebook = row % codebook_count
    slot = tl.load(slots_pointer + request)
    temperature = tl.load(knobs_pointer + slot)
    top_p = tl.load(knobs_pointer + slot_count + slot)
    repetition_penalty = tl.load(knobs_pointer + 2 * slot_count + slot)
    top_k = tl.load(top_k_pointer + slot).to(tl.int32)
    seed = tl.load(seeds_pointer + slot)
    step = tl.load(steps_pointer + slot)

    head_index = tl.arange(0, BLOCK_HEAD)
    head = tl.load(
        logits_pointer + row.to(tl.int64) * vocabulary_size + head_index,
        mask=head_index < vocabulary_size,
        other=0.0,
    ).to(tl.float32)
    # tanh by exp: 2 / (1 + exp(-2x)) - 1, which caps every bfloat16 logit as torch's tanh does.
    scaled = head * inverse_softcap
    capped = softcap * (2.0 / (1.0 + tl.exp(-2.0 * scaled)) - 1.0)
    capped = tl.where(head_index < vocabulary_size, capped, float("-inf"))

    window_index = slot * PENALIZED_CODEBOOKS + codebook
    counts_row_pointer = window_counts_pointer + window_index * codebook_size
    is_penalized = codebook < PENALIZED_CODEBOOKS
    if is_penalized & (repetition_penalty != 1.0):
        counts = tl.load(
            counts_row_pointer + head_index, mask=head_index < codebook_size, other=0
        )
        penalized = tl.where(
            capped > 0, capped / repetition_penalty, capped * repetition_penalty
        )
        capped = tl.where(counts > 0, penalized, capped)
    else:
        pass
    values = tl.where(temperature > 0, capped / temperature, capped)

    koffs = tl.arange(0, CANDIDATES)
    sel_v = tl.full((CANDIDATES,), float("-inf"), tl.float32)
    sel_c = tl.zeros((CANDIDATES,), tl.float32)
    working = values
    bound = tl.minimum(top_k, CANDIDATES)
    for i in tl.range(0, bound):
        # z3fused: one reduction for value+index instead of two
        k, c = tl.max(working, axis=0, return_indices=True)
        sel_v = tl.where(koffs == i, k, sel_v)
        sel_c = tl.where(koffs == i, c.to(tl.float32), sel_c)
        working = tl.where(head_index == c, float("-inf"), working)

    # Special ids past the head (end of audio, pad) insert into the ranked candidates; ties keep the head code, as (V - 1 - id) would.
    base = BLOCK_HEAD
    for special_offset in tl.static_range(SPECIAL_IDS):
        special_id = base + special_offset
        special = tl.load(
            logits_pointer + row.to(tl.int64) * vocabulary_size + special_id
        ).to(tl.float32)
        special_scaled = special * inverse_softcap
        special = softcap * (2.0 / (1.0 + tl.exp(-2.0 * special_scaled)) - 1.0)
        if is_penalized & (repetition_penalty != 1.0) & (special_id < codebook_size):
            special_count = tl.load(counts_row_pointer + special_id)
            special_penalized = tl.where(
                special > 0,
                special / repetition_penalty,
                special * repetition_penalty,
            )
            special = tl.where(special_count > 0, special_penalized, special)
        else:
            pass
        special = tl.where(temperature > 0, special / temperature, special)
        special = tl.where(special_id == pad_id, float("-inf"), special)
        position = tl.sum(tl.where(sel_v >= special, 1, 0), axis=0)
        prev_v = tl.gather(sel_v, tl.maximum(koffs - 1, 0), axis=0)
        prev_c = tl.gather(sel_c, tl.maximum(koffs - 1, 0), axis=0)
        sel_v = tl.where(
            koffs < position, sel_v, tl.where(koffs == position, special, prev_v)
        )
        sel_c = tl.where(
            koffs < position,
            sel_c,
            tl.where(koffs == position, special_id.to(tl.float32), prev_c),
        )
    else:
        pass

    values = tl.where(koffs < top_k, sel_v, float("-inf"))
    probabilities = tl.exp(values - tl.max(values, axis=0))
    probabilities = probabilities / tl.sum(probabilities, axis=0)
    mass_before = tl.cumsum(probabilities, axis=0) - probabilities
    probabilities = tl.where(mass_before > top_p, 0.0, probabilities)

    # The same integer hash of (seed, step, codebook, rank) the torch fallback uses for its noise; the rank stride is the vocabulary block, so a slot's stream does not depend on the batch's candidate count.
    stream = (step * codebook_count + codebook).to(tl.int64) * BLOCK_VOCABULARY
    mixed = (
        (koffs.to(tl.int64) + 1) * 0x9E3779B1
        + seed.to(tl.int64) * 0x85EBCA6B
        + stream * 0x27D4EB2F
    )
    mixed = mixed ^ (mixed >> 16)
    mixed = mixed * 0x85EBCA6B
    mixed = mixed ^ (mixed >> 13)
    uniform = (mixed & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)
    noise = -tl.log(1.0 - uniform)
    scores = tl.where(probabilities > 0, probabilities / noise, -1.0)
    choice = tl.where(temperature > 0, tl.argmax(scores, axis=0), 0)
    code = tl.sum(tl.where(koffs == choice, sel_c, 0.0), axis=0).to(tl.int64)
    tl.store(codes_pointer + row, code)

    if is_penalized:
        ring_pointer = windows_pointer + window_index * WINDOW_ROWS + step % WINDOW_ROWS
        evicted = tl.load(ring_pointer)
        entered = tl.where(code < codebook_size, code, NO_CODE_ID)
        # The mask is not applied on this backend, so a skipped update adds zero at index 0 instead.
        tl.atomic_add(
            counts_row_pointer + tl.maximum(evicted, 0),
            tl.where(evicted != NO_CODE_ID, -1, 0),
        )
        tl.atomic_add(
            counts_row_pointer + tl.maximum(entered, 0),
            tl.where(entered != NO_CODE_ID, 1, 0),
        )
        tl.store(ring_pointer, entered)
    else:
        pass


def sample_code_rows_npu(
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
    """One triton program per (request, codebook) row on the NPU: the ranked block of the CUDA kernel rebuilt from argmax, exp and gather."""
    request_count, codebook_count, vocabulary_size = logits.shape
    if request_count == 0:
        return torch.empty((0, codebook_count), dtype=torch.int64, device=logits.device)
    else:
        pass
    codes = torch.empty(
        (request_count, codebook_count), dtype=torch.int64, device=logits.device
    )
    sample_code_rows_npu_kernel[(request_count * codebook_count,)](
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
        softcap,
        1.0 / softcap,
        NO_CODE_ID=NO_CODE,
        WINDOW_ROWS=windows.shape[2],
        PENALIZED_CODEBOOKS=windows.shape[1],
        BLOCK_HEAD=codebook_size,
        BLOCK_VOCABULARY=triton.next_power_of_2(vocabulary_size),
        SPECIAL_IDS=vocabulary_size - codebook_size,
        CANDIDATES=max(MIN_CANDIDATE_COUNT, triton.next_power_of_2(max_top_k)),
    )
    return codes


def sample_code_rows_torch(
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
    """The ranked block of the kernel without triton: top-k, top-p and the same exponential race over per-row noise.

    The noise is an integer hash of (seed, step, codebook, rank) instead of the slot's Philox stream, so a run stays reproducible per seed but does not replay CUDA's draws. Masks and index rows are cached per device: each one replaces an index_put_ or an arange the NPU would dispatch per step.
    """
    request_count, codebook_count, vocabulary_size = logits.shape
    device = logits.device
    if request_count == 0:
        return torch.empty((0, codebook_count), dtype=torch.int64, device=device)
    else:
        pass
    candidate_count = max(MIN_CANDIDATE_COUNT, triton.next_power_of_2(max_top_k))
    slot_ids = slots[:request_count].to(torch.int64)
    row_knobs = knobs.index_select(1, slot_ids)
    temperature = row_knobs[0].to(torch.float32)
    top_p = row_knobs[1].to(torch.float32)
    repetition_penalty = row_knobs[2].to(torch.float32).view(request_count, 1, 1)
    row_top_k = (
        top_k.index_select(0, slot_ids).to(torch.int64).view(request_count, 1, 1)
    )
    seed = seeds.index_select(0, slot_ids).to(torch.int64)
    step = steps.index_select(0, slot_ids).to(torch.int64)

    scores = logits.to(torch.float32) * (1.0 / softcap)
    scores = softcap * torch.tanh(scores)
    if 0 <= pad_id < vocabulary_size:
        scores.add_(_pad_bias(device, vocabulary_size, pad_id))
    else:
        pass
    penalized_codebooks = min(windows.shape[1], codebook_count)
    counts_width = window_counts.shape[-1]
    if penalized_codebooks and counts_width:
        counts = window_counts.index_select(0, slot_ids)[:, :penalized_codebooks]
        head = scores[:, :penalized_codebooks, :counts_width]
        penalized = torch.where(
            head > 0, head / repetition_penalty, head * repetition_penalty
        )
        scores[:, :penalized_codebooks, :counts_width] = torch.where(
            counts > 0, penalized, head
        )
    else:
        pass

    # Temperature 0 keeps the best candidate: the draw below is then forced to rank 0.
    is_sampling = (temperature > 0).view(request_count, 1, 1)
    scores = scores / torch.where(
        is_sampling, temperature.view(request_count, 1, 1), 1.0
    )

    keep = min(candidate_count, vocabulary_size)
    candidate_values, candidate_codes = torch.topk(scores, keep, dim=-1)
    candidate_values = torch.where(
        _rank_row(device, keep) < row_top_k,
        candidate_values,
        _negative_inf_row(device, keep),
    )
    probabilities = torch.softmax(candidate_values, dim=-1)
    mass_before = torch.cumsum(probabilities, dim=-1) - probabilities
    probabilities = torch.where(
        mass_before > top_p.view(request_count, 1, 1),
        zero_scalar(device),
        probabilities,
    )
    # The rank stride is the vocabulary block, so a slot's stream does not depend on the batch's candidate count.
    vocabulary_block = triton.next_power_of_2(vocabulary_size)
    stream = (
        step.view(request_count, 1) * codebook_count
        + torch.arange(codebook_count, device=device, dtype=torch.int64).view(1, -1)
    ) * vocabulary_block
    mixed = (
        (torch.arange(keep, device=device, dtype=torch.int64) + 1).view(1, 1, keep)
        * 0x9E3779B1
        + seed.view(request_count, 1, 1) * 0x85EBCA6B
        + stream.view(request_count, codebook_count, 1) * 0x27D4EB2F
    )
    mixed = mixed ^ (mixed >> 16)
    mixed = mixed * 0x85EBCA6B
    mixed = mixed ^ (mixed >> 13)
    uniform = (mixed & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)
    noise = -torch.log1p(-uniform)
    scored = torch.where(
        probabilities > 0,
        probabilities / noise,
        torch.full_like(probabilities, -1.0),
    )
    choice = scored.argmax(dim=-1)
    choice = torch.where(
        is_sampling.view(request_count, 1), choice, torch.zeros_like(choice)
    )
    codes = torch.gather(candidate_codes, -1, choice.unsqueeze(-1)).squeeze(-1)
    if penalized_codebooks:
        # The drawn code replaces the ring's oldest row and the per-code counts follow it; flat indices keep the ring on a scatter_ instead of an advanced-index put.
        ring_rows = windows.shape[2]
        ring_pos = (step % ring_rows).view(-1, 1)
        codebook_ids = torch.arange(
            penalized_codebooks, device=device, dtype=torch.int64
        ).view(1, -1)
        flat_ring = (
            (slot_ids.view(-1, 1) * penalized_codebooks + codebook_ids) * ring_rows
            + ring_pos
        ).reshape(-1)
        flat_windows = windows.view(-1)
        evicted = flat_windows.index_select(0, flat_ring).view(-1, penalized_codebooks)
        drawn = codes[:, :penalized_codebooks]
        entered = torch.where(
            drawn < codebook_size, drawn, torch.full_like(drawn, NO_CODE)
        )
        if counts_width:
            state_rows = (
                slot_ids.view(-1, 1) * penalized_codebooks + codebook_ids
            ).reshape(-1)
            delta = torch.zeros(
                (state_rows.numel(), counts_width), dtype=torch.int32, device=device
            )
            evicted_flat = evicted.reshape(-1)
            entered_flat = entered.reshape(-1)
            delta.scatter_add_(
                1,
                evicted_flat.clamp(min=0).view(-1, 1),
                torch.where(evicted_flat >= 0, -1, 0).to(torch.int32).view(-1, 1),
            )
            delta.scatter_add_(
                1,
                entered_flat.clamp(min=0, max=counts_width - 1).view(-1, 1),
                torch.where((entered_flat >= 0) & (entered_flat < counts_width), 1, 0)
                .to(torch.int32)
                .view(-1, 1),
            )
            window_counts.reshape(-1, counts_width).index_add_(0, state_rows, delta)
        else:
            pass
        flat_windows.scatter_(0, flat_ring, entered.reshape(-1))
    else:
        pass
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
