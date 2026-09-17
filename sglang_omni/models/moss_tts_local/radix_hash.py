# SPDX-License-Identifier: Apache-2.0
"""Capture-safe GPU radix-key hash for MOSS-TTS Local generated frames.

The scheduler appends one radix-cache token id per generated frame to a
request's KV chain, and the radix tree keys on those ids. The text channel
alone is the same assistant-slot id for every continuing frame, so the key
must hash the full multi-channel row (text + RVQ codes) to keep a radix match
implying identical audio content.

Prompt rows are hashed once, off the decode hot path, by
``moss_tts.request_builders.build_row_cache_key_ids`` (host-side blake2b); that
call is fine to keep -- it never runs inside a CUDA-graph capture region. The
*generated*-row key, by contrast, is computed every decode step on a tensor
that the local-frame decode just produced on device. Hashing it host-side
forces a GPU->CPU sync (``.cpu()``/``numpy``) every frame, which blocks
CUDA-graph capture and the async-decode lookahead (#734/#736). This module
hashes the row tensor with a fixed-coefficient polynomial in one optional
Triton kernel, with the original int64 Torch implementation retained as an
exact fallback. Both paths stay on-device and are graph-capturable.

See ``docs/design/gpu_radix_hash.md`` for the capture-safety argument, the
collision analysis, and the two-layer verification rubric.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None

# <|endoftext|> = 151643 opens the special/control id band. Generated radix
# keys fold strictly below it; the scheduler finishes any request whose
# generated id crosses this boundary (``Req._check_vocab_boundary_finish``), so
# a real (continuing) audio frame must never land in or above the band.
RADIX_HASH_SPACE = 151643

# Polynomial-hash constants.
#
# _MOD is the Mersenne prime 2**31 - 1. With the accumulator and every channel
# value reduced below _MOD (< 2**31) and _BASE < _MOD, each Horner step
# ``acc * _BASE + v`` stays below 2**31 * 2**31 = 2**62, comfortably inside
# signed int64 (max 2**63 - 1). So the int64 ops never overflow and the result
# is bit-reproducible on CPU and GPU -- no implementation-defined wraparound.
#
# _BASE is a large prime well below _MOD. Folding each channel in as a power of
# _BASE (Horner) makes the hash order-sensitive (a channel permutation changes
# the key) and spreads neighbours (a single-channel +/-1 changes the key by a
# power of _BASE mod _MOD). Both constants are arbitrary fixed primes chosen
# only for these size/spread properties; the generated-row key space is private
# to the radix cache, so the exact values carry no on-disk/ABI contract.
_MOD = 2147483647  # 2**31 - 1, Mersenne prime M31
_BASE = 1000000007  # 1e9 + 7, prime, < _MOD

_TRITON_BLOCK_SIZE = 128


if triton is not None:

    @triton.jit(
        do_not_specialize=[
            "batch_size",
            "num_channels",
            "row_stride",
            "channel_stride",
            "text_stride",
        ]
    )
    def _radix_row_hash_kernel(
        rows_ptr,
        next_text_ptr,
        out_ptr,
        batch_size,
        num_channels,
        row_stride,
        channel_stride,
        text_stride,
        end_id,
        hash_space,
        MOD: tl.constexpr,
        BASE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = row < batch_size
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        # The loop runs inside one kernel. Batch, channel count, and strides
        # stay runtime parameters, including during CUDA-graph capture.
        for channel in range(num_channels):
            value = tl.load(
                rows_ptr + row * row_stride + channel * channel_stride,
                mask=row_mask,
                other=0,
            ).to(tl.int64)
            value = value % MOD
            # Triton uses signed remainder; Torch uses floor remainder.
            value = tl.where(value < 0, value + MOD, value)
            acc = (acc * BASE + value) % MOD

        folded = acc % hash_space
        next_text_value = tl.load(
            next_text_ptr + row * text_stride,
            mask=row_mask,
            other=0,
        ).to(tl.int64)
        output = tl.where(next_text_value == end_id, next_text_value, folded)
        tl.store(out_ptr + row, output, mask=row_mask)

    @triton.jit(
        do_not_specialize=[
            "batch_size",
            "num_channels",
            "stop_stride",
            "code_row_stride",
            "code_col_stride",
        ]
    )
    def _build_rows_and_hash_kernel(
        stop_ptr,
        codes_ptr,
        rows_ptr,
        ids_ptr,
        batch_size,
        num_channels,
        stop_stride,
        code_row_stride,
        code_col_stride,
        row_stride,
        slot_id,
        end_id,
        hash_space,
        MOD: tl.constexpr,
        BASE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = row < batch_size
        stop = tl.load(
            stop_ptr + row * stop_stride,
            mask=row_mask,
            other=1,
        ).to(tl.int64)
        text = tl.where(stop == 0, slot_id, end_id).to(tl.int64)
        tl.store(rows_ptr + row * row_stride, text, mask=row_mask)
        acc = text % MOD
        acc = tl.where(acc < 0, acc + MOD, acc)

        # Write the row and consume each code once. The raw code is preserved
        # in rows while its floor-reduced value feeds the polynomial hash.
        for channel in range(num_channels):
            raw = tl.load(
                codes_ptr + row * code_row_stride + channel * code_col_stride,
                mask=row_mask,
                other=0,
            ).to(tl.int64)
            value = raw % MOD
            value = tl.where(value < 0, value + MOD, value)
            acc = (acc * BASE + value) % MOD
            tl.store(rows_ptr + row * row_stride + channel + 1, raw, mask=row_mask)

        folded = acc % hash_space
        output = tl.where(text == end_id, text, folded)
        tl.store(ids_ptr + row, output, mask=row_mask)

else:
    _radix_row_hash_kernel = None
    _build_rows_and_hash_kernel = None


def poly_row_hash(rows: torch.Tensor) -> torch.Tensor:
    """Fixed-coefficient polynomial hash of each row, in ``[0, _MOD)``.

    ``rows`` is ``[B, C]`` integer. Returns ``[B]`` int64 on ``rows.device``.
    Pure elementwise int64 torch ops (mul / add / remainder) over a static
    channel count -- no host sync, CUDA-graph capturable.
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D [B, C], got shape {tuple(rows.shape)}")
    work = rows.to(torch.int64)
    acc = torch.zeros(work.shape[0], dtype=torch.int64, device=work.device)
    # Static trip count (one frame = a fixed number of channels): the loop
    # unrolls into a fixed op sequence at capture time.
    for channel in range(work.shape[1]):
        # Reduce defensively in case a caller passes a raw id >= _MOD.
        value = torch.remainder(work[:, channel], _MOD)
        acc = torch.remainder(acc * _BASE + value, _MOD)
    return acc


def gpu_radix_row_hash(
    rows: torch.Tensor,
    next_text: torch.Tensor,
    end_id: int,
    *,
    hash_space: int = RADIX_HASH_SPACE,
) -> torch.Tensor:
    """Capture-safe radix token ids for a batch of generated frames.

    ``rows`` is ``[B, C]`` int64 (text channel + RVQ codes); ``next_text`` is
    ``[B]`` (the text-channel id, ``end_id`` for a stop frame). Continuing
    frames get a key in ``[0, hash_space)``; EOS rows keep the raw ``end_id``
    so the existing eos detection still fires. device/dtype follow ``rows``.
    """
    if (
        _radix_row_hash_kernel is not None
        and rows.device.type == "cuda"
        and rows.ndim == 2
        and rows.dtype in (torch.int32, torch.int64)
        and next_text.device == rows.device
        and next_text.ndim == 1
        and next_text.numel() == rows.shape[0]
        and next_text.dtype in (torch.int32, torch.int64)
        and hash_space > 0
    ):
        output = torch.empty((rows.shape[0],), dtype=torch.int64, device=rows.device)
        if rows.shape[0] == 0:
            return output
        with torch.cuda.device(rows.device):
            _radix_row_hash_kernel[(triton.cdiv(rows.shape[0], _TRITON_BLOCK_SIZE),)](
                rows,
                next_text,
                output,
                rows.shape[0],
                rows.shape[1],
                rows.stride(0),
                rows.stride(1),
                next_text.stride(0),
                end_id,
                hash_space,
                MOD=_MOD,
                BASE=_BASE,
                BLOCK_SIZE=_TRITON_BLOCK_SIZE,
                num_warps=4,
            )
        return output

    folded = torch.remainder(poly_row_hash(rows), hash_space)
    return torch.where(next_text == end_id, next_text.to(torch.int64), folded)


def build_rows_and_radix_token_ids(
    stop_choice: torch.Tensor,
    codes: torch.Tensor,
    slot_id: int,
    end_id: int,
    *,
    hash_space: int = RADIX_HASH_SPACE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a generated frame row and its radix id in one device pass.

    ``stop_choice`` follows the local decoder convention: zero continues and
    any nonzero value emits ``end_id``. The fused CUDA path writes the raw code
    row (needed by the embedding/history paths) while hashing the same values,
    so it is behavior-equivalent to constructing ``rows`` and then calling
    :func:`gpu_radix_row_hash` separately. Unsupported inputs use that exact
    Torch sequence as a fallback.
    """
    if (
        _build_rows_and_hash_kernel is not None
        and stop_choice.device.type == "cuda"
        and codes.device == stop_choice.device
        and stop_choice.ndim == 1
        and codes.ndim == 2
        and stop_choice.numel() == codes.shape[0]
        and stop_choice.dtype in (torch.int32, torch.int64)
        and codes.dtype in (torch.int32, torch.int64)
        and hash_space > 0
    ):
        batch_size, num_channels = codes.shape
        rows = torch.empty(
            (batch_size, num_channels + 1),
            dtype=torch.int64,
            device=codes.device,
        )
        ids = torch.empty((batch_size,), dtype=torch.int64, device=codes.device)
        if batch_size == 0:
            return rows, ids
        with torch.cuda.device(codes.device):
            _build_rows_and_hash_kernel[(triton.cdiv(batch_size, _TRITON_BLOCK_SIZE),)](
                stop_choice,
                codes,
                rows,
                ids,
                batch_size,
                num_channels,
                stop_choice.stride(0),
                codes.stride(0),
                codes.stride(1),
                rows.stride(0),
                slot_id,
                end_id,
                hash_space,
                MOD=_MOD,
                BASE=_BASE,
                BLOCK_SIZE=_TRITON_BLOCK_SIZE,
                num_warps=4,
            )
        return rows, ids

    next_text = torch.where(
        stop_choice == 0,
        torch.full((codes.shape[0],), slot_id, dtype=torch.long, device=codes.device),
        torch.full((codes.shape[0],), end_id, dtype=torch.long, device=codes.device),
    )
    rows = torch.empty(
        (codes.shape[0], codes.shape[1] + 1), dtype=torch.long, device=codes.device
    )
    rows[:, 0] = next_text
    rows[:, 1:] = codes
    return rows, gpu_radix_row_hash(rows, next_text, end_id, hash_space=hash_space)
