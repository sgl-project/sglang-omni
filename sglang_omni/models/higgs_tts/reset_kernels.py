# SPDX-License-Identifier: Apache-2.0
"""Optional CUDA kernels for Higgs TTS sampler state."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


if triton is not None:

    # note (Dayuxiaoshui): ``row`` changes with every request. Left to Triton's
    # default specialization (row == 1, row % 16 == 0) it would JIT three
    # variants at unpredictable points during serving, each a 40-450 ms stall.
    # Excluding every integer argument leaves one binary per pool.
    @triton.jit(do_not_specialize=["row", "no_seed", "codes_row_stride"])
    def _reset_sampler_row_kernel(
        delay_count,
        eoc_countdown,
        generation_done,
        last_codes,
        seeds,
        step_count,
        row,
        no_seed,
        codes_row_stride,
        num_codebooks: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.arange(0, block_size)
        tl.store(delay_count + row, 0)
        tl.store(eoc_countdown + row, -1)
        tl.store(generation_done + row, 0)
        tl.store(seeds + row, no_seed)
        tl.store(step_count + row, 0)
        tl.store(
            last_codes + row * codes_row_stride + offsets,
            0,
            mask=offsets < num_codebooks,
        )

else:
    _reset_sampler_row_kernel = None


def reset_sampler_row(
    delay_count: torch.Tensor,
    eoc_countdown: torch.Tensor,
    generation_done: torch.Tensor,
    last_codes: torch.Tensor,
    seeds: torch.Tensor,
    step_count: torch.Tensor,
    row: int,
    no_seed: int,
) -> bool:
    """Reset one CUDA state row in one launch, or return ``False``.

    ``False`` means the caller must take its generic path: no Triton, a
    non-CUDA pool, or a ``last_codes`` layout whose codebooks are not
    contiguous, which the kernel does not address.
    """
    if (
        _reset_sampler_row_kernel is None
        or not delay_count.is_cuda
        or last_codes.stride(1) != 1
    ):
        return False
    # The kernel cannot bounds-check; keep the IndexError the tensor
    # indexing of the generic path would have raised.
    if not 0 <= row < delay_count.shape[0]:
        raise IndexError(f"row {row} out of range for {delay_count.shape[0]} rows")

    num_codebooks = last_codes.shape[1]
    block_size = triton.next_power_of_2(num_codebooks)
    _reset_sampler_row_kernel[(1,)](
        delay_count,
        eoc_countdown,
        generation_done,
        last_codes,
        seeds,
        step_count,
        row,
        no_seed,
        # Row stride rather than ``num_codebooks``: a row-sliced view of
        # ``last_codes`` is reset in place instead of past its own row.
        last_codes.stride(0),
        num_codebooks,
        block_size,
        num_warps=1,
    )
    return True


__all__ = ["reset_sampler_row"]
