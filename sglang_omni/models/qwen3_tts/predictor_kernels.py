# SPDX-License-Identifier: Apache-2.0
"""Optional CUDA kernels for the Qwen3-TTS residual-code predictor."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _gather_codec_embedding_and_add_kernel(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
        token_stride,
        embedding_stride,
        gathered_stride,
        accumulated_stride,
        hidden_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < hidden_size
        token_id = tl.load(token_ids + row * token_stride)
        values = tl.load(
            embedding_weight + token_id * embedding_stride + offsets,
            mask=mask,
        )
        accumulated_offsets = accumulated + row * accumulated_stride + offsets
        gathered_offsets = gathered + row * gathered_stride + offsets
        current = tl.load(accumulated_offsets, mask=mask)
        tl.store(gathered_offsets, values, mask=mask)
        tl.store(accumulated_offsets, current + values, mask=mask)

else:
    _gather_codec_embedding_and_add_kernel = None


def _contiguous_storage_ranges_overlap(
    first: torch.Tensor, second: torch.Tensor
) -> bool:
    first_start = first.data_ptr()
    first_end = first_start + first.numel() * first.element_size()
    second_start = second.data_ptr()
    second_end = second_start + second.numel() * second.element_size()
    return first_start < second_end and second_start < first_end


def gather_codec_embedding_and_add(
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    gathered: torch.Tensor,
    accumulated: torch.Tensor,
) -> bool:
    """Gather BF16 embedding rows and add them to an accumulator in one launch.

    Return ``False`` without writes when the caller must use the eager path.
    """

    if _gather_codec_embedding_and_add_kernel is None:
        return False
    if not (
        token_ids.is_cuda
        and embedding_weight.is_cuda
        and gathered.is_cuda
        and accumulated.is_cuda
    ):
        return False
    if token_ids.ndim != 1 or embedding_weight.ndim != 2:
        return False
    if gathered.ndim != 2 or accumulated.ndim != 2:
        return False
    batch_size = token_ids.shape[0]
    hidden_size = embedding_weight.shape[1]
    if batch_size == 0 or hidden_size == 0:
        return False
    if gathered.shape != (batch_size, hidden_size):
        return False
    if accumulated.shape != (batch_size, hidden_size):
        return False
    if token_ids.dtype not in (torch.int32, torch.int64):
        return False
    if (
        embedding_weight.dtype != torch.bfloat16
        or gathered.dtype != torch.bfloat16
        or accumulated.dtype != torch.bfloat16
    ):
        return False
    if not (
        token_ids.device
        == embedding_weight.device
        == gathered.device
        == accumulated.device
    ):
        return False
    if (
        not token_ids.is_contiguous()
        or not embedding_weight.is_contiguous()
        or not gathered.is_contiguous()
        or not accumulated.is_contiguous()
    ):
        return False
    if (
        token_ids.stride(0) != 1
        or embedding_weight.stride(1) != 1
        or gathered.stride(1) != 1
        or accumulated.stride(1) != 1
    ):
        return False
    if (
        _contiguous_storage_ranges_overlap(gathered, accumulated)
        or _contiguous_storage_ranges_overlap(gathered, embedding_weight)
        or _contiguous_storage_ranges_overlap(accumulated, embedding_weight)
    ):
        return False

    block_size = 256
    grid = (batch_size, triton.cdiv(hidden_size, block_size))
    _gather_codec_embedding_and_add_kernel[grid](
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
        token_ids.stride(0),
        embedding_weight.stride(0),
        gathered.stride(0),
        accumulated.stride(0),
        hidden_size=hidden_size,
        block_size=block_size,
        num_warps=4,
    )
    return True
