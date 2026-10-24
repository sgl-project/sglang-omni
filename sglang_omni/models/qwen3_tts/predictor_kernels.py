# SPDX-License-Identifier: Apache-2.0
"""Optional CUDA kernels for the Qwen3-TTS residual-code predictor."""

from __future__ import annotations

import torch

from sglang_omni.platforms import current_platform

if not current_platform.is_npu():
    try:
        import triton
        import triton.language as tl
    except ImportError:  # pragma: no cover - depends on runtime image
        triton = None
        tl = None
else:
    triton = None
    tl = None


def has_triton_runtime() -> bool:
    return triton is not None and not current_platform.is_npu()


if has_triton_runtime():

    @triton.jit
    def gather_codec_embedding_row_and_add(
        token_id,
        row,
        offsets,
        embedding_weight,
        gathered,
        accumulated,
        projected_weight,
        projected,
        embedding_stride,
        gathered_stride,
        accumulated_stride,
        projected_stride,
        hidden_size: tl.constexpr,
        projected_size: tl.constexpr,
    ):
        mask = offsets < hidden_size
        values = tl.load(
            embedding_weight + token_id * embedding_stride + offsets,
            mask=mask,
        )
        accumulated_offsets = accumulated + row * accumulated_stride + offsets
        gathered_offsets = gathered + row * gathered_stride + offsets
        current = tl.load(accumulated_offsets, mask=mask)
        tl.store(gathered_offsets, values, mask=mask)
        tl.store(accumulated_offsets, current + values, mask=mask)
        if projected_size > 0:
            projected_mask = offsets < projected_size
            projected_values = tl.load(
                projected_weight + token_id * projected_size + offsets,
                mask=projected_mask,
            )
            tl.store(
                projected + row * projected_stride + offsets,
                projected_values,
                mask=projected_mask,
            )
        else:
            pass

    @triton.jit
    def gather_codec_embedding_and_add_kernel(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
        projected_weight,
        projected,
        token_stride,
        embedding_stride,
        gathered_stride,
        accumulated_stride,
        projected_stride,
        hidden_size: tl.constexpr,
        projected_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        gather_codec_embedding_row_and_add(
            tl.load(token_ids + row * token_stride),
            row,
            block * block_size + tl.arange(0, block_size),
            embedding_weight,
            gathered,
            accumulated,
            projected_weight,
            projected,
            embedding_stride,
            gathered_stride,
            accumulated_stride,
            projected_stride,
            hidden_size,
            projected_size,
        )

else:
    gather_codec_embedding_row_and_add = None
    gather_codec_embedding_and_add_kernel = None


def contiguous_storage_ranges_overlap(
    first: torch.Tensor, second: torch.Tensor
) -> bool:
    first_start = first.data_ptr()
    first_end = first_start + first.numel() * first.element_size()
    second_start = second.data_ptr()
    second_end = second_start + second.numel() * second.element_size()
    return first_start < second_end and second_start < first_end


def codec_gather_projected_size(
    batch_size: int,
    device: torch.device,
    embedding_weight: torch.Tensor,
    gathered: torch.Tensor,
    accumulated: torch.Tensor,
    projected_weight: torch.Tensor | None,
    projected: torch.Tensor | None,
) -> int | None:
    """The projected row width a fused codec gather of batch_size rows writes, 0
    without a projection, or None when the tables and buffers do not fit it."""

    if gather_codec_embedding_row_and_add is None:
        return None
    else:
        pass
    if not (embedding_weight.is_cuda and gathered.is_cuda and accumulated.is_cuda):
        return None
    else:
        pass
    if embedding_weight.ndim != 2 or gathered.ndim != 2 or accumulated.ndim != 2:
        return None
    else:
        pass
    hidden_size = embedding_weight.shape[1]
    if batch_size == 0 or hidden_size == 0:
        return None
    else:
        pass
    if gathered.shape != (batch_size, hidden_size):
        return None
    else:
        pass
    if accumulated.shape != (batch_size, hidden_size):
        return None
    else:
        pass
    if (
        embedding_weight.dtype != torch.bfloat16
        or gathered.dtype != torch.bfloat16
        or accumulated.dtype != torch.bfloat16
    ):
        return None
    else:
        pass
    if not (device == embedding_weight.device == gathered.device == accumulated.device):
        return None
    else:
        pass
    if (
        not embedding_weight.is_contiguous()
        or not gathered.is_contiguous()
        or not accumulated.is_contiguous()
    ):
        return None
    else:
        pass
    if (
        embedding_weight.stride(1) != 1
        or gathered.stride(1) != 1
        or accumulated.stride(1) != 1
    ):
        return None
    else:
        pass
    if (
        contiguous_storage_ranges_overlap(gathered, accumulated)
        or contiguous_storage_ranges_overlap(gathered, embedding_weight)
        or contiguous_storage_ranges_overlap(accumulated, embedding_weight)
    ):
        return None
    else:
        pass

    assert (projected_weight is None) == (
        projected is None
    ), "projected_weight and projected are passed together"
    if projected_weight is None:
        return 0
    elif not (
        projected_weight.ndim == 2
        and projected.ndim == 2
        and projected_weight.shape[0] == embedding_weight.shape[0]
        # note (ratish): the gather spans the embedding width,
        # so a projected row must fit in it.
        and projected_weight.shape[1] <= hidden_size
        and projected.shape == (batch_size, projected_weight.shape[1])
        and projected_weight.dtype == torch.bfloat16
        and projected.dtype == torch.bfloat16
        and projected_weight.device == projected.device == device
        and projected_weight.is_contiguous()
        and projected.is_contiguous()
        and not contiguous_storage_ranges_overlap(projected, gathered)
        and not contiguous_storage_ranges_overlap(projected, accumulated)
        and not contiguous_storage_ranges_overlap(projected, projected_weight)
    ):
        return None
    else:
        return projected_weight.shape[1]


def gather_codec_embedding_and_add(
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    gathered: torch.Tensor,
    accumulated: torch.Tensor,
    projected_weight: torch.Tensor | None = None,
    projected: torch.Tensor | None = None,
) -> bool:
    """Gather BF16 embedding rows and add them to an accumulator in one launch;
    with projected_weight, also gather the same rows of that table into projected.
    Return False without writes when the caller must use the eager path."""

    if gather_codec_embedding_and_add_kernel is None:
        return False
    else:
        pass
    if not (
        token_ids.is_cuda
        and token_ids.ndim == 1
        and token_ids.dtype in (torch.int32, torch.int64)
        and token_ids.is_contiguous()
        and token_ids.stride(0) == 1
    ):
        return False
    else:
        pass
    batch_size = token_ids.shape[0]
    projected_size = codec_gather_projected_size(
        batch_size,
        token_ids.device,
        embedding_weight,
        gathered,
        accumulated,
        projected_weight,
        projected,
    )
    if projected_size is None:
        return False
    elif projected_size == 0:
        projected_weight = embedding_weight
        projected = gathered
    else:
        pass

    hidden_size = embedding_weight.shape[1]
    block_size = 256
    grid = (batch_size, triton.cdiv(hidden_size, block_size))
    gather_codec_embedding_and_add_kernel[grid](
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
        projected_weight,
        projected,
        token_ids.stride(0),
        embedding_weight.stride(0),
        gathered.stride(0),
        accumulated.stride(0),
        projected.stride(0),
        hidden_size=hidden_size,
        projected_size=projected_size,
        block_size=block_size,
        num_warps=4,
    )
    return True
