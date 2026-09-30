# SPDX-License-Identifier: Apache-2.0
"""Optional accelerator kernels for the Qwen3-TTS residual-code predictor."""

from __future__ import annotations

import torch

from sglang_omni.platforms import current_platform

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


def has_triton_runtime() -> bool:
    return triton is not None and not current_platform.is_npu()


if triton is not None and current_platform.is_npu():

    @triton.jit
    def short_cache_gqa_kernel(
        q: tl.tensor,
        key: tl.tensor,
        value: tl.tensor,
        output: tl.tensor,
        query_batch_stride: tl.constexpr,
        query_head_stride: tl.constexpr,
        key_batch_stride: tl.constexpr,
        key_head_stride: tl.constexpr,
        key_position_stride: tl.constexpr,
        value_batch_stride: tl.constexpr,
        value_head_stride: tl.constexpr,
        value_position_stride: tl.constexpr,
        length: tl.constexpr,
        block: tl.constexpr,
    ) -> None:
        batch = tl.program_id(0)
        head = tl.program_id(1)
        column = tl.arange(0, 128)
        position = tl.arange(0, block)
        query = tl.load(
            q + batch * query_batch_stride + head * query_head_stride + column
        ).to(tl.float32)
        keys = tl.load(
            key
            + batch * key_batch_stride
            + (head // 2) * key_head_stride
            + position[:, None] * key_position_stride
            + column[None, :],
            position[:, None] < length,
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * (128**-0.5)
        scores = tl.where(position < length, scores, -float("inf"))
        scores = tl.exp(scores - tl.max(scores, axis=0))
        weights = scores / tl.sum(scores, axis=0)
        values = tl.load(
            value
            + batch * value_batch_stride
            + (head // 2) * value_head_stride
            + position[:, None] * value_position_stride
            + column[None, :],
            position[:, None] < length,
            other=0,
        ).to(tl.float32)
        result = tl.sum(values * weights[:, None], axis=0)
        tl.store(output + (batch * 16 + head) * 128 + column, result)

else:
    short_cache_gqa_kernel = None


def short_cache_gqa_npu(
    q: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor | None:
    """Fuse single-query 1.7B Predictor GQA without materializing KV slices."""
    if short_cache_gqa_kernel is None or q.device.type != "npu":
        return None
    else:
        pass
    if q.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        return None
    else:
        pass
    batch = q.shape[0]
    length = key.shape[2]
    if not (
        0 < batch <= 32
        and q.shape[1:] == (16, 1, 128)
        and key.shape == (batch, 8, length, 128)
        and 0 < length <= 16
        and q.dtype == key.dtype == value.dtype == torch.bfloat16
        and q.device == key.device == value.device
        and q.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    ):
        return None
    else:
        pass
    result = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    short_cache_gqa_kernel[(batch, 16)](
        q,
        key,
        value,
        result,
        q.stride(0),
        q.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        length,
        triton.next_power_of_2(length),
    )
    return result


if has_triton_runtime():

    @triton.jit
    def gather_codec_embedding_and_add_kernel(
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
    gather_codec_embedding_and_add_kernel = None


def contiguous_storage_ranges_overlap(
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

    if gather_codec_embedding_and_add_kernel is None:
        return False
    else:
        pass
    if not (
        token_ids.is_cuda
        and embedding_weight.is_cuda
        and gathered.is_cuda
        and accumulated.is_cuda
    ):
        return False
    else:
        pass
    if token_ids.ndim != 1 or embedding_weight.ndim != 2:
        return False
    else:
        pass
    if gathered.ndim != 2 or accumulated.ndim != 2:
        return False
    else:
        pass
    batch_size = token_ids.shape[0]
    hidden_size = embedding_weight.shape[1]
    if batch_size == 0 or hidden_size == 0:
        return False
    else:
        pass
    if gathered.shape != (batch_size, hidden_size):
        return False
    else:
        pass
    if accumulated.shape != (batch_size, hidden_size):
        return False
    else:
        pass
    if token_ids.dtype not in (torch.int32, torch.int64):
        return False
    else:
        pass
    if (
        embedding_weight.dtype != torch.bfloat16
        or gathered.dtype != torch.bfloat16
        or accumulated.dtype != torch.bfloat16
    ):
        return False
    else:
        pass
    if not (
        token_ids.device
        == embedding_weight.device
        == gathered.device
        == accumulated.device
    ):
        return False
    else:
        pass
    if (
        not token_ids.is_contiguous()
        or not embedding_weight.is_contiguous()
        or not gathered.is_contiguous()
        or not accumulated.is_contiguous()
    ):
        return False
    else:
        pass
    if (
        token_ids.stride(0) != 1
        or embedding_weight.stride(1) != 1
        or gathered.stride(1) != 1
        or accumulated.stride(1) != 1
    ):
        return False
    else:
        pass
    if (
        contiguous_storage_ranges_overlap(gathered, accumulated)
        or contiguous_storage_ranges_overlap(gathered, embedding_weight)
        or contiguous_storage_ranges_overlap(accumulated, embedding_weight)
    ):
        return False
    else:
        pass

    block_size = 256
    grid = (batch_size, triton.cdiv(hidden_size, block_size))
    gather_codec_embedding_and_add_kernel[grid](
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
