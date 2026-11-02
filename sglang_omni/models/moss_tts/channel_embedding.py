# SPDX-License-Identifier: Apache-2.0
"""Fused multi-channel embedding sum for MOSS-TTS Delay input rows."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHANNEL_EMBEDDING_BLOCK = 1024


@triton.jit
def channel_embedding_sum_kernel(
    token_ids_ptr,
    weight_pointers_ptr,
    vocab_sizes_ptr,
    output_ptr,
    token_row_stride,
    token_channel_stride,
    HIDDEN: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Round after each channel to preserve the eager ordered sum."""

    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    column_mask = columns < HIDDEN
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for channel in tl.static_range(CHANNELS):
        table = tl.load(weight_pointers_ptr + channel).to(
            tl.pointer_type(output_ptr.dtype.element_ty)
        )
        token_id = tl.load(
            token_ids_ptr + row * token_row_stride + channel * token_channel_stride
        )
        in_range = (token_id >= 0) & (token_id < tl.load(vocab_sizes_ptr + channel))
        values = tl.load(
            table + token_id * HIDDEN + columns,
            mask=column_mask & in_range,
            other=0.0,
        )
        total = (total + values.to(tl.float32)).to(output_ptr.dtype.element_ty)
        total = total.to(tl.float32)
    tl.store(
        output_ptr + row * HIDDEN + columns,
        total.to(output_ptr.dtype.element_ty),
        mask=column_mask,
    )


class ChannelEmbeddingTables:
    """Refresh device pointers when embedding storage changes."""

    def __init__(self) -> None:
        self.weight_pointers: tuple[int, ...] = ()
        self.weight_pointer_tensor: torch.Tensor | None = None
        self.vocab_size_tensor: torch.Tensor | None = None

    def refresh(self, weights: list[torch.Tensor]) -> bool:
        """Point the table at weights; return False when they cannot be fused."""

        first = weights[0]
        hidden = int(first.shape[1])
        if not all(
            weight.is_cuda
            and weight.device == first.device
            and weight.dtype == first.dtype
            and weight.ndim == 2
            and int(weight.shape[1]) == hidden
            and weight.is_contiguous()
            for weight in weights
        ):
            return False
        else:
            pass
        pointers = tuple(weight.data_ptr() for weight in weights)
        if pointers == self.weight_pointers:
            return True
        else:
            pass
        if torch.cuda.is_current_stream_capturing():
            return False
        else:
            pass
        self.weight_pointer_tensor = torch.tensor(
            pointers, dtype=torch.int64, device=first.device
        )
        self.vocab_size_tensor = torch.tensor(
            [int(weight.shape[0]) for weight in weights],
            dtype=torch.int64,
            device=first.device,
        )
        self.weight_pointers = pointers
        return True

    def embed(
        self, token_ids: torch.Tensor, weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """Sum weights[channel][token_ids[:, channel]] over channels."""

        first = weights[0]
        rows = int(token_ids.shape[0])
        hidden = int(first.shape[1])
        output = torch.empty((rows, hidden), device=first.device, dtype=first.dtype)
        if rows == 0:
            return output
        else:
            pass
        assert self.weight_pointer_tensor is not None
        assert self.vocab_size_tensor is not None
        channel_embedding_sum_kernel[
            (rows, triton.cdiv(hidden, CHANNEL_EMBEDDING_BLOCK))
        ](
            token_ids,
            self.weight_pointer_tensor,
            self.vocab_size_tensor,
            output,
            token_ids.stride(0),
            token_ids.stride(1),
            HIDDEN=hidden,
            CHANNELS=len(weights),
            BLOCK=CHANNEL_EMBEDDING_BLOCK,
            num_warps=4,
        )
        return output


__all__ = ["ChannelEmbeddingTables"]
