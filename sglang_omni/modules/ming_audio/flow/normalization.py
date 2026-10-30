"""Pointwise normalization dispatch with native intermediate rounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False
else:
    HAS_TRITON = True

NATIVE_RMS_CHANNEL_COUNT = 1024
NATIVE_REDUCTION_MAX_THREADS = 512
NATIVE_REDUCTION_WARP_WIDTH = 32
NATIVE_REDUCTION_VECTOR_WIDTH = 4
RMS_KERNEL_WARP_COUNT = 4
VOCODER_RMS_CHANNEL_COUNT = 896
VOCODER_RMS_ROW_TILE = 4


class TensorSquare(Protocol):
    def __call__(self, hidden_states: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError


class NormalizationAffine(Protocol):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        variance: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        /,
    ) -> torch.Tensor:
        raise NotImplementedError


class RowNormalization(Protocol):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        /,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class NormalizationDispatch:
    square_float32: TensorSquare
    affine: NormalizationAffine
    normalize_contiguous: RowNormalization | None = None


def square_float32(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states.to(torch.float32).pow(2)


def normalization_affine(
    hidden_states: torch.Tensor,
    variance: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return (hidden_states * torch.rsqrt(variance + epsilon)).to(weight.dtype) * weight


if HAS_TRITON:

    @triton.jit
    def native_order_rms_kernel(
        hidden_states: tl.tensor,
        weight: tl.tensor,
        output: tl.tensor,
        epsilon: tl.constexpr,
        channel_count: tl.constexpr,
        reduction_lane_count: tl.constexpr,
        warp_width: tl.constexpr,
        vector_width: tl.constexpr,
    ) -> None:
        row_index = tl.program_id(0)
        lane_indexes = tl.arange(0, reduction_lane_count)
        first_sum = tl.full((reduction_lane_count,), 0, tl.float32)
        second_sum = tl.full((reduction_lane_count,), 0, tl.float32)
        third_sum = tl.full((reduction_lane_count,), 0, tl.float32)
        fourth_sum = tl.full((reduction_lane_count,), 0, tl.float32)
        # note (Zhuofeng Li): Preserve the native four-accumulator FP32 mean and descending warp tree.
        for group_index in tl.static_range(
            channel_count // (vector_width * reduction_lane_count)
        ):
            offsets = (
                row_index * channel_count
                + lane_indexes * vector_width
                + group_index * vector_width * reduction_lane_count
            )
            first_value = tl.load(hidden_states + offsets).to(tl.float32)
            second_value = tl.load(hidden_states + offsets + 1).to(tl.float32)
            third_value = tl.load(hidden_states + offsets + 2).to(tl.float32)
            fourth_value = tl.load(hidden_states + offsets + 3).to(tl.float32)
            first_sum = first_sum + first_value * first_value
            second_sum = second_sum + second_value * second_value
            third_sum = third_sum + third_value * third_value
            fourth_sum = fourth_sum + fourth_value * fourth_value
        partial_sums = ((first_sum + second_sum) + third_sum) + fourth_sum
        for shift in tl.static_range(
            0, tl.constexpr((reduction_lane_count // warp_width).bit_length() - 1)
        ):
            offset = reduction_lane_count // (2 << shift)
            other_sums = tl.gather(partial_sums, lane_indexes ^ offset, axis=0)
            partial_sums = partial_sums + other_sums
        squared_sum = tl.sum(
            tl.where(lane_indexes < warp_width, partial_sums, 0.0), axis=0
        )
        variance = squared_sum * (1.0 / channel_count)
        inverse_root_mean_square = tl.rsqrt(variance + epsilon)
        channel_indexes = tl.arange(0, channel_count)
        values = tl.load(
            hidden_states + row_index * channel_count + channel_indexes
        ).to(tl.float32)
        weights = tl.load(weight + channel_indexes)
        normalized = (
            (values * inverse_root_mean_square).to(weights.dtype).to(tl.float32)
        )
        result = normalized * weights.to(tl.float32)
        tl.store(output + row_index * channel_count + channel_indexes, result)

else:
    pass


def normalize_native_rms(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    row_count = hidden_states.numel() // NATIVE_RMS_CHANNEL_COUNT
    block_row_count = min(
        1 << (row_count.bit_length() - 1),
        NATIVE_REDUCTION_MAX_THREADS // NATIVE_REDUCTION_WARP_WIDTH,
    )
    reduction_lane_count = min(
        NATIVE_RMS_CHANNEL_COUNT // NATIVE_REDUCTION_VECTOR_WIDTH,
        NATIVE_REDUCTION_MAX_THREADS // block_row_count,
    )
    output = torch.empty_like(hidden_states)
    native_order_rms_kernel[(row_count,)](
        hidden_states,
        weight,
        output,
        epsilon,
        NATIVE_RMS_CHANNEL_COUNT,
        reduction_lane_count,
        NATIVE_REDUCTION_WARP_WIDTH,
        NATIVE_REDUCTION_VECTOR_WIDTH,
        num_warps=RMS_KERNEL_WARP_COUNT,
        enable_fp_fusion=False,
    )
    return output


if HAS_TRITON:

    @triton.jit
    def native_strided_rms_kernel(
        hidden_states: tl.tensor,
        weight: tl.tensor,
        output: tl.tensor,
        row_count: tl.tensor,
        epsilon: tl.constexpr,
        channel_count: tl.constexpr,
        reduction_lane_count: tl.constexpr,
        channel_block_size: tl.constexpr,
        row_tile_size: tl.constexpr,
        vector_width: tl.constexpr,
    ) -> None:
        row_indexes = (
            tl.program_id(0) * row_tile_size + tl.arange(0, row_tile_size)[None, :]
        )
        lane_indexes = tl.arange(0, reduction_lane_count)[:, None]
        first_sum = tl.full((reduction_lane_count, row_tile_size), 0, tl.float32)
        second_sum = tl.full((reduction_lane_count, row_tile_size), 0, tl.float32)
        third_sum = tl.full((reduction_lane_count, row_tile_size), 0, tl.float32)
        fourth_sum = tl.full((reduction_lane_count, row_tile_size), 0, tl.float32)
        # note (Zhuofeng Li): Native strided means vectorize rows and reduce channels in four sequential accumulators.
        for group_index in tl.static_range(
            tl.cdiv(channel_count, vector_width * reduction_lane_count)
        ):
            first_channels = (
                lane_indexes + group_index * vector_width * reduction_lane_count
            )
            second_channels = first_channels + reduction_lane_count
            third_channels = second_channels + reduction_lane_count
            fourth_channels = third_channels + reduction_lane_count
            first_value = tl.load(
                hidden_states + row_indexes + first_channels * row_count,
                mask=(first_channels < channel_count) & (row_indexes < row_count),
                other=0,
            ).to(tl.float32)
            second_value = tl.load(
                hidden_states + row_indexes + second_channels * row_count,
                mask=(second_channels < channel_count) & (row_indexes < row_count),
                other=0,
            ).to(tl.float32)
            third_value = tl.load(
                hidden_states + row_indexes + third_channels * row_count,
                mask=(third_channels < channel_count) & (row_indexes < row_count),
                other=0,
            ).to(tl.float32)
            fourth_value = tl.load(
                hidden_states + row_indexes + fourth_channels * row_count,
                mask=(fourth_channels < channel_count) & (row_indexes < row_count),
                other=0,
            ).to(tl.float32)
            first_sum = first_sum + first_value * first_value
            second_sum = second_sum + second_value * second_value
            third_sum = third_sum + third_value * third_value
            fourth_sum = fourth_sum + fourth_value * fourth_value
        partial_sums = ((first_sum + second_sum) + third_sum) + fourth_sum
        # note (Zhuofeng Li): Spell out every descending level because a tiled tl.sum can choose a different FP32 tree.
        for shift in tl.static_range(
            0, tl.constexpr(reduction_lane_count.bit_length() - 1)
        ):
            offset = reduction_lane_count // (2 << shift)
            other_sums = tl.gather(
                partial_sums,
                tl.broadcast_to(
                    lane_indexes ^ offset, (reduction_lane_count, row_tile_size)
                ),
                axis=0,
            )
            partial_sums = partial_sums + other_sums
        squared_sum = tl.sum(tl.where(lane_indexes == 0, partial_sums, 0.0), axis=0)
        variance = squared_sum * (1.0 / channel_count)
        inverse_root_mean_square = tl.rsqrt(variance + epsilon)
        channel_indexes = tl.arange(0, channel_block_size)[:, None]
        offsets = row_indexes + channel_indexes * row_count
        valid_values = (channel_indexes < channel_count) & (row_indexes < row_count)
        values = tl.load(hidden_states + offsets, mask=valid_values, other=0).to(
            tl.float32
        )
        weights = tl.load(
            weight + channel_indexes, mask=channel_indexes < channel_count, other=0
        )
        normalized = (
            (values * inverse_root_mean_square[None, :])
            .to(weights.dtype)
            .to(tl.float32)
        )
        result = weights.to(tl.float32) * normalized
        tl.store(output + offsets, result, mask=valid_values)

else:
    pass


def normalize_native_strided_rms(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    row_count = hidden_states.shape[1]
    block_column_count = min(
        1 << ((row_count // NATIVE_REDUCTION_VECTOR_WIDTH).bit_length() - 1),
        NATIVE_REDUCTION_WARP_WIDTH,
    )
    reduction_lane_count = (
        NATIVE_REDUCTION_MAX_THREADS
        // NATIVE_REDUCTION_VECTOR_WIDTH
        // block_column_count
    )
    output = torch.empty_like(hidden_states)
    native_strided_rms_kernel[(triton.cdiv(row_count, VOCODER_RMS_ROW_TILE),)](
        hidden_states,
        weight,
        output,
        row_count,
        epsilon,
        VOCODER_RMS_CHANNEL_COUNT,
        reduction_lane_count,
        triton.next_power_of_2(VOCODER_RMS_CHANNEL_COUNT),
        VOCODER_RMS_ROW_TILE,
        NATIVE_REDUCTION_VECTOR_WIDTH,
        num_warps=RMS_KERNEL_WARP_COUNT,
        enable_fp_fusion=False,
    )
    return output
