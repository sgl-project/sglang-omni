# SPDX-License-Identifier: Apache-2.0
"""Residual layer norm of the w2v-BERT teacher layer: the residual add, its float32 layer norm and an optional SiLU or second norm in one Triton kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

# note (zhaochenyang20): Triton's default 4 warps; a 999-row, 1024-wide residual norm took 3.3 us with 4 warps against 3.1-3.3 us for 1 to 8 and 4.1 us for 16 (CUDA graph replay, one H100).
NORM_WARPS = 4


@triton.jit
def residual_layer_norm_kernel(
    update_pointer,
    residual_pointer,
    sum_pointer,
    weight_pointer,
    bias_pointer,
    normalized_pointer,
    second_weight_pointer,
    second_bias_pointer,
    second_normalized_pointer,
    width,
    epsilon,
    update_scale,
    HAS_UPDATE: tl.constexpr,
    HAS_SECOND_NORM: tl.constexpr,
    SHOULD_APPLY_SILU: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    """One row: sum = update * update_scale + residual, rounded to the sum's dtype, then its layer norm and optional SiLU.

    With a second norm, the first norm's float32 output is stored and normalized again.
    """
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_WIDTH)
    is_column = columns < width
    row_offsets = row * width + columns
    residual_sum_row = tl.load(
        residual_pointer + row_offsets, mask=is_column, other=0.0
    ).to(tl.float32)
    if HAS_UPDATE:
        update = tl.load(update_pointer + row_offsets, mask=is_column, other=0.0).to(
            tl.float32
        )
        residual_sum_row = (
            (update * update_scale + residual_sum_row)
            .to(sum_pointer.dtype.element_ty)
            .to(tl.float32)
        )
        tl.store(sum_pointer + row_offsets, residual_sum_row, mask=is_column)
    else:
        pass
    mean = tl.sum(residual_sum_row, axis=0) / width
    centered = tl.where(is_column, residual_sum_row - mean, 0.0)
    reciprocal_deviation = 1.0 / tl.sqrt(
        tl.sum(centered * centered, axis=0) / width + epsilon
    )
    weight = tl.load(weight_pointer + columns, mask=is_column, other=0.0)
    bias = tl.load(bias_pointer + columns, mask=is_column, other=0.0)
    normalized_row = weight * (reciprocal_deviation * centered) + bias
    if SHOULD_APPLY_SILU:
        normalized_row = normalized_row / (1.0 + tl.exp(-normalized_row))
    else:
        pass
    tl.store(
        normalized_pointer + row_offsets,
        normalized_row.to(normalized_pointer.dtype.element_ty),
        mask=is_column,
    )
    if HAS_SECOND_NORM:
        second_mean = tl.sum(tl.where(is_column, normalized_row, 0.0), axis=0) / width
        second_centered = tl.where(is_column, normalized_row - second_mean, 0.0)
        second_reciprocal_deviation = 1.0 / tl.sqrt(
            tl.sum(second_centered * second_centered, axis=0) / width + epsilon
        )
        second_weight = tl.load(
            second_weight_pointer + columns, mask=is_column, other=0.0
        )
        second_bias = tl.load(second_bias_pointer + columns, mask=is_column, other=0.0)
        second_normalized_row = (
            second_weight * (second_reciprocal_deviation * second_centered)
            + second_bias
        )
        tl.store(
            second_normalized_pointer + row_offsets,
            second_normalized_row.to(second_normalized_pointer.dtype.element_ty),
            mask=is_column,
        )
    else:
        pass


def residual_layer_norm(
    residual: torch.Tensor,
    layer_norm: nn.LayerNorm,
    normalized_dtype: torch.dtype,
    update: torch.Tensor | None = None,
    update_scale: float = 1.0,
    second_layer_norm: nn.LayerNorm | None = None,
    should_apply_silu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(sum, layer_norm(sum) in normalized_dtype, second_layer_norm(layer_norm(sum)) in bfloat16) over the last dimension.

    The sum is update * update_scale + residual in residual's dtype, or residual itself without an update; norms are float32, and SiLU, when applied, acts on the first norm's float32 output. Non-CUDA tensors run the same math in torch ops.
    """
    width = residual.shape[-1]
    if residual.is_cuda:
        if update is None:
            sum_states = residual
        else:
            sum_states = torch.empty_like(residual)
        normalized_states = torch.empty(
            residual.shape, dtype=normalized_dtype, device=residual.device
        )
        if second_layer_norm is None:
            second_normalized_states = None
            second_weight, second_bias = layer_norm.weight, layer_norm.bias
        else:
            second_normalized_states = torch.empty(
                residual.shape, dtype=torch.bfloat16, device=residual.device
            )
            second_weight, second_bias = (
                second_layer_norm.weight,
                second_layer_norm.bias,
            )
        residual_layer_norm_kernel[(residual.numel() // width,)](
            residual if update is None else update,
            residual,
            sum_states,
            layer_norm.weight,
            layer_norm.bias,
            normalized_states,
            second_weight,
            second_bias,
            (
                normalized_states
                if second_normalized_states is None
                else second_normalized_states
            ),
            width,
            layer_norm.eps,
            update_scale,
            HAS_UPDATE=update is not None,
            HAS_SECOND_NORM=second_layer_norm is not None,
            SHOULD_APPLY_SILU=should_apply_silu,
            BLOCK_WIDTH=triton.next_power_of_2(width),
            num_warps=NORM_WARPS,
        )
        return sum_states, normalized_states, second_normalized_states
    else:
        if update is None:
            sum_states = residual
        else:
            sum_states = (update.float() * update_scale + residual.float()).to(
                residual.dtype
            )
        float_normalized_states = F.layer_norm(
            sum_states.float(),
            (width,),
            layer_norm.weight,
            layer_norm.bias,
            layer_norm.eps,
        )
        if should_apply_silu:
            float_normalized_states = F.silu(float_normalized_states)
        else:
            pass
        if second_layer_norm is None:
            second_normalized_states = None
        else:
            second_normalized_states = F.layer_norm(
                float_normalized_states,
                (width,),
                second_layer_norm.weight,
                second_layer_norm.bias,
                layer_norm.eps,
            ).to(torch.bfloat16)
        return (
            sum_states,
            float_normalized_states.to(normalized_dtype),
            second_normalized_states,
        )
