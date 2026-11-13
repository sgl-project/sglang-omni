# SPDX-License-Identifier: Apache-2.0
"""Residual layer norm of the Zonos3 reference encoder's transformer stages, bit for bit with scaled_residual and the float32 nn.LayerNorm that bf16 autocast runs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from sglang_omni.models.zonos3.components.codec_kernels import scaled_residual
from sglang_omni.platforms import current_platform

# note (zhaochenyang20): torch's vectorized CUDA layer norm reduces a row with 4 warps of 32 lanes, each lane folding float4 loads into a Welford partial; the norm kernel replays that order, so its statistics and outputs match nn.LayerNorm bit for bit.
TORCH_LAYER_NORM_WARPS = 4
TORCH_LAYER_NORM_VECTOR_SIZE = 4
TORCH_LAYER_NORM_THREADS = 128


@triton.jit
def welford_combine(mean_b, m2_b, count_b, mean_a, m2_a, count_a):
    """torch's cuWelfordCombine(dataB, dataA): dataB is the merging lane's own partial, dataA its partner's."""
    delta = mean_b - mean_a
    count = count_a + count_b
    has_count = count > 0.0
    coefficient = libdevice.div_rn(1.0, tl.where(has_count, count, 1.0))
    weight_a = libdevice.mul_rn(count_a, coefficient)
    weight_b = libdevice.mul_rn(count_b, coefficient)
    mean = libdevice.fma(weight_a, mean_a, libdevice.mul_rn(weight_b, mean_b))
    m2 = libdevice.fma(
        libdevice.mul_rn(libdevice.mul_rn(delta, delta), count_a),
        weight_b,
        libdevice.add_rn(m2_a, m2_b),
    )
    return tl.where(has_count, mean, 0.0), tl.where(has_count, m2, 0.0), count


@triton.jit
def welford_step(states, mean, m2, count, is_vector, next_count):
    """torch's cuWelfordOnlineSum on the threads whose vector exists; every such thread has folded the same count."""
    delta = states - mean
    next_mean = libdevice.fma(delta, libdevice.div_rn(1.0, next_count), mean)
    next_m2 = libdevice.fma(delta, states - next_mean, m2)
    return (
        tl.where(is_vector, next_mean, mean),
        tl.where(is_vector, next_m2, m2),
        tl.where(is_vector, count + 1.0, count),
    )


@triton.jit
def row_states(
    residual_pointer,
    update_pointer,
    scale_pointer,
    row,
    vectors,
    is_vector,
    WIDTH: tl.constexpr,
    HAS_UPDATE: tl.constexpr,
):
    """(128, 4) float32 tile of one row at float4 vectors: residual + update * scale, rounded as scaled_residual rounds, or the residual alone."""
    columns = vectors[:, None] * 4 + tl.arange(0, 4)[None, :]
    states = tl.load(
        residual_pointer + row * WIDTH + columns, mask=is_vector[:, None], other=0.0
    ).to(tl.float32)
    if HAS_UPDATE:
        states = libdevice.add_rn(
            states,
            libdevice.mul_rn(
                tl.load(
                    update_pointer + row * WIDTH + columns,
                    mask=is_vector[:, None],
                    other=0.0,
                ).to(tl.float32),
                tl.load(scale_pointer + columns, mask=is_vector[:, None], other=0.0),
            ),
        )
    else:
        pass
    return states, columns


@triton.jit
def residual_norm_kernel(
    residual_pointer,
    update_pointer,
    scale_pointer,
    summed_pointer,
    norm_weight_pointer,
    norm_bias_pointer,
    normalized_pointer,
    statistics_pointer,
    epsilon,
    WIDTH: tl.constexpr,
    ITERATIONS: tl.constexpr,
    HAS_UPDATE: tl.constexpr,
):
    """One row: the float32 sum, stored with an update, then its layer norm in torch's reduction order, stored in bfloat16.

    Thread t folds float4 vectors t, t + 128, ... into its Welford partial lane by lane, and the partials merge as torch's shuffle-down tree does; the reduction's first thread holds that order, so it publishes the row statistics.
    """
    row = tl.program_id(0).to(tl.int64)
    threads = tl.arange(0, 128)
    mean = tl.zeros((128,), tl.float32)
    m2 = tl.zeros((128,), tl.float32)
    count = tl.zeros((128,), tl.float32)
    for iteration in tl.static_range(ITERATIONS):
        vectors = threads + iteration * 128
        is_vector = vectors < WIDTH // 4
        states, columns = row_states(
            residual_pointer,
            update_pointer,
            scale_pointer,
            row,
            vectors,
            is_vector,
            WIDTH,
            HAS_UPDATE,
        )
        if HAS_UPDATE:
            tl.store(
                summed_pointer + row * WIDTH + columns, states, mask=is_vector[:, None]
            )
        else:
            pass
        # Column pairs (0, 2) and (1, 3) of each float4 vector, then its four columns in order.
        outer_columns, inner_columns = tl.split(tl.reshape(states, (128, 2, 2)))
        column_0, column_2 = tl.split(outer_columns)
        column_1, column_3 = tl.split(inner_columns)
        mean, m2, count = welford_step(
            column_0, mean, m2, count, is_vector, iteration * 4.0 + 1.0
        )
        mean, m2, count = welford_step(
            column_1, mean, m2, count, is_vector, iteration * 4.0 + 2.0
        )
        mean, m2, count = welford_step(
            column_2, mean, m2, count, is_vector, iteration * 4.0 + 3.0
        )
        mean, m2, count = welford_step(
            column_3, mean, m2, count, is_vector, iteration * 4.0 + 4.0
        )
    row_mean, row_m2, _ = tl.reduce((mean, m2, count), 0, welford_combine)
    tl.store(statistics_pointer + row * 2, row_mean)
    tl.store(
        statistics_pointer + row * 2 + 1,
        libdevice.rsqrt(
            libdevice.add_rn(libdevice.div_rn(row_m2, WIDTH * 1.0), epsilon)
        ),
    )
    tl.debug_barrier()
    row_mean = tl.load(statistics_pointer + row * 2)
    reciprocal_deviation = tl.load(statistics_pointer + row * 2 + 1)
    for iteration in tl.static_range(ITERATIONS):
        vectors = threads + iteration * 128
        is_vector = vectors < WIDTH // 4
        states, columns = row_states(
            residual_pointer,
            update_pointer,
            scale_pointer,
            row,
            vectors,
            is_vector,
            WIDTH,
            HAS_UPDATE,
        )
        normalized_states = libdevice.fma(
            tl.load(norm_weight_pointer + columns, mask=is_vector[:, None], other=0.0),
            libdevice.mul_rn(reciprocal_deviation, states - row_mean),
            tl.load(norm_bias_pointer + columns, mask=is_vector[:, None], other=0.0),
        )
        tl.store(
            normalized_pointer + row * WIDTH + columns,
            normalized_states.to(tl.bfloat16),
            mask=is_vector[:, None],
        )


def residual_norm(
    residual: torch.Tensor,
    layer_norm: nn.LayerNorm,
    update: torch.Tensor | None,
    update_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(sum, bfloat16 norm of the sum) over the last dimension; the sum is scaled_residual(residual, update, update_scale) in float32, or the residual itself without an update.

    The norm is the float32 nn.LayerNorm that bf16 autocast runs, cast to the bfloat16 its next projection reads.
    """
    width = residual.shape[-1]
    if update is None:
        pass
    else:
        assert (
            update_scale is not None and update_scale.dtype == torch.float32
        ), update_scale
    if current_platform.is_cuda() and residual.is_cuda:
        assert (
            layer_norm.weight.dtype == torch.float32
            and width % TORCH_LAYER_NORM_VECTOR_SIZE == 0
        ), (
            layer_norm.weight.dtype,
            width,
        )
        residual = residual.contiguous()
        if update is None:
            summed = residual
        else:
            update = update.contiguous()
            summed = torch.empty(
                residual.shape, dtype=torch.float32, device=residual.device
            )
        normalized_states = torch.empty(
            residual.shape, dtype=torch.bfloat16, device=residual.device
        )
        row_count = residual.numel() // width
        residual_norm_kernel[(row_count,)](
            residual,
            residual if update is None else update,
            layer_norm.weight if update_scale is None else update_scale,
            summed,
            layer_norm.weight,
            layer_norm.bias,
            normalized_states,
            torch.empty((row_count, 2), dtype=torch.float32, device=residual.device),
            layer_norm.eps,
            WIDTH=width,
            ITERATIONS=triton.cdiv(
                width, TORCH_LAYER_NORM_VECTOR_SIZE * TORCH_LAYER_NORM_THREADS
            ),
            HAS_UPDATE=update is not None,
            num_warps=TORCH_LAYER_NORM_WARPS,
            enable_fp_fusion=False,
        )
        return summed, normalized_states
    else:
        if update is None:
            summed = residual
        else:
            summed = scaled_residual(residual, update, update_scale)
        return summed, F.layer_norm(
            summed.float(), (width,), layer_norm.weight, layer_norm.bias, layer_norm.eps
        ).to(torch.bfloat16)
