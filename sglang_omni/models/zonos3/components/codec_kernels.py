# SPDX-License-Identifier: Apache-2.0
"""Triton kernels shared by the Zonos3 codec modules: transformer elementwise chains and the quantizers' nearest-code search.

Each kernel runs on the CUDA platform, whose libdevice has the correctly rounded intrinsics it calls, and rounds exactly where the eager chain rounds, so its outputs match the eager ops that run elsewhere, ROCm included, bit for bit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang_omni.platforms import current_platform

TRITON_DTYPES: dict[torch.dtype, tl.dtype] = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
}
# F.normalize's default floor on the norm.
NORMALIZE_EPSILON = 1e-12
# note (zhaochenyang20): one program per frame scanning all 4096 entries with 8 warps found a 4.8 s reference's codes in 3.9 us per codebook on one H100; frame blocks rescanned the codebook in serial chunks and ran 5-20 us.
NEAREST_CODE_WARPS = 8


# note (zhaochenyang20): the vocoder meets new trunk shapes while serving; unspecialized sizes keep it to one compiled variant per width and dtype.
@triton.jit(
    do_not_specialize=[
        "batch_size",
        "frame_count",
        "head_count",
        "state_frames",
        "leading_frames",
        "rotary_batch_frames",
    ]
)
def rotary_states_kernel(
    query_key_value_pointer,
    rotary_cos_pointer,
    rotary_signed_sin_pointer,
    states_pointer,
    batch_size,
    frame_count,
    head_count,
    state_frames,
    leading_frames,
    rotary_batch_frames,
    MODEL_DIMENSION: tl.constexpr,
    HEAD_DIMENSION: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    """One (batch, frame) row of the query, key or value projection into its (B, H, state frames, head) states.

    A batch row's rotary tables start rotary_batch_frames frames after the previous row's: 0 when the batch shares them.
    """
    row = tl.program_id(0)
    projection_index = tl.program_id(1)
    batch_index = row // frame_count
    frame = row % frame_count
    channels = tl.arange(0, BLOCK_CHANNELS)
    is_channel = channels < MODEL_DIMENSION
    head_channels = channels % HEAD_DIMENSION
    row_pointer = (
        query_key_value_pointer
        + row.to(tl.int64) * (3 * MODEL_DIMENSION)
        + projection_index * MODEL_DIMENSION
    )
    projection_values = tl.load(row_pointer + channels, mask=is_channel, other=0.0)
    if projection_index < 2:
        partner_values = tl.load(
            row_pointer + (channels ^ 1), mask=is_channel, other=0.0
        )
        rotary_offsets = (
            batch_index * rotary_batch_frames + frame
        ) * HEAD_DIMENSION + head_channels
        rotary_cos = tl.load(
            rotary_cos_pointer + rotary_offsets, mask=is_channel, other=0.0
        )
        rotary_signed_sin = tl.load(
            rotary_signed_sin_pointer + rotary_offsets, mask=is_channel, other=0.0
        )
        states = libdevice.add_rn(
            libdevice.mul_rn(projection_values.to(tl.float32), rotary_cos),
            libdevice.mul_rn(partner_values.to(tl.float32), rotary_signed_sin),
        ).to(states_pointer.dtype.element_ty)
    else:
        states = projection_values.to(states_pointer.dtype.element_ty)
    head_rows = (
        projection_index * batch_size + batch_index
    ) * head_count + channels // HEAD_DIMENSION
    tl.store(
        states_pointer
        + (head_rows.to(tl.int64) * state_frames + leading_frames + frame)
        * HEAD_DIMENSION
        + head_channels,
        states,
        mask=is_channel,
    )


def write_rotary_states(
    query_key_value: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_signed_sin: torch.Tensor,
    states: torch.Tensor,
    leading_frames: int,
) -> None:
    """Write (B, T, 3 * D) projections into frames [leading_frames, leading_frames + T) of (3, B, H, frames, head) states.

    Query and key are rotated in float32: each channel times rotary_cos plus its (real, imaginary) partner channel times rotary_signed_sin, tables of (T, head) shared by the batch or (B, 1, T, head) per batch row; the three are then copied into the states' dtype.
    """
    batch_size, frame_count, _ = query_key_value.shape
    _, _, head_count, state_frames, head_dimension = states.shape
    if current_platform.is_cuda() and query_key_value.is_cuda:
        # note (zhaochenyang20): one kernel instead of six per layer; with the fused residual adds a 4.8 s reference's 80 encoder transformer layers run in 6.5 ms instead of 8.8 ms on one H100, bit for bit.
        assert query_key_value.is_contiguous() and states.is_contiguous()
        if rotary_cos.dim() == 2:
            rotary_batch_frames = 0
        else:
            assert rotary_cos.shape == (batch_size, 1, frame_count, head_dimension)
            rotary_batch_frames = frame_count
        rotary_states_kernel[(batch_size * frame_count, 3)](
            query_key_value,
            rotary_cos.contiguous(),
            rotary_signed_sin.contiguous(),
            states,
            batch_size,
            frame_count,
            head_count,
            state_frames,
            leading_frames,
            rotary_batch_frames,
            MODEL_DIMENSION=head_count * head_dimension,
            HEAD_DIMENSION=head_dimension,
            BLOCK_CHANNELS=triton.next_power_of_2(head_count * head_dimension),
            enable_fp_fusion=False,
        )
    else:
        query_key_value = query_key_value.view(
            batch_size, frame_count, 3, head_count, head_dimension
        ).permute(2, 0, 3, 1, 4)
        query_key = query_key_value[:2]
        rotated_query_key = torch.empty(
            query_key.shape, dtype=torch.float32, device=query_key.device
        )
        torch.mul(query_key, rotary_cos, out=rotated_query_key)
        rotated_query_key.add_(
            query_key.unflatten(-1, (head_dimension // 2, 2)).flip(-1).flatten(-2)
            * rotary_signed_sin
        )
        frame_states = states[:, :, :, leading_frames : leading_frames + frame_count]
        frame_states[:2].copy_(rotated_query_key)
        frame_states[2].copy_(query_key_value[2])


@triton.jit
def scaled_residual_kernel(
    hidden_states_pointer,
    update_pointer,
    scale_pointer,
    output_pointer,
    MODEL_DIMENSION: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    PRODUCT_DTYPE: tl.constexpr,
):
    """One row of hidden_states + update * scale, the product rounded to PRODUCT_DTYPE as eager type promotion does."""
    row = tl.program_id(0).to(tl.int64)
    channels = tl.arange(0, BLOCK_CHANNELS)
    is_channel = channels < MODEL_DIMENSION
    hidden_states = tl.load(
        hidden_states_pointer + row * MODEL_DIMENSION + channels,
        mask=is_channel,
        other=0.0,
    )
    update = tl.load(
        update_pointer + row * MODEL_DIMENSION + channels, mask=is_channel, other=0.0
    )
    scale = tl.load(scale_pointer + channels, mask=is_channel, other=0.0)
    product = (
        libdevice.mul_rn(update.to(tl.float32), scale.to(tl.float32))
        .to(PRODUCT_DTYPE)
        .to(tl.float32)
    )
    tl.store(
        output_pointer + row * MODEL_DIMENSION + channels,
        libdevice.add_rn(hidden_states.to(tl.float32), product).to(
            output_pointer.dtype.element_ty
        ),
        mask=is_channel,
    )


def scaled_residual(
    hidden_states: torch.Tensor, update: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """hidden_states + update * scale for (..., D) states and a (D,) scale, in the eager ops' dtypes and rounding."""
    if current_platform.is_cuda() and hidden_states.is_cuda:
        product_dtype = torch.result_type(update, scale)
        hidden_states = hidden_states.contiguous()
        update = update.contiguous()
        output = torch.empty(
            hidden_states.shape,
            dtype=torch.promote_types(hidden_states.dtype, product_dtype),
            device=hidden_states.device,
        )
        model_dimension = hidden_states.shape[-1]
        scaled_residual_kernel[(hidden_states.numel() // model_dimension,)](
            hidden_states,
            update,
            scale,
            output,
            MODEL_DIMENSION=model_dimension,
            BLOCK_CHANNELS=triton.next_power_of_2(model_dimension),
            PRODUCT_DTYPE=TRITON_DTYPES[product_dtype],
            enable_fp_fusion=False,
        )
        return output
    else:
        return hidden_states + update * scale


@triton.jit
def straight_through_store(output_pointer, projected, codebook_pointer):
    """Store projected + (selected - projected) for one codebook dimension, rounded as the eager adds round."""
    tl.store(
        output_pointer,
        libdevice.add_rn(
            projected, libdevice.sub_rn(tl.load(codebook_pointer), projected)
        ),
    )


@triton.jit
def nearest_code_kernel(
    projected_pointer,
    codebook_columns_pointer,
    codebook_squares_pointer,
    codebook_pointer,
    codes_pointer,
    straight_through_pointer,
    frame_count,
    projected_batch_stride,
    projected_channel_stride,
    projected_frame_stride,
    CODEBOOK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    """One (frame, batch): the cosine nearest code of an 8-dimensional projection and its straight-through latent.

    The norm, square sum and dot products replay the eager reductions: four accumulators over elements i and i + 4 combined in order, and one fused multiply-add chain per dot product.
    """
    frame = tl.program_id(0)
    batch_index = tl.program_id(1).to(tl.int64)
    frame_pointer = (
        projected_pointer
        + batch_index * projected_batch_stride
        + frame * projected_frame_stride
    )
    x0 = tl.load(frame_pointer)
    x1 = tl.load(frame_pointer + projected_channel_stride)
    x2 = tl.load(frame_pointer + 2 * projected_channel_stride)
    x3 = tl.load(frame_pointer + 3 * projected_channel_stride)
    x4 = tl.load(frame_pointer + 4 * projected_channel_stride)
    x5 = tl.load(frame_pointer + 5 * projected_channel_stride)
    x6 = tl.load(frame_pointer + 6 * projected_channel_stride)
    x7 = tl.load(frame_pointer + 7 * projected_channel_stride)
    norm_squared = libdevice.add_rn(
        libdevice.add_rn(
            libdevice.add_rn(
                libdevice.fma(x4, x4, libdevice.mul_rn(x0, x0)),
                libdevice.fma(x5, x5, libdevice.mul_rn(x1, x1)),
            ),
            libdevice.fma(x6, x6, libdevice.mul_rn(x2, x2)),
        ),
        libdevice.fma(x7, x7, libdevice.mul_rn(x3, x3)),
    )
    denominator = tl.maximum(libdevice.sqrt_rn(norm_squared), EPSILON)
    q0 = libdevice.div_rn(x0, denominator)
    q1 = libdevice.div_rn(x1, denominator)
    q2 = libdevice.div_rn(x2, denominator)
    q3 = libdevice.div_rn(x3, denominator)
    q4 = libdevice.div_rn(x4, denominator)
    q5 = libdevice.div_rn(x5, denominator)
    q6 = libdevice.div_rn(x6, denominator)
    q7 = libdevice.div_rn(x7, denominator)
    query_squares = libdevice.add_rn(
        libdevice.add_rn(
            libdevice.add_rn(
                libdevice.add_rn(libdevice.mul_rn(q0, q0), libdevice.mul_rn(q4, q4)),
                libdevice.add_rn(libdevice.mul_rn(q1, q1), libdevice.mul_rn(q5, q5)),
            ),
            libdevice.add_rn(libdevice.mul_rn(q2, q2), libdevice.mul_rn(q6, q6)),
        ),
        libdevice.add_rn(libdevice.mul_rn(q3, q3), libdevice.mul_rn(q7, q7)),
    )
    entries = tl.arange(0, CODEBOOK_SIZE)
    column_pointers = codebook_columns_pointer + entries
    doubled_dot_products = libdevice.mul_rn(2.0 * q0, tl.load(column_pointers))
    doubled_dot_products = libdevice.fma(
        2.0 * q1, tl.load(column_pointers + CODEBOOK_SIZE), doubled_dot_products
    )
    doubled_dot_products = libdevice.fma(
        2.0 * q2, tl.load(column_pointers + 2 * CODEBOOK_SIZE), doubled_dot_products
    )
    doubled_dot_products = libdevice.fma(
        2.0 * q3, tl.load(column_pointers + 3 * CODEBOOK_SIZE), doubled_dot_products
    )
    doubled_dot_products = libdevice.fma(
        2.0 * q4, tl.load(column_pointers + 4 * CODEBOOK_SIZE), doubled_dot_products
    )
    doubled_dot_products = libdevice.fma(
        2.0 * q5, tl.load(column_pointers + 5 * CODEBOOK_SIZE), doubled_dot_products
    )
    doubled_dot_products = libdevice.fma(
        2.0 * q6, tl.load(column_pointers + 6 * CODEBOOK_SIZE), doubled_dot_products
    )
    doubled_dot_products = libdevice.fma(
        2.0 * q7, tl.load(column_pointers + 7 * CODEBOOK_SIZE), doubled_dot_products
    )
    code = tl.argmax(
        -libdevice.add_rn(
            libdevice.sub_rn(query_squares, doubled_dot_products),
            tl.load(codebook_squares_pointer + entries),
        ),
        axis=0,
        tie_break_left=True,
    ).to(tl.int64)
    tl.store(codes_pointer + batch_index * frame_count + frame, code)
    entry_pointer = codebook_pointer + code * 8
    latent_pointer = straight_through_pointer + batch_index * 8 * frame_count + frame
    straight_through_store(latent_pointer, x0, entry_pointer)
    straight_through_store(latent_pointer + frame_count, x1, entry_pointer + 1)
    straight_through_store(latent_pointer + 2 * frame_count, x2, entry_pointer + 2)
    straight_through_store(latent_pointer + 3 * frame_count, x3, entry_pointer + 3)
    straight_through_store(latent_pointer + 4 * frame_count, x4, entry_pointer + 4)
    straight_through_store(latent_pointer + 5 * frame_count, x5, entry_pointer + 5)
    straight_through_store(latent_pointer + 6 * frame_count, x6, entry_pointer + 6)
    straight_through_store(latent_pointer + 7 * frame_count, x7, entry_pointer + 7)


def nearest_codes(
    projected: torch.Tensor,
    codebook_columns: torch.Tensor,
    codebook_squares: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T) cosine nearest codes of (B, 8, T) float32 projections and their (B, 8, T) straight-through latents projected + (selected - projected).

    codebook_columns is the (8, codebook size) transpose of the L2-normalized codebook and codebook_squares its squared entry norms; distances are |q|^2 - 2 q e + |e|^2 of the normalized projection q, as the eager chain computes them.
    """
    batch_size, codebook_dimension, frame_count = projected.shape
    codebook_size = codebook.shape[0]
    assert codebook_dimension == 8 and projected.dtype == torch.float32, projected.shape
    if current_platform.is_cuda() and projected.is_cuda:
        # note (zhaochenyang20): one kernel replaces the normalize, square-sum, matmul, distance, argmax, gather and straight-through kernels of each codebook; a 4.8 s reference's ten codebooks with the semantic head replayed in 0.43 ms instead of 0.90 ms on one H100, bit for bit.
        codes = torch.empty(
            (batch_size, frame_count), dtype=torch.int64, device=projected.device
        )
        straight_through = torch.empty(
            projected.shape, dtype=torch.float32, device=projected.device
        )
        nearest_code_kernel[(frame_count, batch_size)](
            projected,
            codebook_columns,
            codebook_squares,
            codebook,
            codes,
            straight_through,
            frame_count,
            *projected.stride(),
            CODEBOOK_SIZE=codebook_size,
            EPSILON=NORMALIZE_EPSILON,
            num_warps=NEAREST_CODE_WARPS,
            enable_fp_fusion=False,
        )
        return codes, straight_through
    else:
        queries = F.normalize(
            projected.transpose(1, 2).reshape(-1, codebook_dimension), dim=-1
        )
        squared_distances = (
            queries.pow(2).sum(1, keepdim=True)
            - 2 * queries @ codebook_columns
            + codebook_squares[None]
        )
        codes = (-squared_distances).argmax(1).view(batch_size, -1)
        selected_entries = F.embedding(codes, codebook).transpose(1, 2)
        return codes, projected + (selected_entries - projected)
