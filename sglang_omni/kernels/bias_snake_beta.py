# SPDX-License-Identifier: Apache-2.0
"""SnakeBeta of a conv's output plus its bias, and a residual when given, written into the
caller's buffer: the activation of a channels-last conv stack fused with the adds in front of
it, rounded where the eager adds and SnakeBeta round in the activations' dtype, or computed in
fp32 and rounded once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

VECTOR_BYTES = 16
VECTORS_PER_PROGRAM = 256


class SnakeBetaParameters(Protocol):
    """SnakeBeta's log-scale alpha and beta per channel, and the epsilon its reciprocal adds."""

    alpha: torch.Tensor
    beta: torch.Tensor
    no_div_by_zero: float


@dataclass(frozen=True, kw_only=True)
class SnakeBetaCoefficients:
    """A SnakeBeta's exp(alpha) and 1 / (exp(beta) + epsilon) per channel, computed by the ops
    its forward runs, so the kernel reads the values the forward computes."""

    alpha_scale: torch.Tensor
    inverse_beta: torch.Tensor

    @classmethod
    @torch.no_grad()
    def from_activation(cls, activation: SnakeBetaParameters) -> SnakeBetaCoefficients:
        return cls(
            alpha_scale=torch.exp(activation.alpha),
            inverse_beta=1.0 / (torch.exp(activation.beta) + activation.no_div_by_zero),
        )


@triton.jit(do_not_specialize=["row_vectors", "channel_vectors"])
def bias_snake_beta_channels_last_kernel(
    conv_ptr,
    conv_batch_stride,
    bias_ptr,
    residual_ptr,
    sum_ptr,
    out_ptr,
    out_batch_stride,
    alpha_scale_ptr,
    inverse_beta_ptr,
    row_vectors,
    channel_vectors,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_SUM: tl.constexpr,
    SHOULD_ROUND_LIKE_EAGER: tl.constexpr,
    VECTORS_PER_PROGRAM: tl.constexpr,
    VECTOR_ELEMENTS: tl.constexpr,
):
    # note (ratish): eager rounding lets the compiler pack bf16 pairs, about 6 % faster; the fp32
    # path rounds each output once and lands closer to fp32.
    out_dtype = out_ptr.dtype.element_ty
    batch = tl.program_id(1).to(tl.int64)
    vector = tl.program_id(0) * VECTORS_PER_PROGRAM + tl.arange(0, VECTORS_PER_PROGRAM)
    element = tl.arange(0, VECTOR_ELEMENTS)
    mask = (vector < row_vectors)[:, None]
    offsets = vector[:, None] * VECTOR_ELEMENTS + element[None, :]
    channel = (vector % channel_vectors)[:, None] * VECTOR_ELEMENTS + element[None, :]
    contiguous_base = batch * row_vectors * VECTOR_ELEMENTS
    alpha_scale = tl.load(alpha_scale_ptr + channel, mask=mask, other=0).to(tl.float32)
    inverse_beta = tl.load(inverse_beta_ptr + channel, mask=mask, other=0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + channel, mask=mask, other=0).to(tl.float32)
    conv_values = tl.load(
        conv_ptr + batch * conv_batch_stride + offsets, mask=mask, other=0
    ).to(tl.float32)
    if SHOULD_ROUND_LIKE_EAGER:
        total = (conv_values + bias).to(out_dtype).to(tl.float32)
    else:
        total = conv_values + bias
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + contiguous_base + offsets, mask=mask, other=0)
        total = total + residual.to(tl.float32)
        if SHOULD_ROUND_LIKE_EAGER:
            total = total.to(out_dtype).to(tl.float32)
        else:
            pass
    else:
        pass
    if WRITE_SUM:
        tl.store(sum_ptr + contiguous_base + offsets, total.to(out_dtype), mask=mask)
    else:
        pass
    if SHOULD_ROUND_LIKE_EAGER:
        phase = (total * alpha_scale).to(out_dtype).to(tl.float32)
        sine = libdevice.sin(phase).to(out_dtype).to(tl.float32)
        sine_squared = (sine * sine).to(out_dtype).to(tl.float32)
        scaled = (inverse_beta * sine_squared).to(out_dtype).to(tl.float32)
    else:
        sine = libdevice.sin(total * alpha_scale)
        scaled = inverse_beta * (sine * sine)
    tl.store(
        out_ptr + batch * out_batch_stride + offsets,
        (total + scaled).to(out_dtype),
        mask=mask,
    )


def fused_bias_snake_beta(
    conv_output: torch.Tensor,
    bias: torch.Tensor,
    coefficients: SnakeBetaCoefficients,
    out: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    sum_out: torch.Tensor | None = None,
    should_round_like_eager: bool = True,
) -> None:
    """SnakeBeta(conv_output + bias, + residual if given) into out, the sum into sum_out if
    given, rounded as the eager adds and SnakeBeta round, or with should_round_like_eager False
    in fp32 and rounded once. [B, L, C] activations on CUDA; the rows of conv_output and out
    may sit in longer buffers, residual and sum_out are contiguous.
    """
    batch_size, length, channels = conv_output.shape
    vector_elements = VECTOR_BYTES // out.element_size()
    assert (
        channels % vector_elements == 0
    ), f"{channels} channels split into {vector_elements}-element vectors"
    row_strides = (channels, 1)
    assert (
        conv_output.stride()[1:] == row_strides and out.stride()[1:] == row_strides
    ), "each row of conv_output and out is one contiguous span"
    assert all(
        tensor is None or tensor.is_contiguous() for tensor in (residual, sum_out)
    ), "residual and sum_out are contiguous"
    # note (ratish): each thread moves one 16-byte vector of a row's contiguous span and reads
    # its channels' parameters at the vector's offset modulo the channel count, so no lane
    # idles on a power-of-two channel tile.
    row_vectors = length * channels // vector_elements
    grid = (triton.cdiv(row_vectors, VECTORS_PER_PROGRAM), batch_size)
    bias_snake_beta_channels_last_kernel[grid](
        conv_output,
        conv_output.stride(0),
        bias,
        conv_output if residual is None else residual,
        out if sum_out is None else sum_out,
        out,
        out.stride(0),
        coefficients.alpha_scale,
        coefficients.inverse_beta,
        row_vectors,
        channels // vector_elements,
        HAS_RESIDUAL=residual is not None,
        WRITE_SUM=sum_out is not None,
        SHOULD_ROUND_LIKE_EAGER=should_round_like_eager,
        VECTORS_PER_PROGRAM=VECTORS_PER_PROGRAM,
        VECTOR_ELEMENTS=vector_elements,
        num_warps=VECTORS_PER_PROGRAM // 32,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
