# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Causal BF16 Conv1D with bias, SnakeBeta, and residual epilogues.
The convolution output rounds to BF16 before bias addition, matching Torch's cuDNN path.
"""

import math
from functools import cache

import cutlass
import torch
from cuda.bindings import driver
from cutlass import cute
from cutlass.base_dsl.jit_executor import JitCompiledFunction
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from sglang.srt.utils.custom_op import register_custom_op

from sglang_omni.models.qwen3_tts.cutedsl.runtime import device_arch
from sglang_omni.models.qwen3_tts.cutedsl.sm90.conv1d import Conv1dLaunch as Conv1dSm90
from sglang_omni.models.qwen3_tts.cutedsl.sm100.conv1d import (
    Conv1dLaunch as Conv1dSm100,
)
from sglang_omni.models.qwen3_tts.cutedsl.snake_beta import CuTeDSLSnakeBeta


@register_custom_op(op_name="qtts_cutedsl_pointwise_conv", mutates_args=[], out_shape=0)
def pointwise_conv(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """Preserve the fused pointwise convolution through incremental decoder compilation."""
    return conv1d(hidden_states, weight, bias, 1, residual=residual)


@cache
def compile_kernel(
    weight_shape: tuple[int, int, int],
    dilation: int,
    channel_tile: int,
    stages: int,
    has_bias: bool,
    has_residual: bool,
    has_activation: bool,
    full_rows: bool,
    architecture: str,
) -> JitCompiledFunction:
    output_channels, kernel_size, input_channels = weight_shape
    reduction_tile = 32 if input_channels == 96 else 64
    store_channels = 96 if channel_tile == 96 else 64 if kernel_size == 1 else 32

    def fake(elements: int | cute.SymInt) -> cute.Tensor:
        return make_fake_compact_tensor(
            cutlass.BFloat16, (elements,), stride_order=(0,), assumed_align=16
        )

    # note (Hao Wang): Reuse binaries across audio lengths, retaining a tail-mask specialization.
    activation = fake(cutlass.sym_int64(divisibility=8))
    output = fake(cutlass.sym_int64(divisibility=8))
    launch = Conv1dSm90 if architecture == "sm_90a" else Conv1dSm100
    return cute.compile(
        launch(
            weight_shape,
            dilation,
            channel_tile,
            stages,
            reduction_tile,
            store_channels,
            full_rows,
        ),
        activation,
        fake(math.prod(weight_shape)),
        output,
        fake(output_channels) if has_bias else None,
        output if has_residual else None,
        fake(output_channels) if has_activation else None,
        fake(output_channels) if has_activation else None,
        fake(65536) if has_activation else None,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        options=f"--enable-tvm-ffi --gpu-arch={architecture} --ptxas-options=--fmad=false",
    )


def conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dilation: int,
    activation: CuTeDSLSnakeBeta | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convolve contiguous BTC activations with prepared OWC weights; stride and groups are one."""
    batch_size, sequence_length, input_channels = hidden_states.shape
    output_channels, kernel_size, _ = weight.shape
    architecture = device_arch(hidden_states.device)
    row_tile = 64 if architecture == "sm_90a" else 128
    multiprocessors = torch.cuda.get_device_properties(
        hidden_states.device
    ).multi_processor_count
    if architecture != "sm_90a":
        if (
            kernel_size == 1
            and input_channels == output_channels
            and batch_size
            * math.ceil(sequence_length / row_tile)
            * math.ceil(output_channels / 64)
            <= multiprocessors
        ):
            # note (Hao Wang): Short streaming chunks fit one wave with narrower channel tiles.
            channel_tile, stages, blocks_per_multiprocessor = 64, 4, 1
        elif output_channels == 96:
            channel_tile, stages, blocks_per_multiprocessor = (
                96,
                2 if kernel_size == 1 else 4,
                2,
            )
        elif input_channels == output_channels == 192 and kernel_size == 1:
            channel_tile, stages, blocks_per_multiprocessor = 64, 2, 2
        elif input_channels == output_channels == 384 and kernel_size == 1:
            channel_tile, stages, blocks_per_multiprocessor = 128, 2, 2
        else:
            channel_tile, stages, blocks_per_multiprocessor = (
                192 if input_channels == output_channels else 128,
                4,
                1,
            )
            if (
                batch_size
                * math.ceil(sequence_length / row_tile)
                * math.ceil(output_channels / channel_tile)
                < multiprocessors // 2
            ):
                channel_tile = 128 if input_channels == output_channels else 64
            else:
                pass
    elif output_channels == 96:
        channel_tile, stages, blocks_per_multiprocessor = 96, 4, 2
    elif (
        input_channels == output_channels
        and output_channels in (192, 384)
        and kernel_size == 1
    ):
        channel_tile, stages, blocks_per_multiprocessor = 64, 4, 2
    else:
        channel_tile, stages, blocks_per_multiprocessor = (
            192,
            4 if kernel_size == 1 else 6,
            1,
        )
        if (
            batch_size
            * math.ceil(sequence_length / 64)
            * math.ceil(output_channels / channel_tile)
            < multiprocessors // 2
        ):
            channel_tile = 64
        else:
            pass
    tiles = (
        batch_size
        * math.ceil(sequence_length / row_tile)
        * math.ceil(output_channels / channel_tile)
    )
    output = hidden_states.new_empty((batch_size, sequence_length, output_channels))
    compile_kernel(
        tuple(weight.shape),
        dilation,
        channel_tile,
        stages,
        bias is not None,
        residual is not None,
        activation is not None,
        sequence_length % row_tile == 0,
        architecture,
    )(
        hidden_states.view(-1),
        weight.view(-1),
        output.view(-1),
        bias,
        residual.view(-1) if residual is not None else None,
        activation.scaled_alpha if activation is not None else None,
        activation.inverse_beta if activation is not None else None,
        activation.sine_lookup if activation is not None else None,
        driver.CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream),
        batch_size,
        sequence_length,
        min(tiles, blocks_per_multiprocessor * multiprocessors),
    )
    return output
