# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""BF16 SnakeBeta with prepared coefficients and exact sine lookup."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.base_dsl.jit_executor import JitCompiledFunction
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from sglang.srt.utils.custom_op import register_custom_op

from sglang_omni.models.qwen3_tts.cutedsl.runtime import device_arch, supports_auxiliary
from sglang_omni.utils.snake_beta import FusedSnakeBeta, fuse_vocoder_decoder


@cache
def sine_table(device: torch.device) -> torch.Tensor:
    """Share one immutable 128 KiB table across inference modules on a device."""
    return (
        torch.arange(65536, device=device, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .sin()
    )


@cute.kernel
def snake_beta_kernel(
    x: cute.Tensor,
    alpha: cute.Tensor,
    inverse_beta: cute.Tensor,
    sine_table: cute.Tensor,
    output: cute.Tensor,
    elements: cutlass.Int64,
    channels: cutlass.Constexpr,
    length: cutlass.Int32,
    channels_last: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vector: cutlass.Constexpr,
) -> None:
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    offset = (cutlass.Int64(block) * threads + tid) * vector
    if offset < elements:
        values = (x.iterator.raw_ptr() + offset).load(
            count=vector, alignment=vector * 2
        )
        if cutlass.const_expr(channels_last):
            channel = offset % channels
            a = (alpha.iterator.raw_ptr() + channel).load(
                count=vector, alignment=vector * 2
            )
            r = (inverse_beta.iterator.raw_ptr() + channel).load(
                count=vector, alignment=vector * 2
            )
        else:
            channel = (offset // length) % channels
            a = alpha[channel]
            r = inverse_beta[channel]
        scaled = values * a
        if cutlass.const_expr(sine_table is not None):
            offsets = scaled.bitcast(cutlass.Uint16).to(cutlass.Int32)
            sine = sine_table.iterator.raw_ptr().masked_gather(
                offsets, cutlass.full((vector,), True, cutlass.Boolean)
            )
        else:
            sine = cute.math.sin(scaled.to(cutlass.Float32), fastmath=False).to(
                cutlass.BFloat16
            )
        square = sine * sine
        product = square * r
        result = values + product
        (output.iterator.raw_ptr() + offset).store(result, alignment=vector * 2)
    else:
        pass


@cute.jit
def snake_beta_host(
    x: cute.Tensor,
    alpha: cute.Tensor,
    inverse_beta: cute.Tensor,
    sine_table: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
    channels: cutlass.Constexpr,
    length: cutlass.Int32,
    elements: cutlass.Int64,
    channels_last: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vector: cutlass.Constexpr,
) -> None:
    # note (Hao Wang): CUTLASS 4.8 shape extraction narrows to Int32; pass numel separately.
    blocks = cutlass.Int32((elements + threads * vector - 1) // (threads * vector))
    snake_beta_kernel(
        x,
        alpha,
        inverse_beta,
        sine_table,
        output,
        elements,
        channels,
        length,
        channels_last,
        threads,
        vector,
    ).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1),
        stream=stream,
    )


@cache
def compile_kernel(
    channels: int,
    channels_last: bool,
    threads: int,
    vector: int,
    arch: str,
    lookup: bool = False,
) -> JitCompiledFunction:
    def fake(n: int) -> cute.Tensor:
        return make_fake_compact_tensor(cutlass.BFloat16, (n,), stride_order=(0,))

    # note (Hao Wang): full audio decoding varies in length; keep it out of the JIT cache key.
    activation = make_fake_compact_tensor(
        cutlass.BFloat16, (cutlass.sym_int64(divisibility=vector),), stride_order=(0,)
    )
    return cute.compile(
        snake_beta_host,
        activation,
        fake(channels),
        fake(channels),
        fake(65536) if lookup else None,
        activation,
        make_fake_stream(),
        channels,
        cutlass.Int32(0),
        cutlass.Int64(0),
        channels_last,
        threads,
        vector,
        options=f"--enable-tvm-ffi --gpu-arch={arch} --ptxas-options=--fmad=false",
    )


@register_custom_op(op_name="qtts_cutedsl_snake_beta", mutates_args=[], out_shape=0)
def launch(
    x: torch.Tensor,
    alpha: torch.Tensor,
    inverse_beta: torch.Tensor,
    sine_lookup: torch.Tensor,
) -> torch.Tensor:
    """Launch with prepared coefficients while remaining opaque to torch.compile."""
    channels_last = not x.is_contiguous()
    physical = x.transpose(1, 2) if channels_last else x
    output = torch.empty_like(physical)
    contiguous_width = x.shape[1] if channels_last else x.shape[2]
    vector = 4 if x.numel() <= 65536 else 8
    while contiguous_width % vector:
        vector //= 2
    table = sine_lookup if x.numel() >= 4 * 1024**2 else None
    if x.numel():
        compile_kernel(
            x.shape[1],
            channels_last,
            128,
            vector,
            device_arch(x.device),
            table is not None,
        )(
            physical.view(-1),
            alpha,
            inverse_beta,
            table,
            output.view(-1),
            cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
            x.shape[2],
            x.numel(),
        )
    else:
        pass
    return output.transpose(1, 2) if channels_last else output


class CuTeDSLSnakeBeta(FusedSnakeBeta):
    """Inference-only SnakeBeta retaining checkpoint parameters and prepared BF16 coefficients."""

    def __init__(self, original: torch.nn.Module) -> None:
        super().__init__(original)
        self.register_buffer(
            "scaled_alpha", self.alpha.detach().exp().contiguous(), persistent=False
        )
        self.register_buffer(
            "inverse_beta",
            (1.0 / (self.beta.detach().exp() + self.no_div_by_zero)).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "sine_lookup", sine_table(self.alpha.device), persistent=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 3
            and hidden_states.device == self.scaled_alpha.device
            and hidden_states.shape[1] == self.scaled_alpha.numel()
            and (
                hidden_states.is_contiguous()
                or hidden_states.transpose(1, 2).is_contiguous()
            )
        ):
            return launch(
                hidden_states, self.scaled_alpha, self.inverse_beta, self.sine_lookup
            )
        else:
            return super().forward(hidden_states)


def fuse_cutedsl_vocoder(decoder: torch.nn.Module) -> int:
    """Prepare supported SnakeBeta modules after loading weights on their final device."""
    replacements = 0
    for module in list(decoder.modules()):
        for name, child in list(module.named_children()):
            if (
                type(child).__name__
                in ("SnakeBeta", "Qwen3OmniMoeSnakeBeta", "FusedSnakeBeta")
                and supports_auxiliary(child.alpha)
                and child.beta.dtype == torch.bfloat16
                and child.alpha.ndim == 1
                and child.beta.shape == child.alpha.shape
            ):
                setattr(module, name, CuTeDSLSnakeBeta(child))
                replacements += 1
            else:
                pass
    return replacements + fuse_vocoder_decoder(decoder)
