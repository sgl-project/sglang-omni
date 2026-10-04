# SPDX-License-Identifier: Apache-2.0
"""CuTe pointwise kernels for contiguous PersonaPlex singleton inference."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import tvm_ffi
from cuda.bindings.driver import CUstream
from cutlass.cute.math import RoundingMode
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.utils import SmemAllocator

from sglang_omni.models.personaplex.architecture import DEPFORMER

NORM_VECTOR_ELEMENTS = 4
NORM_THREADS = DEPFORMER.dim // NORM_VECTOR_ELEMENTS
WARP_THREADS = 32
GATE_BLOCK_ELEMENTS = 256
CUTE_COMPILE_OPTIONS = "--enable-tvm-ffi"
CUTE_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}


@cute.kernel
def rms_norm_f32_kernel(
    hidden_states: cute.Tensor,
    alpha: cute.Tensor,
    normalized_states: cute.Tensor,
    epsilon: cutlass.Float32,
) -> None:
    lane_index, _, _ = cute.arch.thread_idx()
    lane_sum = cutlass.Float32(0)
    # note (Codex): The FP32 mean's accumulation order affects BF16 rounding.
    for vector_index in cutlass.range_constexpr(NORM_VECTOR_ELEMENTS):
        column_index = lane_index * NORM_VECTOR_ELEMENTS + vector_index
        hidden_value = hidden_states[0, column_index].to(cutlass.Float32)
        squared_value = cute.math.mul(
            hidden_value, hidden_value, rounding=RoundingMode.NEAREST_EVEN
        )
        lane_sum = cute.math.add(
            lane_sum, squared_value, rounding=RoundingMode.NEAREST_EVEN
        )
    shared_sums = SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout((NORM_THREADS,))
    )
    shared_sums[lane_index] = lane_sum
    cute.arch.sync_threads()
    for reduction_stage in cutlass.range_constexpr(
        NORM_THREADS.bit_length() - WARP_THREADS.bit_length()
    ):
        reduction_offset = NORM_THREADS >> (reduction_stage + 1)
        if lane_index < reduction_offset:
            shared_sums[lane_index] = cute.math.add(
                shared_sums[lane_index],
                shared_sums[lane_index + reduction_offset],
                rounding=RoundingMode.NEAREST_EVEN,
            )
        else:
            pass
        cute.arch.sync_threads()
    lane_sum = shared_sums[lane_index % WARP_THREADS]
    for reduction_stage in cutlass.range_constexpr(WARP_THREADS.bit_length() - 1):
        shuffled_sum = cute.arch.shuffle_sync_bfly(
            lane_sum, offset=WARP_THREADS >> (reduction_stage + 1)
        )
        lane_sum = cute.math.add(
            lane_sum, shuffled_sum, rounding=RoundingMode.NEAREST_EVEN
        )
    inverse_rms = cute.math.rsqrt(lane_sum / DEPFORMER.dim + epsilon)
    for vector_index in cutlass.range_constexpr(NORM_VECTOR_ELEMENTS):
        column_index = lane_index + vector_index * NORM_THREADS
        hidden_value = hidden_states[0, column_index].to(cutlass.Float32)
        alpha_value = alpha[column_index].to(cutlass.Float32)
        normalized_states[0, column_index] = (
            hidden_value * (alpha_value * inverse_rms)
        ).to(normalized_states.element_type)


@cute.jit
def launch_rms_norm_f32(
    hidden_states: cute.Tensor,
    alpha: cute.Tensor,
    normalized_states: cute.Tensor,
    epsilon: cutlass.Float32,
    stream: CUstream,
) -> None:
    rms_norm_f32_kernel(hidden_states, alpha, normalized_states, epsilon).launch(
        grid=(1, 1, 1), block=(NORM_THREADS, 1, 1), stream=stream
    )


@cute.kernel
def silu_gate_kernel(gate_up: cute.Tensor, activated_states: cute.Tensor) -> None:
    block_index, _, _ = cute.arch.block_idx()
    lane_index, _, _ = cute.arch.thread_idx()
    column_index = block_index * GATE_BLOCK_ELEMENTS + lane_index
    gate = gate_up[0, column_index].to(cutlass.Float32)
    up = gate_up[0, column_index + DEPFORMER.ffn_hidden].to(cutlass.Float32)
    activated_gate = cute.math.div(
        gate,
        cutlass.Float32(1) + cute.math.exp(-gate),
        rounding=RoundingMode.NEAREST_EVEN,
    )
    # note (Codex): Eager SiLU rounds before the separate multiplication kernel.
    rounded_gate = activated_gate.to(gate_up.element_type).to(cutlass.Float32)
    activated_states[0, column_index] = (rounded_gate * up).to(
        activated_states.element_type
    )


@cute.jit
def launch_silu_gate(
    gate_up: cute.Tensor, activated_states: cute.Tensor, stream: CUstream
) -> None:
    silu_gate_kernel(gate_up, activated_states).launch(
        grid=(cute.ceil_div(DEPFORMER.ffn_hidden, GATE_BLOCK_ELEMENTS), 1, 1),
        block=(GATE_BLOCK_ELEMENTS, 1, 1),
        stream=stream,
    )


@cache
def compile_rms_norm_f32(
    dtype: torch.dtype, alpha_dtype: torch.dtype, device_index: int
) -> tvm_ffi.Function:
    with torch.cuda.device(device_index):
        hidden_states = make_fake_tensor(
            CUTE_DTYPES[dtype], (1, DEPFORMER.dim), (DEPFORMER.dim, 1)
        )
        return cute.compile(
            launch_rms_norm_f32,
            hidden_states,
            make_fake_tensor(CUTE_DTYPES[alpha_dtype], (DEPFORMER.dim,), (1,)),
            hidden_states,
            cutlass.Float32(0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options=CUTE_COMPILE_OPTIONS,
        )


@cache
def compile_silu_gate(dtype: torch.dtype, device_index: int) -> tvm_ffi.Function:
    with torch.cuda.device(device_index):
        return cute.compile(
            launch_silu_gate,
            make_fake_tensor(
                CUTE_DTYPES[dtype],
                (1, 2 * DEPFORMER.ffn_hidden),
                (2 * DEPFORMER.ffn_hidden, 1),
            ),
            make_fake_tensor(
                CUTE_DTYPES[dtype],
                (1, DEPFORMER.ffn_hidden),
                (DEPFORMER.ffn_hidden, 1),
            ),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options=CUTE_COMPILE_OPTIONS,
        )


def fused_rms_norm_f32(
    hidden_states: torch.Tensor, alpha: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Normalize contiguous singleton production inputs with FP32 arithmetic."""
    assert not torch.is_grad_enabled()
    assert hidden_states.shape == (1, DEPFORMER.dim) and hidden_states.is_contiguous()
    assert alpha.shape == (DEPFORMER.dim,) and alpha.is_contiguous()
    normalized_states = torch.empty_like(hidden_states)
    normalize = compile_rms_norm_f32(
        hidden_states.dtype, alpha.dtype, hidden_states.get_device()
    )
    normalize(hidden_states, alpha, normalized_states, epsilon)
    return normalized_states


def fused_silu_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """Activate a contiguous singleton production projection with eager rounding."""
    assert not torch.is_grad_enabled()
    assert gate_up.shape == (1, 2 * DEPFORMER.ffn_hidden) and gate_up.is_contiguous()
    activated_states = gate_up.new_empty((1, DEPFORMER.ffn_hidden))
    activate = compile_silu_gate(gate_up.dtype, gate_up.get_device())
    activate(gate_up, activated_states)
    return activated_states
