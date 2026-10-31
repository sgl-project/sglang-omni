# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Hao Wang
"""Packed BF16 decode blocks with caller-owned token-major K/V caches."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.base_dsl.jit_executor import JitCompiledFunction
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from torch import nn

from sglang_omni.models.qwen3_tts.cutedsl.runtime import MAX_DECODE_ROWS, device_arch
from sglang_omni.models.qwen3_tts.cutedsl.sm90.post_attention import (
    post_attention_host as post_sm90,
)
from sglang_omni.models.qwen3_tts.cutedsl.sm90.post_attention_wide import (
    post_attention_host as post_wide_sm90,
)
from sglang_omni.models.qwen3_tts.cutedsl.sm90.pre_attention import (
    pre_attention_host as pre_sm90,
)
from sglang_omni.models.qwen3_tts.cutedsl.sm100.post_attention import (
    post_attention_host as post_sm100,
)
from sglang_omni.models.qwen3_tts.cutedsl.sm100.post_attention_wide import (
    post_attention_host as post_wide_sm100,
)
from sglang_omni.models.qwen3_tts.cutedsl.sm100.pre_attention import (
    pre_attention_host as pre_sm100,
)

PAYLOAD_EMPTY = -2147450880  # note (Hao Wang): paired BF16 negative-zero sentinels.
WORKSPACE_COUNTERS = {1024: 112, 2048: 96}


def fake_tensor(dtype: type[cutlass.Numeric], *shape: int | cute.SymInt) -> cute.Tensor:
    return make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape))))
    )


@cache
def compile_pre_attention(
    batch_tile: int,
    hidden_size: int,
    rope_dtype: torch.dtype,
    slot_dtype: torch.dtype,
    epsilon: float,
    architecture: str,
) -> JitCompiledFunction:
    batch_size = cutlass.sym_int64()
    cache_slots = cutlass.sym_int64()
    rope_positions = cutlass.sym_int64()
    rope_type = cutlass.Float32 if rope_dtype == torch.float32 else cutlass.BFloat16
    slot_type = cutlass.Int64 if slot_dtype == torch.int64 else cutlass.Int32
    return cute.compile(
        pre_sm90 if architecture == "sm_90a" else pre_sm100,
        fake_tensor(cutlass.BFloat16, batch_size, hidden_size),
        fake_tensor(cutlass.BFloat16, hidden_size),
        fake_tensor(cutlass.BFloat16, 64, hidden_size // 64, 64, 64),
        fake_tensor(cutlass.BFloat16, 128),
        fake_tensor(cutlass.BFloat16, 128),
        fake_tensor(rope_type, rope_positions, 128),
        fake_tensor(cutlass.Int64, batch_size),
        fake_tensor(cutlass.BFloat16, cache_slots, 8, 128),
        fake_tensor(cutlass.BFloat16, batch_size, 16, 1, 128),
        make_fake_stream(),
        batch_tile,
        epsilon,
        hidden_size,
        fake_tensor(slot_type, batch_size),
        fake_tensor(cutlass.BFloat16, cache_slots, 8, 128),
        options=f"--enable-tvm-ffi --gpu-arch={architecture} --ptxas-options=--fmad=false",
    )


@cache
def compile_post_attention(
    batch_tile: int, hidden_size: int, epsilon: float, architecture: str
) -> JitCompiledFunction:
    batch_size = cutlass.sym_int64()
    channel_tile = 128 if hidden_size == 2048 else 64
    gate_channels = channel_tile // 2
    if architecture == "sm_90a":
        kernel = post_wide_sm90 if hidden_size == 2048 else post_sm90
    else:
        kernel = post_wide_sm100 if hidden_size == 2048 else post_sm100
    return cute.compile(
        kernel,
        fake_tensor(cutlass.BFloat16, batch_size, 2048),
        fake_tensor(cutlass.BFloat16, batch_size, hidden_size),
        fake_tensor(
            cutlass.BFloat16, hidden_size // channel_tile, 32, channel_tile, 64
        ),
        fake_tensor(
            cutlass.BFloat16,
            3 * hidden_size // gate_channels,
            hidden_size // 64,
            channel_tile,
            64,
        ),
        fake_tensor(
            cutlass.BFloat16,
            hidden_size // channel_tile,
            3 * hidden_size // 64,
            channel_tile,
            64,
        ),
        fake_tensor(cutlass.BFloat16, hidden_size),
        fake_tensor(cutlass.Int32, 2, batch_tile, hidden_size // 2),
        fake_tensor(
            cutlass.Int32,
            2,
            3 * hidden_size // gate_channels,
            batch_tile,
            gate_channels // 2,
        ),
        fake_tensor(cutlass.Int32, WORKSPACE_COUNTERS[hidden_size]),
        fake_tensor(cutlass.BFloat16, batch_size, hidden_size),
        make_fake_stream(),
        batch_tile,
        epsilon,
        options=f"--enable-tvm-ffi --gpu-arch={architecture}",
    )


class SlotPreAttention(nn.Module):
    """RMSNorm, QKV, head norms, RoPE and direct cache-slot stores."""

    def __init__(
        self,
        norm_weight: torch.Tensor,
        qkv_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.hidden_size = norm_weight.numel()
        self.architecture = device_arch(norm_weight.device)
        self.epsilon = epsilon
        self.register_buffer("norm_weight", norm_weight.detach())
        self.register_buffer("q_norm_weight", q_norm_weight.detach())
        self.register_buffer("k_norm_weight", k_norm_weight.detach())
        self.register_buffer(
            "packed_weights",
            qkv_weight.detach()
            .reshape(64, 64, self.hidden_size // 64, 64)
            .permute(0, 2, 1, 3)
            .contiguous(),
        )

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rope_cache: torch.Tensor,
        cache_slots: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        assert 1 <= batch_size <= MAX_DECODE_ROWS
        queries = torch.empty(
            (batch_size, 16, 1, 128),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        compile_pre_attention(
            16 if batch_size <= 16 else 32,
            self.hidden_size,
            rope_cache.dtype,
            cache_slots.dtype,
            self.epsilon,
            self.architecture,
        )(
            hidden_states,
            self.norm_weight,
            self.packed_weights,
            self.q_norm_weight,
            self.k_norm_weight,
            rope_cache,
            positions,
            key_cache,
            queries,
            cuda.CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream),
            cache_slots,
            value_cache,
        )
        return queries.reshape(batch_size, 2048)


class PackedPostAttention(nn.Module):
    """Output projection, residual, RMSNorm and MLP with a fully resident cooperative grid.

    Calls and graph replays sharing an instance must be serialized; concurrent streams need separate instances. Warm each batch size before graph capture.
    """

    def __init__(
        self,
        output_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.hidden_size = output_weight.shape[0]
        self.architecture = device_arch(output_weight.device)
        self.epsilon = epsilon
        self.register_buffer("norm_weight", norm_weight.detach())
        channel_tile = 128 if self.hidden_size == 2048 else 64
        self.gate_channels = channel_tile // 2
        tiles = 3 * self.hidden_size // self.gate_channels
        gate_up = torch.cat(
            (
                gate_weight.reshape(tiles, self.gate_channels, self.hidden_size),
                up_weight.reshape(tiles, self.gate_channels, self.hidden_size),
            ),
            dim=1,
        ).detach()
        self.register_buffer(
            "packed_gate_up",
            gate_up.reshape(tiles, channel_tile, self.hidden_size // 64, 64)
            .permute(0, 2, 1, 3)
            .contiguous(),
        )
        self.register_buffer(
            "packed_output",
            output_weight.detach()
            .reshape(self.hidden_size // channel_tile, channel_tile, 32, 64)
            .permute(0, 2, 1, 3)
            .contiguous(),
        )
        self.register_buffer(
            "packed_down",
            down_weight.detach()
            .reshape(
                self.hidden_size // channel_tile,
                channel_tile,
                3 * self.hidden_size // 64,
                64,
            )
            .permute(0, 2, 1, 3)
            .contiguous(),
        )
        self.workspaces: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @torch.compiler.disable
    def forward(self, attention: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        batch_size = attention.shape[0]
        assert 1 <= batch_size <= MAX_DECODE_ROWS
        batch_tile = 16 if batch_size <= 16 else 32
        if batch_size not in self.workspaces:
            self.workspaces[batch_size] = (
                torch.full(
                    (2, batch_tile, self.hidden_size // 2),
                    PAYLOAD_EMPTY,
                    device=attention.device,
                    dtype=torch.int32,
                ),
                torch.full(
                    (
                        2,
                        3 * self.hidden_size // self.gate_channels,
                        batch_tile,
                        self.gate_channels // 2,
                    ),
                    PAYLOAD_EMPTY,
                    device=attention.device,
                    dtype=torch.int32,
                ),
                torch.zeros(
                    WORKSPACE_COUNTERS[self.hidden_size],
                    device=attention.device,
                    dtype=torch.int32,
                ),
            )
        else:
            pass
        output = torch.empty_like(residual, memory_format=torch.contiguous_format)
        compile_post_attention(
            batch_tile, self.hidden_size, self.epsilon, self.architecture
        )(
            attention.reshape(batch_size, 2048),
            residual.reshape(batch_size, self.hidden_size),
            self.packed_output,
            self.packed_gate_up,
            self.packed_down,
            self.norm_weight,
            *self.workspaces[batch_size],
            output.reshape(batch_size, self.hidden_size),
            cuda.CUstream(torch.cuda.current_stream(attention.device).cuda_stream),
        )
        return output
