# SPDX-License-Identifier: Apache-2.0
"""Fused CUDA paths of the Zonos3 codec decoder: Triton norm kernels, FA3 local-window attention, matmul resampling."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from sglang_omni.models.zonos3.components.codec_kernels import (
    scaled_residual,
    write_rotary_states,
)
from sglang_omni.models.zonos3.components.codec_layers import ConvNeXtBlock
from sglang_omni.models.zonos3.components.codec_transformer import (
    ProjectedTransformer,
    rotary_tables,
)
from sglang_omni.platforms import current_platform

if current_platform.is_cuda():
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
else:
    pass

# note (zhaochenyang20): 4 frames and 4 warps per program ran the iSTFT head's eight blocks over 1200 frames in 258 us instead of 506 us eager on one H100; 1 to 8 frames and 8 warps were no faster.
CONVNEXT_BLOCK_FRAMES = 4
CONVNEXT_WARPS = 4


@triton.jit
def scaled_residual_norm_kernel(
    hidden_states_pointer,
    update_pointer,
    scale_pointer,
    norm_weight_pointer,
    norm_bias_pointer,
    output_pointer,
    normalized_pointer,
    epsilon,
    MODEL_DIMENSION: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    """One row of hidden_states + update * scale, rounded like the eager ops, and its layer norm."""
    row = tl.program_id(0).to(tl.int64)
    channels = tl.arange(0, BLOCK_CHANNELS)
    is_channel = channels < MODEL_DIMENSION
    offsets = row * MODEL_DIMENSION + channels
    output_dtype = output_pointer.dtype.element_ty
    hidden_states = tl.load(
        hidden_states_pointer + offsets, mask=is_channel, other=0.0
    ).to(tl.float32)
    update = tl.load(update_pointer + offsets, mask=is_channel, other=0.0).to(
        tl.float32
    )
    scale = tl.load(scale_pointer + channels, mask=is_channel, other=0.0).to(tl.float32)
    product = libdevice.mul_rn(update, scale).to(output_dtype).to(tl.float32)
    output = libdevice.add_rn(hidden_states, product).to(output_dtype)
    tl.store(output_pointer + offsets, output, mask=is_channel)
    float_output = output.to(tl.float32)
    mean = tl.sum(float_output, axis=0) / MODEL_DIMENSION
    centered = tl.where(is_channel, float_output - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / MODEL_DIMENSION
    norm_weight = tl.load(
        norm_weight_pointer + channels, mask=is_channel, other=0.0
    ).to(tl.float32)
    norm_bias = tl.load(norm_bias_pointer + channels, mask=is_channel, other=0.0).to(
        tl.float32
    )
    normalized = (
        norm_weight * (libdevice.rsqrt(variance + epsilon) * centered) + norm_bias
    )
    tl.store(normalized_pointer + offsets, normalized.to(output_dtype), mask=is_channel)


def scaled_residual_norm(
    hidden_states: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    layer_norm: nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T, D) hidden_states + update * scale and its layer norm, which the next projection reads."""
    assert hidden_states.dtype == update.dtype == scale.dtype, (
        hidden_states.dtype,
        update.dtype,
        scale.dtype,
    )
    hidden_states = hidden_states.contiguous()
    update = update.contiguous()
    model_dimension = hidden_states.shape[-1]
    output = torch.empty_like(hidden_states)
    normalized = torch.empty_like(hidden_states)
    scaled_residual_norm_kernel[(hidden_states.numel() // model_dimension,)](
        hidden_states,
        update,
        scale,
        layer_norm.weight,
        layer_norm.bias,
        output,
        normalized,
        layer_norm.eps,
        MODEL_DIMENSION=model_dimension,
        BLOCK_CHANNELS=triton.next_power_of_2(model_dimension),
    )
    return output, normalized


@triton.jit(do_not_specialize=["frame_count"])
def depthwise_norm_kernel(
    hidden_states_pointer,
    convolution_weight_pointer,
    convolution_bias_pointer,
    norm_weight_pointer,
    norm_bias_pointer,
    normalized_pointer,
    frame_count,
    epsilon,
    CHANNELS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_FRAMES: tl.constexpr,
):
    """Causal depthwise convolution of BLOCK_FRAMES frames of (B, T, C) states, rounded to their dtype, then layer norm over C."""
    batch_index = tl.program_id(0).to(tl.int64)
    frame_positions = tl.program_id(1) * BLOCK_FRAMES + tl.arange(0, BLOCK_FRAMES)
    channels = tl.arange(0, BLOCK_CHANNELS)
    is_channel = channels < CHANNELS
    is_frame = frame_positions < frame_count
    batch_pointer = hidden_states_pointer + batch_index * frame_count * CHANNELS
    convolved = (
        tl.zeros((BLOCK_FRAMES, BLOCK_CHANNELS), dtype=tl.float32)
        + tl.load(convolution_bias_pointer + channels, mask=is_channel, other=0.0).to(
            tl.float32
        )[None, :]
    )
    for tap in tl.static_range(KERNEL_SIZE):
        source_frames = frame_positions + (tap - (KERNEL_SIZE - 1))
        samples = tl.load(
            batch_pointer + source_frames[:, None] * CHANNELS + channels[None, :],
            mask=((source_frames >= 0) & is_frame)[:, None] & is_channel[None, :],
            other=0.0,
        ).to(tl.float32)
        tap_weight = tl.load(
            convolution_weight_pointer + channels * KERNEL_SIZE + tap,
            mask=is_channel,
            other=0.0,
        ).to(tl.float32)
        convolved = convolved + tap_weight[None, :] * samples
    output_dtype = normalized_pointer.dtype.element_ty
    convolved = tl.where(
        is_channel[None, :], convolved.to(output_dtype).to(tl.float32), 0.0
    )
    mean = tl.sum(convolved, axis=1) / CHANNELS
    centered = tl.where(is_channel[None, :], convolved - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, axis=1) / CHANNELS
    norm_weight = tl.load(
        norm_weight_pointer + channels, mask=is_channel, other=0.0
    ).to(tl.float32)
    norm_bias = tl.load(norm_bias_pointer + channels, mask=is_channel, other=0.0).to(
        tl.float32
    )
    normalized = (
        norm_weight[None, :] * (libdevice.rsqrt(variance + epsilon)[:, None] * centered)
        + norm_bias[None, :]
    )
    tl.store(
        normalized_pointer
        + (batch_index * frame_count + frame_positions[:, None]) * CHANNELS
        + channels[None, :],
        normalized.to(output_dtype),
        mask=is_frame[:, None] & is_channel[None, :],
    )


def convnext_stack(
    blocks: nn.Sequential | nn.ModuleList, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Run ConvNeXt blocks over (B, C, T) states and return the (B, T, C) result.

    On the CUDA platform the states stay frame-major and one kernel convolves and normalizes each block's input.
    """
    if current_platform.is_cuda() and hidden_states.is_cuda:
        frame_major_states = hidden_states.transpose(1, 2).contiguous()
        batch_size, frame_count, channel_count = frame_major_states.shape
        for block in blocks:
            assert isinstance(block, ConvNeXtBlock)
            normalized = torch.empty_like(frame_major_states)
            depthwise_norm_kernel[
                (batch_size, triton.cdiv(frame_count, CONVNEXT_BLOCK_FRAMES))
            ](
                frame_major_states,
                block.dwconv.weight,
                block.dwconv.bias,
                block.norm.weight,
                block.norm.bias,
                normalized,
                frame_count,
                block.norm.eps,
                CHANNELS=channel_count,
                KERNEL_SIZE=block.dwconv.kernel_size[0],
                BLOCK_CHANNELS=triton.next_power_of_2(channel_count),
                BLOCK_FRAMES=CONVNEXT_BLOCK_FRAMES,
                num_warps=CONVNEXT_WARPS,
            )
            frame_major_states = frame_major_states + block.pwconv2(
                F.gelu(block.pwconv1(normalized))
            )
        return frame_major_states
    else:
        for block in blocks:
            hidden_states = block(hidden_states)
        return hidden_states.transpose(1, 2)


# note (zhaochenyang20): the strided convolution ran as a 0.44 ms TF32 cuDNN implicit GEMM per 5 s decode on one H100; the float32 window matmul takes 0.02 ms.
def polyphase_resample(
    waveform: torch.Tensor,
    kernel: torch.Tensor,
    input_ratio: int,
    output_ratio: int,
) -> torch.Tensor:
    """sinc_resample of (B, samples) by output_ratio / input_ratio as one matmul of strided input windows by the (output_ratio, 1, taps) kernel."""
    batch_size, sample_count = waveform.shape
    tap_count = kernel.shape[-1]
    kernel_half_width = (tap_count - input_ratio) // 2
    windows = F.pad(
        waveform, (kernel_half_width, kernel_half_width + input_ratio)
    ).unfold(-1, tap_count, input_ratio)
    resampled = torch.matmul(windows, kernel[:, 0].t()).reshape(batch_size, -1)
    return resampled[:, : math.ceil(output_ratio * sample_count / input_ratio)]


def fused_transformer_forward(
    stage: ProjectedTransformer, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Map (B, C_in, T) to (B, C_out, T) as ProjectedTransformer.forward does, with fused residual norms and FA3 attention."""
    hidden_states = stage.input_proj(hidden_states.transpose(1, 2)).contiguous()
    batch_size, frame_count, model_dimension = hidden_states.shape
    head_count = stage.spec.head_count
    head_dimension = model_dimension // head_count
    rotary_cos, rotary_signed_sin = rotary_tables(
        torch.arange(frame_count, device=hidden_states.device, dtype=torch.float32),
        head_dimension,
    )
    # Key frames (query - context, query] are visible: FA3's left window excludes the query's own frame.
    window_size = (stage.spec.context_frames - 1, 0)
    normalized_states = stage.layers[0].norm1(hidden_states)
    for layer_index, layer in enumerate(stage.layers):
        query_key_value = layer.qkv(normalized_states)
        query_key_value_states = query_key_value.new_empty(
            (3, batch_size, head_count, frame_count, head_dimension)
        )
        write_rotary_states(
            query_key_value, rotary_cos, rotary_signed_sin, query_key_value_states, 0
        )
        query, key, value = query_key_value_states.transpose(2, 3)
        attention_output = flash_attn_with_kvcache(
            query, key, value, causal=True, window_size=window_size
        )
        hidden_states, normalized_states = scaled_residual_norm(
            hidden_states,
            layer.attention_out(
                attention_output.reshape(batch_size, frame_count, model_dimension)
            ),
            layer.attention_scale,
            layer.norm2,
        )
        feedforward = layer.ffn_output(F.gelu(layer.ffn_input(normalized_states)))
        if layer_index + 1 < len(stage.layers):
            hidden_states, normalized_states = scaled_residual_norm(
                hidden_states,
                feedforward,
                layer.ffn_scale,
                stage.layers[layer_index + 1].norm1,
            )
        else:
            hidden_states = scaled_residual(hidden_states, feedforward, layer.ffn_scale)
    if stage.output_proj is None:
        pass
    else:
        hidden_states = stage.output_proj(hidden_states)
    return hidden_states.transpose(1, 2)
