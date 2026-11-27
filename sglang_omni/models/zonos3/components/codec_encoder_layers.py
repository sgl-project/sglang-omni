# SPDX-License-Identifier: Apache-2.0
"""Encoder-only layers of the Zonos3 codec: the acoustic convolution blocks and the transformer stage pass."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import sdpa_kernel

from sglang_omni.models.zonos3.components.codec_encoder_kernels import biased_snake
from sglang_omni.models.zonos3.components.codec_kernels import (
    scaled_residual,
    write_rotary_states,
)
from sglang_omni.models.zonos3.components.codec_layers import CausalConv1d
from sglang_omni.models.zonos3.components.codec_transformer import (
    CODEC_ATTENTION_BACKEND_LOCK,
    CODEC_ATTENTION_BACKENDS,
    ProjectedTransformer,
)
from sglang_omni.models.zonos3.components.encoder_norm_kernels import residual_norm

RESIDUAL_UNIT_KERNEL_SIZE = 7
RESIDUAL_UNIT_DILATIONS = (1, 3, 9)
# The memory-efficient attention kernel's causal mask aligned at the first query and key.
CAUSAL_FROM_TOP_LEFT_MASK = 1


def channels_last_convolution(
    convolution: CausalConv1d, padded_states: torch.Tensor
) -> torch.Tensor:
    """Bias-free convolution of (B, C, T) input that carries its left padding; when the encoder runs fused the input, the weight and the output are (B, T, C)-ordered memory, the layout cuDNN's kernels compute in, so no transpose runs around them."""
    return F.conv2d(
        padded_states.unsqueeze(2),
        convolution.weight.unsqueeze(2),
        None,
        stride=(1, convolution.stride[0]),
        dilation=(1, convolution.dilation[0]),
    ).squeeze(2)


class Snake(nn.Module):
    """Learned per-channel frequency of a snake activation; biased_snake applies it."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))


class ResidualUnit(nn.Module):
    """Snake, dilated causal conv, snake, pointwise conv, plus skip; EncoderBlock runs it."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.input_activation = Snake(channels)
        self.dilated_conv = CausalConv1d(
            channels, channels, RESIDUAL_UNIT_KERNEL_SIZE, dilation=dilation
        )
        self.output_activation = Snake(channels)
        self.pointwise_conv = CausalConv1d(channels, channels, 1)


class EncoderBlock(nn.Module):
    """Three residual units followed by a strided causal downsampling conv."""

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        self.residual_units = nn.ModuleList(
            ResidualUnit(input_channels, dilation)
            for dilation in RESIDUAL_UNIT_DILATIONS
        )
        self.activation = Snake(input_channels)
        self.downsample_conv = CausalConv1d(
            input_channels, output_channels, 2 * stride, stride=stride
        )

    def forward(
        self,
        convolved_states: torch.Tensor,
        bias: torch.Tensor,
        uses_fused_encoder: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the previous bias-free convolution's (B, C_in, T) output and its bias to this block's bias-free downsampled output and its bias.

        Every convolution runs without its bias; the snake that reads its output adds the bias and the residual skip.
        """
        first_unit = self.residual_units[0]
        skip_states, activated_states = biased_snake(
            convolved_states,
            bias,
            first_unit.input_activation.alpha,
            first_unit.dilated_conv.left_padding,
            None,
            True,
            uses_fused_encoder,
        )
        for unit_index, unit in enumerate(self.residual_units):
            _, activated_states = biased_snake(
                channels_last_convolution(unit.dilated_conv, activated_states),
                unit.dilated_conv.bias,
                unit.output_activation.alpha,
                unit.pointwise_conv.left_padding,
                None,
                False,
                uses_fused_encoder,
            )
            is_last_unit = unit_index + 1 == len(self.residual_units)
            if is_last_unit:
                next_activation = self.activation
                next_left_padding = self.downsample_conv.left_padding
            else:
                next_unit = self.residual_units[unit_index + 1]
                next_activation = next_unit.input_activation
                next_left_padding = next_unit.dilated_conv.left_padding
            skip_states, activated_states = biased_snake(
                channels_last_convolution(unit.pointwise_conv, activated_states),
                unit.pointwise_conv.bias,
                next_activation.alpha,
                next_left_padding,
                skip_states,
                not is_last_unit,
                uses_fused_encoder,
            )
        return (
            channels_last_convolution(self.downsample_conv, activated_states),
            self.downsample_conv.bias,
        )


def encoder_transformer_forward(
    stage: ProjectedTransformer,
    hidden_states: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_signed_sin: torch.Tensor,
    uses_fused_encoder: bool,
) -> torch.Tensor:
    """Map (B, C_in, T) to (B, C_out, T) as ProjectedTransformer.forward does under bf16 autocast, bit for bit.

    The rotary tables cover at least T frames. On the CUDA platform the residual adds feed fused norms; when the encoder runs fused the memory-efficient attention kernel also masks the causal local window itself: the same key tiles and scores as the 64-frame block mask, without the block mask, the zero-padded key windows and the output copy.
    """
    # note (zhaochenyang20): the fused norms, the windowed attention kernel and shared RoPE tables ran a 4.8 s reference's 80 encoder layers in 5.05 ms instead of 6.50 ms of graph replay on one H100, bit for bit.
    hidden_states = stage.input_proj(hidden_states.transpose(1, 2))
    batch_size, frame_count, model_dimension = hidden_states.shape
    head_count = stage.spec.head_count
    head_dimension = model_dimension // head_count
    context_frames = stage.spec.context_frames
    rotary_cos = rotary_cos[:frame_count]
    rotary_signed_sin = rotary_signed_sin[:frame_count]
    if uses_fused_encoder:
        window_mask = None
    else:
        frame_positions = torch.arange(frame_count, device=hidden_states.device)
        window_mask = (frame_positions[None, :] <= frame_positions[:, None]) & (
            frame_positions[None, :] > frame_positions[:, None] - context_frames
        )
    _, normalized_states = residual_norm(
        hidden_states, stage.layers[0].norm1, None, None
    )
    for layer_index, layer in enumerate(stage.layers):
        query_key_value = layer.qkv(normalized_states)
        query_key_value_states = query_key_value.new_empty(
            (3, batch_size, head_count, frame_count, head_dimension)
        )
        write_rotary_states(
            query_key_value, rotary_cos, rotary_signed_sin, query_key_value_states, 0
        )
        query, key, value = query_key_value_states
        if window_mask is None:
            attention_outputs = torch.ops.aten._efficient_attention_forward(  # noqa: leading-underscore  # upstream spelling, the SDPA entry point has no window
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                None,
                None,
                None,
                None,
                None,
                0.0,
                CAUSAL_FROM_TOP_LEFT_MASK,
                False,
                window_size=context_frames if frame_count > context_frames else None,
            )
            attention_output = attention_outputs[0]
        else:
            with CODEC_ATTENTION_BACKEND_LOCK, sdpa_kernel(CODEC_ATTENTION_BACKENDS):
                attention_output = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=window_mask
                ).transpose(1, 2)
        hidden_states, normalized_states = residual_norm(
            hidden_states,
            layer.norm2,
            layer.attention_out(
                attention_output.reshape(batch_size, frame_count, model_dimension)
            ),
            layer.attention_scale,
        )
        feedforward = layer.ffn_output(F.gelu(layer.ffn_input(normalized_states)))
        if layer_index + 1 < len(stage.layers):
            hidden_states, normalized_states = residual_norm(
                hidden_states,
                stage.layers[layer_index + 1].norm1,
                feedforward,
                layer.ffn_scale,
            )
        else:
            hidden_states = scaled_residual(hidden_states, feedforward, layer.ffn_scale)
    assert stage.output_proj is not None
    return stage.output_proj(hidden_states).transpose(1, 2)
