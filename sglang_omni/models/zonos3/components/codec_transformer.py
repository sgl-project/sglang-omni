# SPDX-License-Identifier: Apache-2.0
"""The causal local-window RoPE transformer stage shared by the Zonos3 codec encoder and decoder."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang_omni.models.zonos3.components.codec_kernels import (
    scaled_residual,
    write_rotary_states,
)

TRANSFORMER_LAYER_NORM_EPSILON = 1e-5
ROPE_MAX_PERIOD = 10000.0
# note (zhaochenyang20): Codec attention runs in the weight dtype on the memory-efficient kernel, which takes the local-window block mask; left to dispatch, a bfloat16 call picks cuDNN attention, which spends 30-500 ms of CPU planning every new frame count, and math serves CPU and shapes the kernel rejects. On one H100, bfloat16 attention decodes 5 s of audio in 13.3 ms instead of 16.2 ms and encodes a reference in 24 ms instead of 28 ms, with unchanged waveform SNR to a float32 decode.
CODEC_ATTENTION_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
# sdpa_kernel sets process-wide backend flags and restores them on exit, so the vocoder and reference-encoder threads enter it one at a time.
CODEC_ATTENTION_BACKEND_LOCK = threading.Lock()
# note (zhaochenyang20): The memory-efficient attention kernel tiles 64 queries by 64 keys; local-window attention runs 64-frame query blocks over whole key tiles.
LOCAL_ATTENTION_BLOCK_FRAMES = 64
# Checkpoint names of a layer's feedforward linears, renamed to the module's.
TRANSFORMER_INFIX_RENAMES: list[tuple[str, str]] = [
    (".ffn.0.", ".ffn_input."),
    (".ffn.3.", ".ffn_output."),
]


@dataclass(frozen=True, kw_only=True)
class TransformerStageSpec:
    """Shape of one causal local-window projected transformer stage."""

    input_dimension: int
    model_dimension: int
    output_dimension: int | None
    head_count: int
    layer_count: int
    feedforward_dimension: int
    context_frames: int


def rotary_tables(
    positions: torch.Tensor, head_dimension: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cosines and signed sines (..., head dimension) of float32 frame positions (...)."""
    frequencies = torch.exp(
        torch.arange(head_dimension // 2, device=positions.device, dtype=torch.float32)
        * (-math.log(ROPE_MAX_PERIOD) * 2 / head_dimension)
    )
    angles = positions[..., None] * frequencies
    # A channel pair shares its angle; the swapped partner of the real channel takes -sin, of the imaginary channel +sin.
    rotary_cos = angles.cos().repeat_interleave(2, dim=-1)
    rotary_sine = angles.sin()
    return rotary_cos, torch.stack((-rotary_sine, rotary_sine), dim=-1).flatten(-2)


class ProjectedTransformerLayer(nn.Module):
    """Pre-norm transformer layer with layer-scaled attention and GELU FFN."""

    def __init__(
        self, model_dimension: int, head_count: int, feedforward_dimension: int
    ) -> None:
        super().__init__()
        self.head_count: int = head_count
        self.norm1 = nn.LayerNorm(model_dimension, eps=TRANSFORMER_LAYER_NORM_EPSILON)
        self.qkv = nn.Linear(model_dimension, 3 * model_dimension, bias=False)
        self.attention_out = nn.Linear(model_dimension, model_dimension, bias=False)
        self.attention_scale = nn.Parameter(torch.empty(model_dimension))
        self.norm2 = nn.LayerNorm(model_dimension, eps=TRANSFORMER_LAYER_NORM_EPSILON)
        self.ffn_input = nn.Linear(model_dimension, feedforward_dimension, bias=False)
        self.ffn_output = nn.Linear(feedforward_dimension, model_dimension, bias=False)
        self.ffn_scale = nn.Parameter(torch.empty(model_dimension))

    def rotated_states(
        self,
        hidden_states: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_signed_sin: torch.Tensor,
        leading_frames: int,
        padded_frames: int,
    ) -> torch.Tensor:
        """Query, key and value of (B, T, C) states, query and key rotated, as (3, B, heads, frames, head dimension).

        The T projected frames start at leading_frames of leading_frames + padded_frames state frames; the others are zero.
        """
        batch_size, frame_count, model_dimension = hidden_states.shape
        query_key_value = self.qkv(self.norm1(hidden_states))
        state_shape = (
            3,
            batch_size,
            self.head_count,
            leading_frames + padded_frames,
            model_dimension // self.head_count,
        )
        if leading_frames + padded_frames == frame_count:
            query_key_value_states = query_key_value.new_empty(state_shape)
        else:
            # Zero padding keeps the masked scores finite.
            query_key_value_states = query_key_value.new_zeros(state_shape)
        write_rotary_states(
            query_key_value,
            rotary_cos,
            rotary_signed_sin,
            query_key_value_states,
            leading_frames,
        )
        return query_key_value_states

    def add_attention(
        self, hidden_states: torch.Tensor, attention_output: torch.Tensor
    ) -> torch.Tensor:
        """Add the layer-scaled attention output (B, T, C) and then the feedforward branch to the (B, T, C) states."""
        hidden_states = scaled_residual(
            hidden_states, self.attention_out(attention_output), self.attention_scale
        )
        feedforward = self.ffn_output(F.gelu(self.ffn_input(self.norm2(hidden_states))))
        return scaled_residual(hidden_states, feedforward, self.ffn_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_signed_sin: torch.Tensor,
        window_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """window_mask is None when every earlier frame lies in the attention window, else the additive block mask."""
        batch_size, frame_count, model_dimension = hidden_states.shape
        head_dimension = model_dimension // self.head_count
        if window_mask is None:
            leading_frames = 0
            padded_frames = frame_count
        else:
            block_count, block_frames, window_frames = window_mask.shape
            leading_frames = window_frames - block_frames
            padded_frames = block_count * block_frames
        query, key, value = self.rotated_states(
            hidden_states, rotary_cos, rotary_signed_sin, leading_frames, padded_frames
        )
        if window_mask is None:
            with CODEC_ATTENTION_BACKEND_LOCK, sdpa_kernel(CODEC_ATTENTION_BACKENDS):
                attention_output = F.scaled_dot_product_attention(
                    query, key, value, is_causal=True
                )
            attention_output = attention_output.transpose(1, 2).reshape(
                batch_size, frame_count, model_dimension
            )
        else:
            # Query block j attends to the window_frames key frames that end with it; the windows are overlapping views.
            with CODEC_ATTENTION_BACKEND_LOCK, sdpa_kernel(CODEC_ATTENTION_BACKENDS):
                block_attention = F.scaled_dot_product_attention(
                    query[:, :, leading_frames:].reshape(
                        batch_size * self.head_count,
                        block_count,
                        block_frames,
                        head_dimension,
                    ),
                    key.unfold(2, window_frames, block_frames)
                    .transpose(-1, -2)
                    .flatten(0, 1),
                    value.unfold(2, window_frames, block_frames)
                    .transpose(-1, -2)
                    .flatten(0, 1),
                    attn_mask=window_mask,
                )
            attention_output = hidden_states.new_empty(
                (batch_size, padded_frames, model_dimension)
            )
            attention_output.view(
                batch_size, block_count, block_frames, self.head_count, head_dimension
            ).copy_(
                block_attention.unflatten(0, (batch_size, self.head_count)).permute(
                    0, 2, 3, 1, 4
                )
            )
            attention_output = attention_output[:, :frame_count]
        return self.add_attention(hidden_states, attention_output)


class ProjectedTransformer(nn.Module):
    """Input projection, causal local-window RoPE transformer, optional output projection."""

    def __init__(self, spec: TransformerStageSpec) -> None:
        super().__init__()
        self.spec: TransformerStageSpec = spec
        self.input_proj = nn.Linear(
            spec.input_dimension, spec.model_dimension, bias=False
        )
        self.layers = nn.ModuleList(
            ProjectedTransformerLayer(
                spec.model_dimension, spec.head_count, spec.feedforward_dimension
            )
            for _ in range(spec.layer_count)
        )
        if spec.output_dimension is None:
            self.output_proj: nn.Linear | None = None
        else:
            self.output_proj = nn.Linear(
                spec.model_dimension, spec.output_dimension, bias=False
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map (B, C_in, T) to (B, C_out, T)."""
        hidden_states = self.input_proj(hidden_states.transpose(1, 2))
        frame_count = hidden_states.shape[1]
        device = hidden_states.device
        rotary_cos, rotary_signed_sin = rotary_tables(
            torch.arange(frame_count, device=device, dtype=torch.float32),
            self.spec.model_dimension // self.spec.head_count,
        )
        context_frames = self.spec.context_frames
        if frame_count <= context_frames:
            window_mask = None
        else:
            block_count = math.ceil(frame_count / LOCAL_ATTENTION_BLOCK_FRAMES)
            window_frames = (
                math.ceil(context_frames / LOCAL_ATTENTION_BLOCK_FRAMES) + 1
            ) * LOCAL_ATTENTION_BLOCK_FRAMES
            query_frames = torch.arange(
                block_count * LOCAL_ATTENTION_BLOCK_FRAMES, device=device
            ).view(block_count, LOCAL_ATTENTION_BLOCK_FRAMES, 1)
            key_frames = (
                query_frames[:, :1]
                - (window_frames - LOCAL_ATTENTION_BLOCK_FRAMES)
                + torch.arange(window_frames, device=device)
            )
            is_visible = (
                (key_frames >= 0)
                & (query_frames >= key_frames)
                & (query_frames - context_frames < key_frames)
            )
            window_mask = torch.zeros(
                is_visible.shape, dtype=hidden_states.dtype, device=device
            ).masked_fill_(~is_visible, float("-inf"))
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, rotary_cos, rotary_signed_sin, window_mask
            )
        if self.output_proj is None:
            pass
        else:
            hidden_states = self.output_proj(hidden_states)
        return hidden_states.transpose(1, 2)
