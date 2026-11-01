# SPDX-License-Identifier: Apache-2.0
"""Whisper attention, encoder layers, and explicit streaming histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

HISTORY_ROW_DTYPES = (torch.float16, torch.bfloat16)


def supports_history_rows(device: torch.device, dtype: torch.dtype) -> bool:
    return device.type == "cuda" and dtype in HISTORY_ROW_DTYPES and _is_fa3_supported()


@dataclass(frozen=True, kw_only=True)
class AudioEncoderState:
    """Attention history of one session: its own tensor shaped (layers, 2, 1, heads, frames, head_dim), or the first row_length frames of a preallocated history row."""

    key_value_states: torch.Tensor | None = None
    row: int | None = None
    row_length: int = 0
    row_nbytes: int = 0

    @property
    def past_length(self) -> int:
        if self.row is not None:
            return self.row_length
        elif self.key_value_states is None:
            return 0
        else:
            return self.key_value_states.shape[4]

    @property
    def nbytes(self) -> int:
        if self.row is not None:
            return self.row_nbytes
        elif self.key_value_states is None:
            return 0
        else:
            return self.key_value_states.numel() * self.key_value_states.element_size()


class MiniCPMWhisperEncoderAttention(nn.Module):
    """Whisper self-attention with fused QKV and an additive SDPA mask."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        # note (MayDomine): Whisper's K projection is bias-free.
        with torch.no_grad():
            self.qkv_proj.bias[self.embed_dim : 2 * self.embed_dim].zero_()
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def reshape_heads(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        return states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over this chunk; when streaming, first write its keys and values into the tail of key_value_states."""
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        key, value = self.reshape_heads(key), self.reshape_heads(value)
        if key_value_states is not None:
            current_length = key.shape[2]
            key_value_states[0, :, :, -current_length:] = key
            key_value_states[1, :, :, -current_length:] = value
            key, value = key_value_states[0], key_value_states[1]
        else:
            pass
        attn_output = F.scaled_dot_product_attention(
            self.reshape_heads(query),
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.embed_dim,
        )
        return self.out_proj(attn_output)

    def forward_rows(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_lengths: torch.Tensor,
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Append this chunk's keys and values to each session's history row in place and attend over the row up to its length."""
        batch_size, current_length, _ = hidden_states.shape
        query, key, value = (
            projection.view(batch_size, current_length, self.num_heads, self.head_dim)
            for projection in self.qkv_proj(hidden_states).chunk(3, dim=-1)
        )
        attn_output = flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            k=key,
            v=value,
            cache_seqlens=cache_lengths,
            cache_batch_idx=cache_rows,
            causal=False,
        )
        return self.out_proj(
            attn_output.reshape(batch_size, current_length, self.embed_dim)
        )


class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.self_attn = MiniCPMWhisperEncoderAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attn_mask, key_value_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return residual + hidden_states

    def forward_rows(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_lengths: torch.Tensor,
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn.forward_rows(
            self.self_attn_layer_norm(hidden_states),
            key_cache,
            value_cache,
            cache_lengths,
            cache_rows,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return residual + hidden_states


class MiniCPMWhisperEncoder(nn.Module):
    """Standard Whisper encoder stack driven by an external additive mask."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList(
            [MiniCPMWhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def embed(
        self,
        input_features: torch.Tensor,
        positions: torch.Tensor | None,
        prefix_extra_frames: int,
        suffix_extra_frames: int,
    ) -> torch.Tensor:
        """Convolve mel frames, drop the context rows and add each frame's position embedding."""
        hidden_states = input_features.to(
            device=self.conv1.weight.device, dtype=self.conv1.weight.dtype
        )
        hidden_states = F.gelu(self.conv1(hidden_states))
        hidden_states = F.gelu(self.conv2(hidden_states))
        # note (Junnan Li): Context mel frames affect convolution, but never enter KV.
        prefix_rows = (prefix_extra_frames + 1) // 2
        suffix_rows = (suffix_extra_frames + 1) // 2
        hidden_states = hidden_states[:, :, prefix_rows:]
        if suffix_rows:
            hidden_states = hidden_states[:, :, :-suffix_rows]
        else:
            pass
        hidden_states = hidden_states.permute(0, 2, 1)

        if positions is None:
            embed_pos = self.embed_positions.weight[: hidden_states.shape[1]]
        else:
            embed_pos = self.embed_positions.weight[positions]
        return hidden_states + embed_pos

    def forward(
        self,
        input_features: torch.Tensor,
        attn_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        prefix_extra_frames: int = 0,
        suffix_extra_frames: int = 0,
    ) -> torch.Tensor:
        """Encode mel frames; when streaming, key_value_states holds the batch history and positions holds each row's frame positions."""
        hidden_states = self.embed(
            input_features, positions, prefix_extra_frames, suffix_extra_frames
        )
        for layer_index, layer in enumerate(self.layers):
            if key_value_states is None:
                layer_states = None
            else:
                layer_states = key_value_states[layer_index]
            hidden_states = layer(hidden_states, attn_mask, layer_states)
        return self.layer_norm(hidden_states)

    def forward_rows(
        self,
        input_features: torch.Tensor,
        positions: torch.Tensor,
        history_rows: torch.Tensor,
        cache_lengths: torch.Tensor,
        cache_rows: torch.Tensor,
        prefix_extra_frames: int,
        suffix_extra_frames: int,
    ) -> torch.Tensor:
        """Encode one chunk per session, appending its keys and values to the sessions' history rows."""
        hidden_states = self.embed(
            input_features, positions, prefix_extra_frames, suffix_extra_frames
        )
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer.forward_rows(
                hidden_states,
                history_rows[layer_index, 0],
                history_rows[layer_index, 1],
                cache_lengths,
                cache_rows,
            )
        return self.layer_norm(hidden_states)
