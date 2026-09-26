# SPDX-License-Identifier: Apache-2.0
"""Whisper attention, encoder layers, and explicit streaming histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN


@dataclass(frozen=True, kw_only=True)
class AudioAttentionState:
    """Read-only attention history; empty fields start cached attention."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class AudioEncoderState:
    """Read-only per-layer audio history retained by one perception session."""

    layers: tuple[AudioAttentionState, ...] = ()

    @property
    def past_length(self) -> int:
        if self.layers and self.layers[0].key is not None:
            return self.layers[0].key.shape[2]
        else:
            return 0

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in self.layers
            for tensor in (layer.key, layer.value)
            if tensor is not None
        )


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
        state: AudioAttentionState | None = None,
    ) -> tuple[torch.Tensor, AudioAttentionState | None]:
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        key, value = self.reshape_heads(key), self.reshape_heads(value)
        if state is not None and state.key is not None:
            assert state.value is not None
            key = torch.cat((state.key, key), dim=2)
            value = torch.cat((state.value, value), dim=2)
        else:
            pass
        new_state = (
            AudioAttentionState(key=key, value=value) if state is not None else None
        )
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
        return self.out_proj(attn_output), new_state


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
        state: AudioAttentionState | None = None,
    ) -> tuple[torch.Tensor, AudioAttentionState | None]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, new_state = self.self_attn(hidden_states, attn_mask, state)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return residual + hidden_states, new_state


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

    def forward(
        self,
        input_features: torch.Tensor,
        attn_mask: torch.Tensor,
        state: AudioEncoderState | None = None,
        prefix_extra_frames: int = 0,
        suffix_extra_frames: int = 0,
    ) -> tuple[torch.Tensor, AudioEncoderState | None]:
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

        position = state.past_length if state is not None else 0
        embed_pos = self.embed_positions.weight[
            position : position + hidden_states.shape[1]
        ]
        hidden_states = hidden_states + embed_pos.to(hidden_states.device)

        new_layers: list[AudioAttentionState] = []
        for layer_index, layer in enumerate(self.layers):
            if state is None:
                layer_state = None
            elif layer_index < len(state.layers):
                layer_state = state.layers[layer_index]
            else:
                layer_state = AudioAttentionState()
            hidden_states, new_layer_state = layer(
                hidden_states, attn_mask, layer_state
            )
            if new_layer_state is not None:
                new_layers.append(new_layer_state)
            else:
                pass
        new_state = (
            AudioEncoderState(layers=tuple(new_layers)) if state is not None else None
        )
        return self.layer_norm(hidden_states), new_state
