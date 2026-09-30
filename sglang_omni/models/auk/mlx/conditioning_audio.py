# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
"""MLX audio feature encoder for AuK's Qwen2.5-Omni conditioner."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from transformers import Qwen2_5OmniAudioEncoderConfig


class AudioAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.d_model // self.num_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        batch, length, _ = hidden.shape
        query, key, value = (
            projection(hidden)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.head_dim**-0.5, mask=mask
        )
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None:
        super().__init__()
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        hidden = hidden + self.self_attn(self.self_attn_layer_norm(hidden), mask)
        return hidden + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(hidden))))


class AuKMlxAudioEncoder(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None:
        super().__init__()
        if config.activation_function != "gelu":
            raise ValueError("AuK's audio conditioner requires GELU activation")
        else:
            pass
        self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1)
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj = nn.Linear(config.d_model, config.output_dim)

    def __call__(
        self, input_features: mx.array, feature_lengths: list[int]
    ) -> mx.array:
        """Encode concatenated mel frames, pooling separately within each recording."""
        if not feature_lengths or min(feature_lengths) < 3:
            raise ValueError("Reference audio must contain at least three mel frames")
        elif (
            input_features.ndim != 2
            or input_features.shape[0] != self.config.num_mel_bins
        ):
            raise ValueError("Audio features must have shape (mel bins, frames)")
        elif sum(feature_lengths) != input_features.shape[1]:
            raise ValueError(
                "Audio frame lengths do not match the concatenated features"
            )
        else:
            pass
        window = self.config.n_window * 2
        chunks = []
        chunk_lengths = []
        offset = 0
        for length in feature_lengths:
            for start in range(0, length, window):
                chunk_length = min(window, length - start)
                chunks.append(
                    input_features[:, offset + start : offset + start + chunk_length].T
                )
                chunk_lengths.append(chunk_length)
            offset += length
        maximum = max(chunk_lengths)
        padded = mx.stack(
            [
                mx.pad(chunk, ((0, maximum - length), (0, 0)))
                for chunk, length in zip(chunks, chunk_lengths)
            ]
        ).astype(self.conv1.weight.dtype)
        lengths = mx.array(chunk_lengths)
        valid = mx.arange(maximum)[None, :] < lengths[:, None]
        hidden = nn.gelu(self.conv1(padded)) * valid[:, :, None]
        hidden = nn.gelu(self.conv2(hidden))
        half_dim = self.config.d_model // 2
        log_timescale_increment = math.log(10000) / (half_dim - 1)
        inverse_timescales = mx.exp(
            -log_timescale_increment * mx.arange(half_dim, dtype=mx.float32)
        )
        positions = (
            mx.arange(hidden.shape[1], dtype=mx.float32)[:, None]
            * inverse_timescales[None, :]
        )
        hidden = hidden + mx.concatenate(
            [mx.sin(positions), mx.cos(positions)], axis=-1
        ).astype(hidden.dtype)
        mask = mx.arange(hidden.shape[1])[None, :] < ((lengths + 1) // 2)[:, None]
        mask = mask[:, None, None, :]
        # note (Codex): Separate batch rows enforce attention isolation between windows.
        for layer in self.layers:
            hidden = layer(hidden, mask)
            mx.eval(hidden)
        hidden = mx.concatenate(
            [
                hidden[index, : (length + 1) // 2]
                for index, length in enumerate(chunk_lengths)
            ]
        )
        pooled = []
        offset = 0
        for length in feature_lengths:
            encoded_length = (length + 1) // 2
            paired_length = encoded_length // 2 * 2
            pooled.append(
                hidden[offset : offset + paired_length]
                .reshape(-1, 2, self.config.d_model)
                .mean(axis=1)
            )
            offset += encoded_length
        return self.proj(self.ln_post(mx.concatenate(pooled)))
