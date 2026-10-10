# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-ASR (Copyright 2025 Prince Canuma and contributors).
"""Qwen3-ASR in plain MLX: audio encoder, and the model built from a checkpoint."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from sglang_omni_mlx.checkpoint import load_weights, read_weights
from sglang_omni_mlx.qwen3_asr.audio import AudioLayout, swift_token_count
from sglang_omni_mlx.text_decoder import (
    KVCache,
    TextDecoder,
    TextDecoderConfig,
    text_decoder_config,
)


@dataclass(frozen=True, kw_only=True)
class AudioEncoderConfig:
    num_mel_bins: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    d_model: int
    max_source_positions: int
    n_window: int
    n_window_infer: int
    downsample_hidden_size: int
    output_dim: int


def sinusoidal_positions(length: int, channels: int) -> mx.array:
    timescale_step = math.log(10000.0) / (channels // 2 - 1)
    inverse_timescales = mx.exp(
        -timescale_step * mx.arange(channels // 2, dtype=mx.float32)
    )
    scaled_time = (
        mx.arange(length, dtype=mx.float32)[:, None] * inverse_timescales[None, :]
    )
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class AudioAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.head_count = config.encoder_attention_heads
        self.head_dim = config.d_model // self.head_count
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Full self-attention within each row of the batch (one window each)."""
        batch, length, width = hidden_states.shape
        queries, keys, values = (
            projection(hidden_states)
            .reshape(batch, length, self.head_count, self.head_dim)
            .transpose(0, 2, 1, 3)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.head_dim**-0.5
        )
        return self.out_proj(
            attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        )


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states)
        )
        return hidden_states + self.fc2(
            nn.gelu(self.fc1(self.final_layer_norm(hidden_states)))
        )


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, hidden, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        frequency_bins_after_conv = (
            (((config.num_mel_bins + 1) // 2) + 1) // 2 + 1
        ) // 2
        self.conv_out = nn.Linear(
            hidden * frequency_bins_after_conv, config.d_model, bias=False
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def __call__(self, mel: mx.array, layout: AudioLayout) -> mx.array:
        """[mel_bins, frames] to [audio_tokens, output_dim]."""
        chunk_frame_count = self.config.n_window * 2
        frame_count = mel.shape[-1]
        chunk_lengths = [
            min(chunk_frame_count, frame_count - start)
            for start in range(0, frame_count, chunk_frame_count)
        ]
        longest_chunk = max(chunk_lengths)
        chunks = mx.stack(
            [
                mx.pad(
                    mel[:, start : start + length],
                    [(0, 0), (0, longest_chunk - length)],
                )
                for start, length in zip(
                    range(0, frame_count, chunk_frame_count), chunk_lengths
                )
            ]
        )
        x = chunks[:, :, :, None]
        x = nn.gelu(self.conv2d3(nn.gelu(self.conv2d2(nn.gelu(self.conv2d1(x))))))
        chunk_count, frequency_bins, conv_frames, channels = x.shape
        x = self.conv_out(
            x.transpose(0, 2, 3, 1).reshape(
                chunk_count, conv_frames, channels * frequency_bins
            )
        )
        x = x + sinusoidal_positions(conv_frames, self.config.d_model)[None]

        reference_lengths = [conv_output_frames(length) for length in chunk_lengths]
        if layout is AudioLayout.REFERENCE:
            credited_lengths = reference_lengths
        else:
            # The Swift port credits each chunk by its own length formula and keeps
            # that many rows of the padded conv output.
            credited_lengths = [swift_token_count(length) for length in chunk_lengths]
        kept_lengths = [min(length, conv_frames) for length in credited_lengths]
        hidden_states = mx.concatenate(
            [x[i, :length] for i, length in enumerate(kept_lengths)], axis=0
        )

        # Attention stays within windows of chunks. Like the Swift encoder, each
        # window runs on its own (windows of one length batched together) rather
        # than under one block-diagonal mask, which rounds differently.
        chunks_per_window = max(1, self.config.n_window_infer // chunk_frame_count)
        window_lengths = [
            sum(credited_lengths[start : start + chunks_per_window])
            for start in range(0, chunk_count, chunks_per_window)
        ]
        token_count = hidden_states.shape[0]
        window_bounds: list[tuple[int, int]] = []
        window_start = 0
        for window_length in window_lengths:
            window_end = min(window_start + window_length, token_count)
            if window_end > window_start:
                window_bounds.append((window_start, window_end))
            else:
                pass
            window_start = window_end
        if window_start < token_count:
            window_bounds.append((window_start, token_count))
        else:
            pass
        encoded_windows: dict[int, mx.array] = {}
        for length in sorted({end - start for start, end in window_bounds}):
            same_length = [
                index
                for index, (start, end) in enumerate(window_bounds)
                if end - start == length
            ]
            batch = mx.stack(
                [
                    hidden_states[window_bounds[index][0] : window_bounds[index][1]]
                    for index in same_length
                ]
            )
            for layer in self.layers:
                batch = layer(batch)
            for row, index in enumerate(same_length):
                encoded_windows[index] = batch[row]
        hidden_states = mx.concatenate(
            [encoded_windows[index] for index in range(len(window_bounds))], axis=0
        )
        hidden_states = self.ln_post(hidden_states)
        return self.proj2(nn.gelu(self.proj1(hidden_states)))


def conv_output_frames(frame_count: int) -> int:
    """Frames left after the three stride-2 convolutions."""
    for _ in range(3):
        frame_count = (frame_count - 1) // 2 + 1
    return frame_count


class Qwen3ASR(nn.Module):
    def __init__(
        self, audio_config: AudioEncoderConfig, text_config: TextDecoderConfig
    ) -> None:
        super().__init__()
        self.audio_tower = AudioEncoder(audio_config)
        self.model = TextDecoder(text_config)

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]


def load_qwen3_asr(model_directory: Path) -> Qwen3ASR:
    """Build the model from an MLX checkpoint directory, quantizing as the checkpoint was."""
    config = json.loads((model_directory / "config.json").read_text())
    thinker = config["thinker_config"]
    audio = thinker["audio_config"]
    model = Qwen3ASR(
        AudioEncoderConfig(
            **{name: audio[name] for name in AudioEncoderConfig.__dataclass_fields__}
        ),
        text_decoder_config(thinker["text_config"]),
    )
    load_weights(model, read_weights(model_directory), config.get("quantization"))
    return model
