# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-ASR (Copyright 2025 Prince Canuma and contributors).
"""Qwen3-ASR in plain MLX: audio encoder, Qwen3 text decoder and checkpoint loading."""

from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from sglang_omni_mlx.qwen3_asr.audio import AudioLayout, swift_token_count

KV_CACHE_STEP_TOKENS = 256


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


@dataclass(frozen=True, kw_only=True)
class TextDecoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float


class KVCache:
    """Per-layer key/value cache that grows in fixed steps."""

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        new_token_count = keys.shape[2]
        if self.keys is None or self.offset + new_token_count > self.keys.shape[2]:
            batch, head_count, _, head_dim = keys.shape
            step_count = (
                new_token_count + KV_CACHE_STEP_TOKENS - 1
            ) // KV_CACHE_STEP_TOKENS
            grown = (batch, head_count, step_count * KV_CACHE_STEP_TOKENS, head_dim)
            extra_keys = mx.zeros(grown, keys.dtype)
            extra_values = mx.zeros(grown, values.dtype)
            if self.keys is None:
                self.keys, self.values = extra_keys, extra_values
            else:
                self.keys = mx.concatenate(
                    [self.keys[..., : self.offset, :], extra_keys], axis=2
                )
                self.values = mx.concatenate(
                    [self.values[..., : self.offset, :], extra_values], axis=2
                )
        else:
            pass
        self.keys[..., self.offset : self.offset + new_token_count, :] = keys
        self.values[..., self.offset : self.offset + new_token_count, :] = values
        self.offset += new_token_count
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


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


class TextAttention(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.head_count = config.num_attention_heads
        self.kv_head_count = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.head_count * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.kv_head_count * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.kv_head_count * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.head_count * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, hidden_states: mx.array, cache: KVCache) -> mx.array:
        batch, length, _ = hidden_states.shape
        queries = self.q_norm(
            self.q_proj(hidden_states).reshape(
                batch, length, self.head_count, self.head_dim
            )
        )
        keys = self.k_norm(
            self.k_proj(hidden_states).reshape(
                batch, length, self.kv_head_count, self.head_dim
            )
        )
        values = self.v_proj(hidden_states).reshape(
            batch, length, self.kv_head_count, self.head_dim
        )
        queries = self.rope(queries.transpose(0, 2, 1, 3), offset=cache.offset)
        keys = self.rope(keys.transpose(0, 2, 1, 3), offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values.transpose(0, 2, 1, 3))
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.head_dim**-0.5,
            mask="causal" if length > 1 else None,
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class TextDecoderLayer(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.self_attn = TextAttention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = TextMLP(config)

    def __call__(self, hidden_states: mx.array, cache: KVCache) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), cache
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TextMLP(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class TextDecoder(nn.Module):
    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, embeddings: mx.array, caches: list[KVCache]) -> mx.array:
        """Logits for the last position only."""
        hidden_states = embeddings
        for layer, cache in zip(self.layers, caches):
            hidden_states = layer(hidden_states, cache)
        return self.embed_tokens.as_linear(self.norm(hidden_states[:, -1:, :]))[0, -1]


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
    text = thinker["text_config"]
    model = Qwen3ASR(
        AudioEncoderConfig(
            **{name: audio[name] for name in AudioEncoderConfig.__dataclass_fields__}
        ),
        TextDecoderConfig(
            **{name: text[name] for name in TextDecoderConfig.__dataclass_fields__}
        ),
    )
    weights: dict[str, mx.array] = {}
    for path in sorted(glob.glob(str(model_directory / "*.safetensors"))):
        weights.update(mx.load(path))
    quantization = config.get("quantization")
    if quantization is not None:
        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=lambda path, module: f"{path}.scales" in weights,
        )
    else:
        pass
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
