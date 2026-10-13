# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni audio encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig as AudioConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)

_CONVOLUTION_WEIGHTS = {
    "conv2d1.weight",
    "conv2d2.weight",
    "conv2d3.weight",
}


def qwen3_omni_audio_output_lengths(input_lengths: mx.array) -> mx.array:
    """Match the pinned Hugging Face three-layer audio CNN length arithmetic."""

    return _get_feat_extract_output_lengths(input_lengths.astype(mx.int32))


def chunk_lengths(
    feature_lengths: Sequence[int],
    *,
    chunk_size: int,
) -> tuple[int, ...]:
    lengths: list[int] = []
    for feature_length in feature_lengths:
        if feature_length <= 0:
            raise ValueError("audio feature lengths must be positive")
        else:
            pass
        full_chunks, remainder = divmod(feature_length, chunk_size)
        lengths.extend([chunk_size] * full_chunks)
        if remainder:
            lengths.append(remainder)
        else:
            pass
    return tuple(lengths)


def qwen3_omni_audio_attention_segments(
    feature_lengths: mx.array,
    *,
    n_window: int,
    n_window_infer: int,
) -> tuple[int, ...]:
    """Return HF-equivalent per-sample inference-window token segments."""

    if n_window <= 0 or n_window_infer <= 0:
        raise ValueError("audio window sizes must be positive")
    else:
        pass
    chunk_size = n_window * 2
    window_ratio = n_window_infer // chunk_size
    if window_ratio <= 0:
        raise ValueError("n_window_infer must cover at least one audio chunk")
    else:
        pass

    lengths = tuple(int(value) for value in np.asarray(feature_lengths).tolist())
    chunks = chunk_lengths(lengths, chunk_size=chunk_size)
    chunk_output_lengths = np.asarray(
        qwen3_omni_audio_output_lengths(mx.array(chunks, dtype=mx.int32))
    )
    max_chunk_output = int(chunk_output_lengths.max())
    tokens_per_window = max_chunk_output * window_ratio

    output_lengths = np.asarray(
        qwen3_omni_audio_output_lengths(mx.array(lengths, dtype=mx.int32))
    )
    segments: list[int] = []
    for output_length in output_lengths.tolist():
        full_windows, remainder = divmod(int(output_length), tokens_per_window)
        segments.extend([tokens_per_window] * full_windows)
        if remainder:
            segments.append(remainder)
        else:
            pass
    return tuple(segments)


def sanitize_audio_weights(
    weights: Mapping[str, mx.array],
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, mx.array]:
    """Validate component-local tensors and adapt Torch convolution layouts."""

    sanitized: dict[str, mx.array] = {}
    for source_key, value in weights.items():
        key = source_key
        if key in _CONVOLUTION_WEIGHTS:
            expected_shape = expected_shapes.get(key)
            source_shape = tuple(value.shape)
            if expected_shape is None:
                raise ValueError(
                    f"Qwen3-Omni audio model has no target shape for {key!r}"
                )
            else:
                pass
            if source_shape != expected_shape:
                candidate = value.transpose(0, 2, 3, 1) if value.ndim == 4 else None
                if candidate is None or tuple(candidate.shape) != expected_shape:
                    raise ValueError(
                        f"Qwen3-Omni audio tensor {source_key!r} has shape "
                        f"{source_shape}; expected MLX shape {expected_shape} "
                        "or a Torch Conv2D layout that transposes to it"
                    )
                else:
                    pass
                value = candidate
            else:
                pass
        else:
            pass

        if key in sanitized:
            raise ValueError(
                f"Qwen3-Omni audio weights {source_key!r} and another source "
                f"both map to {key!r}"
            )
        else:
            pass
        sanitized[key] = value
    return sanitized


class AudioAttention(nn.Module):
    """Self-attention over independent packed audio inference windows."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        if self.embed_dim % self.num_heads:
            raise ValueError(
                "audio d_model must be divisible by encoder_attention_heads"
            )
        else:
            pass
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        segment_lengths: tuple[int, ...],
    ) -> mx.array:
        sequence_length = hidden_states.shape[0]
        if sum(segment_lengths) != sequence_length:
            raise ValueError(
                "audio attention segments do not cover the packed sequence"
            )
        else:
            pass

        queries = self.q_proj(hidden_states).reshape(
            sequence_length, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            sequence_length, self.num_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            sequence_length, self.num_heads, self.head_dim
        )
        queries = queries.transpose(1, 0, 2)
        keys = keys.transpose(1, 0, 2)
        values = values.transpose(1, 0, 2)

        outputs: list[mx.array] = []
        start = 0
        for length in segment_lengths:
            end = start + length
            output = mx.fast.scaled_dot_product_attention(
                queries[:, start:end, :][None, ...],
                keys[:, start:end, :][None, ...],
                values[:, start:end, :][None, ...],
                scale=self.scaling,
            )
            outputs.append(output[0].transpose(1, 0, 2))
            start = end

        attended = mx.concatenate(outputs, axis=0).reshape(sequence_length, -1)
        return self.out_proj(attended)


class AudioEncoderLayer(nn.Module):
    """One pre-normalized Qwen3-Omni audio transformer block."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.activation_function = config.activation_function
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def activate(self, hidden_states: mx.array) -> mx.array:
        if self.activation_function == "gelu":
            return nn.gelu(hidden_states)
        else:
            pass
        if self.activation_function in ("gelu_new", "gelu_pytorch_tanh"):
            return nn.gelu_approx(hidden_states)
        else:
            pass
        raise ValueError(
            "unsupported Qwen3-Omni audio activation " f"{self.activation_function!r}"
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        segment_lengths: tuple[int, ...],
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            segment_lengths=segment_lengths,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activate(self.fc1(hidden_states)))
        return residual + hidden_states


class Qwen3OmniMlxAudioEncoder(nn.Module):
    """Qwen3-Omni audio tower with chunk-local positions and block attention."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        self.conv2d1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        frequency_after_conv = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * frequency_after_conv,
            config.d_model,
            bias=False,
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def chunk_and_pad(
        self,
        packed_features: mx.array,
        feature_lengths: mx.array,
    ) -> tuple[mx.array, tuple[int, ...]]:
        lengths = tuple(int(value) for value in np.asarray(feature_lengths).tolist())
        chunk_size = self.n_window * 2
        chunk_frame_lengths = chunk_lengths(lengths, chunk_size=chunk_size)
        chunks: list[mx.array] = []
        sample_offset = 0
        for feature_length in lengths:
            sample_end = sample_offset + feature_length
            for chunk_start in range(sample_offset, sample_end, chunk_size):
                chunks.append(
                    packed_features[
                        :,
                        chunk_start : min(chunk_start + chunk_size, sample_end),
                    ]
                )
            sample_offset = sample_end

        max_chunk_length = max(chunk_frame_lengths)
        padded = [
            mx.pad(chunk, ((0, 0), (0, max_chunk_length - length)))
            for chunk, length in zip(chunks, chunk_frame_lengths, strict=True)
        ]
        return mx.stack(padded, axis=0), chunk_frame_lengths

    def convolve(self, padded_features: mx.array) -> mx.array:
        chunks: list[mx.array] = []
        features = padded_features[:, :, :, None]
        for start in range(0, features.shape[0], self.conv_chunksize):
            feature_chunk = features[start : start + self.conv_chunksize]
            hidden = nn.gelu(self.conv2d1(feature_chunk))
            hidden = nn.gelu(self.conv2d2(hidden))
            hidden = nn.gelu(self.conv2d3(hidden))
            chunks.append(hidden)
        return mx.concatenate(chunks, axis=0)

    def __call__(
        self,
        input_features: mx.array,
        *,
        feature_lens: mx.array,
    ) -> mx.array:
        feature_lengths = feature_lens
        if input_features.shape != (
            self.config.num_mel_bins,
            int(feature_lengths.sum().item()),
        ):
            raise ValueError("Packed audio mel/frame dimensions do not match config")
        else:
            pass
        padded_features, chunk_lengths = self.chunk_and_pad(
            input_features,
            feature_lengths,
        )
        hidden_states = self.convolve(padded_features)
        batch, frequency, time, channels = hidden_states.shape
        hidden_states = hidden_states.transpose(0, 2, 3, 1).reshape(
            batch,
            time,
            channels * frequency,
        )
        hidden_states = self.conv_out(hidden_states)
        half_channels = self.config.d_model // 2
        log_timescale_increment = math.log(10000.0) / (half_channels - 1)
        frequencies = mx.exp(
            -log_timescale_increment * mx.arange(half_channels, dtype=mx.float32)
        )
        angles = mx.arange(hidden_states.shape[1])[:, None] * frequencies
        positions = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        hidden_states = hidden_states + positions[None, :, :]

        chunk_output_lengths = np.asarray(
            qwen3_omni_audio_output_lengths(mx.array(chunk_lengths, dtype=mx.int32))
        )
        hidden_states = mx.concatenate(
            [
                hidden_states[row, : int(valid_length)]
                for row, valid_length in enumerate(chunk_output_lengths.tolist())
            ],
            axis=0,
        )
        segment_lengths = qwen3_omni_audio_attention_segments(
            feature_lengths,
            n_window=self.n_window,
            n_window_infer=self.n_window_infer,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                segment_lengths=segment_lengths,
            )

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        if self.config.activation_function == "gelu":
            hidden_states = nn.gelu(hidden_states)
        elif self.config.activation_function in ("gelu_new", "gelu_pytorch_tanh"):
            hidden_states = nn.gelu_approx(hidden_states)
        else:
            raise ValueError(
                "unsupported Qwen3-Omni audio activation "
                f"{self.config.activation_function!r}"
            )
        hidden_states = self.proj2(hidden_states)
        return hidden_states
