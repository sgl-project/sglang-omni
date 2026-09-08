# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni audio encoder."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten

from sglang_omni.models.qwen3_asr.mlx.model import (
    SinusoidalPositionEmbedding,
    _get_feat_extract_output_lengths,
)
from sglang_omni.models.qwen3_omni.mlx.common import load_qwen3_omni_mlx_component
from sglang_omni.models.qwen3_omni.mlx.config import AudioConfig, Qwen3OmniMlxConfig
from sglang_omni.models.qwen3_omni.mlx.runner import read_qwen3_omni_component_weights

_AUDIO_PREFIXES = ("thinker.audio_tower.", "audio_tower.")
_AUDIO_LOCAL_PREFIXES = (
    "audio_tower.",
    "conv2d1.",
    "conv2d2.",
    "conv2d3.",
    "conv_out.",
    "layers.",
    "ln_post.",
    "proj1.",
    "proj2.",
)
_CONVOLUTION_WEIGHTS = {
    "conv2d1.weight",
    "conv2d2.weight",
    "conv2d3.weight",
}


def _torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    tensor = tensor.detach().cpu()
    if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.float()
    return mx.array(tensor.numpy())


def _mlx_to_torch(array: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(array.astype(mx.float32))))


def qwen3_omni_audio_output_lengths(input_lengths: mx.array) -> mx.array:
    """Match the pinned Hugging Face three-layer audio CNN length arithmetic."""

    return _get_feat_extract_output_lengths(input_lengths.astype(mx.int32))


def _chunk_lengths(
    feature_lengths: Sequence[int],
    *,
    chunk_size: int,
) -> tuple[int, ...]:
    lengths: list[int] = []
    for feature_length in feature_lengths:
        if feature_length <= 0:
            raise ValueError("audio feature lengths must be positive")
        full_chunks, remainder = divmod(feature_length, chunk_size)
        lengths.extend([chunk_size] * full_chunks)
        if remainder:
            lengths.append(remainder)
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
    chunk_size = n_window * 2
    window_ratio = n_window_infer // chunk_size
    if window_ratio <= 0:
        raise ValueError("n_window_infer must cover at least one audio chunk")

    lengths = tuple(int(value) for value in np.asarray(feature_lengths).tolist())
    chunks = _chunk_lengths(lengths, chunk_size=chunk_size)
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
    return tuple(segments)


def sanitize_audio_weights(
    weights: Mapping[str, mx.array],
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, mx.array]:
    """Map official and converted audio weights onto the native MLX model."""

    sanitized: dict[str, mx.array] = {}
    for source_key, value in weights.items():
        key = source_key
        for prefix in _AUDIO_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break

        if key in _CONVOLUTION_WEIGHTS:
            expected_shape = expected_shapes.get(key)
            source_shape = tuple(value.shape)
            if expected_shape is None:
                raise ValueError(
                    f"Qwen3-Omni audio model has no target shape for {key!r}"
                )
            if source_shape != expected_shape:
                candidate = value.transpose(0, 2, 3, 1) if value.ndim == 4 else None
                if candidate is None or tuple(candidate.shape) != expected_shape:
                    raise ValueError(
                        f"Qwen3-Omni audio tensor {source_key!r} has shape "
                        f"{source_shape}; expected MLX shape {expected_shape} "
                        "or a Torch Conv2D layout that transposes to it"
                    )
                value = candidate

        if key in sanitized:
            raise ValueError(
                f"Qwen3-Omni audio weights {source_key!r} and another source "
                f"both map to {key!r}"
            )
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

    def _activate(self, hidden_states: mx.array) -> mx.array:
        if self.activation_function == "gelu":
            return nn.gelu(hidden_states)
        if self.activation_function in ("gelu_new", "gelu_pytorch_tanh"):
            return nn.gelu_approx(hidden_states)
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
        hidden_states = self.fc2(self._activate(self.fc1(hidden_states)))
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
        self.positional_embedding = SinusoidalPositionEmbedding(
            config.max_source_positions,
            config.d_model,
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def _pack_features(
        self,
        input_features: mx.array,
        feature_attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape (batch, mel_bins, frames)")
        if feature_attention_mask.ndim != 2:
            raise ValueError("feature_attention_mask must have shape (batch, frames)")
        if (
            input_features.shape[0] != feature_attention_mask.shape[0]
            or input_features.shape[2] != feature_attention_mask.shape[1]
        ):
            raise ValueError(
                "input_features and feature_attention_mask batch/time shapes differ"
            )
        if input_features.shape[1] != self.config.num_mel_bins:
            raise ValueError(
                f"input_features has {input_features.shape[1]} mel bins; "
                f"expected {self.config.num_mel_bins}"
            )

        mask = np.asarray(feature_attention_mask).astype(bool, copy=False)
        feature_lengths = mask.sum(axis=1, dtype=np.int64)
        if np.any(feature_lengths <= 0):
            raise ValueError("audio feature masks must select at least one frame")
        rows = [
            mx.take(
                input_features[row],
                mx.array(np.flatnonzero(mask[row]), dtype=mx.int32),
                axis=1,
            )
            for row in range(input_features.shape[0])
        ]
        return (
            mx.concatenate(rows, axis=1),
            mx.array(feature_lengths, dtype=mx.int32),
        )

    def _chunk_and_pad(
        self,
        packed_features: mx.array,
        feature_lengths: mx.array,
    ) -> tuple[mx.array, tuple[int, ...]]:
        lengths = tuple(int(value) for value in np.asarray(feature_lengths).tolist())
        chunk_size = self.n_window * 2
        chunk_lengths = _chunk_lengths(lengths, chunk_size=chunk_size)
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

        max_chunk_length = max(chunk_lengths)
        padded = [
            mx.pad(chunk, ((0, 0), (0, max_chunk_length - length)))
            for chunk, length in zip(chunks, chunk_lengths, strict=True)
        ]
        return mx.stack(padded, axis=0), chunk_lengths

    def _convolve(self, padded_features: mx.array) -> mx.array:
        chunks: list[mx.array] = []
        features = padded_features[:, :, :, None].astype(self.conv2d1.weight.dtype)
        for start in range(0, features.shape[0], self.conv_chunksize):
            feature_chunk = features[start : start + self.conv_chunksize]
            hidden = nn.gelu(self.conv2d1(feature_chunk))
            hidden = nn.gelu(self.conv2d2(hidden))
            hidden = nn.gelu(self.conv2d3(hidden))
            chunks.append(hidden)
        return mx.concatenate(chunks, axis=0)

    def encode_features(
        self,
        input_features: mx.array,
        feature_attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        packed_features, feature_lengths = self._pack_features(
            input_features,
            feature_attention_mask,
        )
        padded_features, chunk_lengths = self._chunk_and_pad(
            packed_features,
            feature_lengths,
        )
        hidden_states = self._convolve(padded_features)
        batch, frequency, time, channels = hidden_states.shape
        hidden_states = hidden_states.transpose(0, 2, 3, 1).reshape(
            batch,
            time,
            channels * frequency,
        )
        hidden_states = self.conv_out(hidden_states)
        positions = self.positional_embedding(hidden_states.shape[1]).astype(
            hidden_states.dtype
        )
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
        output_lengths = qwen3_omni_audio_output_lengths(feature_lengths)
        return hidden_states, output_lengths

    def __call__(
        self,
        input_features: mx.array,
        feature_attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return self.encode_features(input_features, feature_attention_mask)


def load_qwen3_omni_mlx_audio(model_path: str) -> Qwen3OmniMlxAudioEncoder:
    """Load the audio component from an official or converted checkpoint."""

    from sglang.srt.hardware_backend.mlx.remote_code_gate import (
        ensure_remote_code_allowed,
        resolve_model_directory,
    )

    directory = Path(resolve_model_directory(model_path))
    ensure_remote_code_allowed(directory, False)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    root_config = Qwen3OmniMlxConfig.from_dict(raw)
    model = Qwen3OmniMlxAudioEncoder(root_config.audio)
    expected_shapes = {
        key: tuple(value.shape) for key, value in tree_flatten(model.parameters())
    }
    weights = read_qwen3_omni_component_weights(
        directory,
        component="audio",
        official_prefixes=_AUDIO_PREFIXES,
        local_prefixes=_AUDIO_LOCAL_PREFIXES,
    )
    return load_qwen3_omni_mlx_component(
        model,
        weights,
        sanitizer=lambda raw_weights: sanitize_audio_weights(
            raw_weights,
            expected_shapes=expected_shapes,
        ),
        quantization=root_config.quantization,
    )


class Qwen3OmniMlxAudioStageEncoder:
    """Torch-compatible stage adapter around the native MLX audio tower."""

    def __init__(self, model_path: str) -> None:
        self.audio_tower = load_qwen3_omni_mlx_audio(model_path)

    @staticmethod
    def _padded_from_packed(
        packed_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if packed_features.ndim != 2:
            raise ValueError(
                "prepacked input_features must have shape (mel_bins, total_frames)"
            )
        lengths = feature_lengths.detach().cpu().to(dtype=torch.long).view(-1)
        if torch.any(lengths <= 0):
            raise ValueError("audio_feature_lengths must be positive")
        if int(lengths.sum().item()) != int(packed_features.shape[-1]):
            raise ValueError(
                "audio_feature_lengths do not sum to the prepacked frame count"
            )
        max_length = int(lengths.max().item())
        padded = torch.zeros(
            lengths.shape[0],
            packed_features.shape[0],
            max_length,
            dtype=packed_features.dtype,
        )
        mask = torch.zeros(lengths.shape[0], max_length, dtype=torch.bool)
        cursor = 0
        for row, length in enumerate(lengths.tolist()):
            padded[row, :, :length] = packed_features[:, cursor : cursor + length]
            mask[row, :length] = True
            cursor += length
        return padded, mask

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        input_features = inputs.get("input_features")
        if not isinstance(input_features, torch.Tensor):
            raise TypeError("input_features must be a torch.Tensor")

        feature_attention_mask = inputs.get("feature_attention_mask")
        if isinstance(feature_attention_mask, torch.Tensor):
            if input_features.ndim == 2:
                input_features = input_features.unsqueeze(0)
            feature_attention_mask = feature_attention_mask.detach().cpu()
            if feature_attention_mask.ndim == 1:
                feature_attention_mask = feature_attention_mask.unsqueeze(0)
            feature_lengths = feature_attention_mask.to(dtype=torch.long).sum(dim=1)
            padded_features = input_features.detach().cpu()
        else:
            feature_lengths = inputs.get("audio_feature_lengths")
            if not isinstance(feature_lengths, torch.Tensor):
                raise ValueError(
                    "audio_feature_lengths or feature_attention_mask is required"
                )
            padded_features, feature_attention_mask = self._padded_from_packed(
                input_features.detach().cpu(),
                feature_lengths,
            )
            feature_lengths = (
                feature_lengths.detach().cpu().to(dtype=torch.long).view(-1)
            )

        audio_embeds, output_lengths = self.audio_tower(
            _torch_to_mlx(padded_features),
            _torch_to_mlx(feature_attention_mask),
        )
        mx.eval(audio_embeds, output_lengths)
        return {
            "audio_embeds": _mlx_to_torch(audio_embeds),
            "audio_feature_lengths": feature_lengths,
            "audio_output_lengths": torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(output_lengths).astype(np.int64, copy=False)
                )
            ),
        }
