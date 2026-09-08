# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni code-to-waveform decoder."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

from sglang_omni.models.qwen3_omni.mlx.common import load_qwen3_omni_mlx_component
from sglang_omni.models.qwen3_omni.mlx.config import Code2WavConfig, Qwen3OmniMlxConfig
from sglang_omni.models.qwen3_omni.mlx.runner import read_qwen3_omni_component_weights

__all__ = [
    "CausalConv1d",
    "CausalConvTranspose1d",
    "Qwen3OmniMlxCode2Wav",
    "SnakeBeta",
    "load_qwen3_omni_mlx_code2wav",
    "sanitize_code2wav_weights",
]

_CODE2WAV_PREFIXES = ("code2wav.",)
_CODE2WAV_LOCAL_PREFIXES = (
    "pre_transformer.",
    "code_embedding.",
    "upsample.",
    "decoder.",
)
_DROPPED_SUFFIXES = ("rotary_emb.inv_freq", "rotary_emb.original_inv_freq")
_TRANSPOSED_CONVOLUTION = re.compile(
    r"(?:upsample\.\d+\.0|decoder\.\d+\.block\.1)\.conv\.weight"
)


def _activation(name: str, value: mx.array) -> mx.array:
    if name == "silu":
        return nn.silu(value)
    if name == "gelu":
        return nn.gelu(value)
    if name == "relu":
        return nn.relu(value)
    raise ValueError(f"unsupported code2wav activation {name!r}")


class CausalConv1d(nn.Module):
    """Channel-first causal Conv1D with Hugging Face length arithmetic."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or dilation <= 0 or stride <= 0:
            raise ValueError("kernel_size, dilation, and stride must be positive")
        effective_kernel = (kernel_size - 1) * dilation + 1
        if effective_kernel < stride:
            raise ValueError("effective kernel size must be at least the stride")
        self.stride = stride
        self.kernel_size = effective_kernel
        self.left_padding = effective_kernel - stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )

    def _extra_right_padding(self, length: int) -> int:
        frames = (length - self.kernel_size + self.left_padding) / self.stride + 1
        ideal_length = (
            (math.ceil(frames) - 1) * self.stride + self.kernel_size - self.left_padding
        )
        return ideal_length - length

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 3:
            raise ValueError(
                f"causal Conv1D expects [batch, channels, time], got {x.shape}"
            )
        if x.shape[-1] <= 0:
            raise ValueError("causal Conv1D requires at least one time step")
        extra = self._extra_right_padding(x.shape[-1])
        x = x.transpose(0, 2, 1)
        x = mx.pad(x, [(0, 0), (self.left_padding, extra), (0, 0)])
        return self.conv(x).transpose(0, 2, 1)


class CausalConvTranspose1d(nn.Module):
    """Channel-first causal transposed Conv1D with exact stride expansion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or stride <= 0:
            raise ValueError("kernel_size and stride must be positive")
        if kernel_size < stride:
            raise ValueError("kernel_size must be at least the stride")
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.right_trim = kernel_size - stride

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 3:
            raise ValueError(
                f"causal transposed Conv1D expects [batch, channels, time], got {x.shape}"
            )
        if x.shape[-1] <= 0:
            raise ValueError("causal transposed Conv1D requires at least one time step")
        x = self.conv(x.transpose(0, 2, 1))
        if self.right_trim:
            x = x[:, : -self.right_trim, :]
        return x.transpose(0, 2, 1)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.full((dim,), 1e-6)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).transpose(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = nn.gelu(self.pwconv1(hidden_states))
        hidden_states = self.pwconv2(hidden_states) * self.gamma
        return residual + hidden_states.transpose(0, 2, 1)


class Code2WavAttention(nn.Module):
    def __init__(self, config: Code2WavConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError(
                "code2wav hidden_size must be divisible by num_attention_heads"
            )
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError(
                "code2wav num_attention_heads must be divisible by "
                "num_key_value_heads"
            )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def __call__(self, hidden_states: mx.array, *, mask: Any) -> mx.array:
        batch, length, _ = hidden_states.shape
        queries = (
            self.q_proj(hidden_states)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(hidden_states)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(hidden_states)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        queries = self.rope(queries)
        keys = self.rope(keys)
        attention = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=self.scale,
            mask=mask,
        )
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(attention)


class Code2WavMlp(nn.Module):
    def __init__(self, config: Code2WavConfig) -> None:
        super().__init__()
        self.hidden_act = config.hidden_act
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            _activation(self.hidden_act, self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class LayerScale(nn.Module):
    def __init__(self, channels: int, initial_scale: float) -> None:
        super().__init__()
        self.scale = mx.full((channels,), initial_scale)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return hidden_states * self.scale


class Code2WavTransformerLayer(nn.Module):
    def __init__(self, config: Code2WavConfig) -> None:
        super().__init__()
        self.self_attn = Code2WavAttention(config)
        self.mlp = Code2WavMlp(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn_layer_scale = LayerScale(
            config.hidden_size,
            config.layer_scale_initial_scale,
        )
        self.mlp_layer_scale = LayerScale(
            config.hidden_size,
            config.layer_scale_initial_scale,
        )

    def __call__(self, hidden_states: mx.array, *, mask: Any) -> mx.array:
        hidden_states = hidden_states + self.self_attn_layer_scale(
            self.self_attn(self.input_layernorm(hidden_states), mask=mask)
        )
        return hidden_states + self.mlp_layer_scale(
            self.mlp(self.post_attention_layernorm(hidden_states))
        )


class Code2WavTransformerModel(nn.Module):
    def __init__(self, config: Code2WavConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = [
            Code2WavTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        mask = create_attention_mask(
            hidden_states,
            window_size=self.config.sliding_window,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
        return self.norm(hidden_states)


class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.alpha = mx.zeros((in_features,)) * alpha
        self.beta = mx.zeros((in_features,)) * alpha

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)[None, :, None]
        beta = mx.exp(self.beta)[None, :, None]
        return x + mx.sin(x * alpha) ** 2 / (beta + 1e-9)


class DecoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.conv1(self.act1(hidden_states))
        hidden_states = self.conv2(self.act2(hidden_states))
        return hidden_states + residual


class DecoderBlock(nn.Module):
    def __init__(self, config: Code2WavConfig, layer_index: int) -> None:
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_index
        out_dim = config.decoder_dim // 2 ** (layer_index + 1)
        upsample_rate = config.upsample_rates[layer_index]
        self.block = [
            SnakeBeta(in_dim),
            CausalConvTranspose1d(
                in_dim,
                out_dim,
                kernel_size=2 * upsample_rate,
                stride=upsample_rate,
            ),
            DecoderResidualUnit(out_dim, 1),
            DecoderResidualUnit(out_dim, 3),
            DecoderResidualUnit(out_dim, 9),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return hidden_states


class Qwen3OmniMlxCode2Wav(nn.Module):
    """Native MLX code2wav model with channel-first waveform boundaries."""

    def __init__(self, config: Code2WavConfig) -> None:
        super().__init__()
        self.config = config
        self.total_upsample = math.prod(config.upsampling_ratios) * math.prod(
            config.upsample_rates
        )
        self.pre_transformer = Code2WavTransformerModel(config)
        self.code_embedding = nn.Embedding(
            config.codebook_size * config.num_quantizers,
            config.hidden_size,
        )
        self.code_offset = tuple(
            index * config.codebook_size for index in range(config.num_quantizers)
        )
        self.upsample = [
            [
                CausalConvTranspose1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel_size=factor,
                    stride=factor,
                ),
                ConvNeXtBlock(config.hidden_size),
            ]
            for factor in config.upsampling_ratios
        ]
        decoder: list[nn.Module] = [
            CausalConv1d(config.hidden_size, config.decoder_dim, kernel_size=7)
        ]
        decoder.extend(
            DecoderBlock(config, index) for index in range(len(config.upsample_rates))
        )
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder.extend(
            [
                SnakeBeta(output_dim),
                CausalConv1d(output_dim, 1, kernel_size=7),
            ]
        )
        self.decoder = decoder

    def _validate_codes(self, codes: mx.array) -> None:
        if codes.ndim != 3:
            raise ValueError(
                "code2wav codes must have shape [batch, quantizers, frames], "
                f"got {codes.shape}"
            )
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} quantizers, "
                f"got {codes.shape[1]}"
            )
        if codes.shape[0] <= 0 or codes.shape[2] <= 0:
            raise ValueError("code2wav codes require a non-empty batch and frame axis")
        minimum = int(mx.min(codes).item())
        maximum = int(mx.max(codes).item())
        if minimum < 0 or maximum >= self.config.codebook_size:
            raise ValueError(
                f"code2wav codes [{minimum}, {maximum}] are outside "
                f"[0, {self.config.codebook_size})"
            )

    def embed_codes(self, codes: mx.array) -> mx.array:
        self._validate_codes(codes)
        codes = codes.astype(mx.int32)
        offsets = mx.array(self.code_offset, dtype=mx.int32)[None, :, None]
        return self.code_embedding(codes + offsets).mean(axis=1)

    def __call__(self, codes: mx.array) -> mx.array:
        hidden_states = self.pre_transformer(self.embed_codes(codes))
        hidden_states = hidden_states.transpose(0, 2, 1)
        for stage in self.upsample:
            for layer in stage:
                hidden_states = layer(hidden_states)
        waveform = hidden_states
        for layer in self.decoder:
            waveform = layer(waveform)
        return mx.clip(waveform, -1.0, 1.0)


def sanitize_code2wav_weights(
    weights: Mapping[str, mx.array],
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, mx.array]:
    """Map official or converted code2wav tensors onto the native MLX model."""

    sanitized: dict[str, mx.array] = {}
    for source_key, value in weights.items():
        key = source_key
        for prefix in _CODE2WAV_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.endswith(_DROPPED_SUFFIXES):
            continue

        if key.endswith(".conv.weight"):
            expected_shape = expected_shapes.get(key)
            source_shape = tuple(value.shape)
            if expected_shape is None:
                raise ValueError(
                    f"Qwen3-Omni code2wav model has no target shape for {key!r}"
                )
            if source_shape != expected_shape:
                candidate = None
                if value.ndim == 3:
                    if _TRANSPOSED_CONVOLUTION.fullmatch(key):
                        candidate = value.transpose(1, 2, 0)
                    else:
                        candidate = value.transpose(0, 2, 1)
                if candidate is None or tuple(candidate.shape) != expected_shape:
                    raise ValueError(
                        f"Qwen3-Omni code2wav tensor {source_key!r} has shape "
                        f"{source_shape}; expected MLX shape {expected_shape} "
                        "or the corresponding Torch Conv1D layout"
                    )
                value = candidate

        if key in sanitized:
            raise ValueError(
                f"Qwen3-Omni code2wav weights {source_key!r} and another source "
                f"both map to {key!r}"
            )
        sanitized[key] = value
    return sanitized


def load_qwen3_omni_mlx_code2wav(model_path: str) -> Qwen3OmniMlxCode2Wav:
    """Load native code2wav from an official or converted Qwen3-Omni checkpoint."""

    from sglang.srt.hardware_backend.mlx.remote_code_gate import (
        ensure_remote_code_allowed,
        resolve_model_directory,
    )

    directory = Path(resolve_model_directory(model_path))
    ensure_remote_code_allowed(directory, False)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    root_config = Qwen3OmniMlxConfig.from_dict(raw)
    model = Qwen3OmniMlxCode2Wav(root_config.code2wav)
    expected_shapes = {
        key: tuple(value.shape) for key, value in tree_flatten(model.parameters())
    }
    weights = read_qwen3_omni_component_weights(
        directory,
        component="code2wav",
        official_prefixes=_CODE2WAV_PREFIXES,
        local_prefixes=_CODE2WAV_LOCAL_PREFIXES,
    )
    if any(key.startswith(_CODE2WAV_PREFIXES) for key in weights):
        # Task 7 removes the generated dense sidecar. Until then, a converted
        # checkpoint can contain both its native root tensors and the legacy
        # Torch sidecar; the native namespace is authoritative for MLX.
        weights = {
            key: value
            for key, value in weights.items()
            if key.startswith(_CODE2WAV_PREFIXES)
        }
    return load_qwen3_omni_mlx_component(
        model,
        weights,
        sanitizer=lambda raw_weights: sanitize_code2wav_weights(
            raw_weights,
            expected_shapes=expected_shapes,
        ),
        quantization=root_config.quantization,
    )
