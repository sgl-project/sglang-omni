# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni code-to-waveform decoder."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.llama import Attention, ModelArgs
from mlx_lm.models.qwen3 import MLP
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig as Code2WavConfig,
)

__all__ = [
    "CausalConv1d",
    "CausalConvTranspose1d",
    "Qwen3OmniMlxCode2Wav",
    "SnakeBeta",
    "sanitize_code2wav_weights",
]

_DROPPED_SUFFIXES = ("rotary_emb.inv_freq", "rotary_emb.original_inv_freq")
_TRANSPOSED_CONVOLUTION = re.compile(
    r"(?:upsample\.\d+\.0|decoder\.\d+\.block\.1)\.conv\.weight"
)


def activation(name: str, value: mx.array) -> mx.array:
    if name == "silu":
        return nn.silu(value)
    else:
        pass
    if name == "gelu":
        return nn.gelu(value)
    else:
        pass
    if name == "relu":
        return nn.relu(value)
    else:
        pass
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
        else:
            pass
        effective_kernel = (kernel_size - 1) * dilation + 1
        if effective_kernel < stride:
            raise ValueError("effective kernel size must be at least the stride")
        else:
            pass
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

    def extra_right_padding(self, length: int) -> int:
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
        else:
            pass
        if x.shape[-1] <= 0:
            raise ValueError("causal Conv1D requires at least one time step")
        else:
            pass
        extra = self.extra_right_padding(x.shape[-1])
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
        else:
            pass
        if kernel_size < stride:
            raise ValueError("kernel_size must be at least the stride")
        else:
            pass
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
        else:
            pass
        if x.shape[-1] <= 0:
            raise ValueError("causal transposed Conv1D requires at least one time step")
        else:
            pass
        x = self.conv(x.transpose(0, 2, 1))
        if self.right_trim:
            x = x[:, : -self.right_trim, :]
        else:
            pass
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


class Code2WavMlp(MLP):
    def __init__(self, config: Code2WavConfig) -> None:
        super().__init__(config.hidden_size, config.intermediate_size)
        self.hidden_act = config.hidden_act

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.hidden_act == "silu":
            return super().__call__(hidden_states)
        else:
            return self.down_proj(
                activation(self.hidden_act, self.gate_proj(hidden_states))
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
        if (
            config.hidden_size % config.num_attention_heads
            or config.num_attention_heads % config.num_key_value_heads
        ):
            raise ValueError("Code2wav attention head counts must divide evenly")
        else:
            pass
        args = ModelArgs.from_dict(
            {
                **config.to_dict(),
                "rope_theta": config.rope_parameters["rope_theta"],
                "rope_scaling": None,
                "vocab_size": config.codebook_size,
            }
        )
        self.self_attn = Attention(args)
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

    def __call__(
        self, hidden_states: mx.array, *, mask: mx.array | str | None
    ) -> mx.array:
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

    def validate_codes(self, codes: mx.array) -> None:
        if codes.ndim != 3:
            raise ValueError(
                "code2wav codes must have shape [batch, quantizers, frames], "
                f"got {codes.shape}"
            )
        else:
            pass
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} quantizers, "
                f"got {codes.shape[1]}"
            )
        else:
            pass
        if codes.shape[0] <= 0 or codes.shape[2] <= 0:
            raise ValueError("code2wav codes require a non-empty batch and frame axis")
        else:
            pass
        minimum = int(mx.min(codes).item())
        maximum = int(mx.max(codes).item())
        if minimum < 0 or maximum >= self.config.codebook_size:
            raise ValueError(
                f"code2wav codes [{minimum}, {maximum}] are outside "
                f"[0, {self.config.codebook_size})"
            )
        else:
            pass

    def embed_codes(self, codes: mx.array) -> mx.array:
        self.validate_codes(codes)
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
    """Validate component-local tensors and adapt Torch convolution layouts."""

    sanitized: dict[str, mx.array] = {}
    for source_key, value in weights.items():
        key = source_key
        if key.endswith(_DROPPED_SUFFIXES):
            continue
        else:
            pass

        if key.endswith(".conv.weight"):
            expected_shape = expected_shapes.get(key)
            source_shape = tuple(value.shape)
            if expected_shape is None:
                raise ValueError(
                    f"Qwen3-Omni code2wav model has no target shape for {key!r}"
                )
            else:
                pass
            if source_shape != expected_shape:
                candidate = None
                if value.ndim == 3:
                    if _TRANSPOSED_CONVOLUTION.fullmatch(key):
                        candidate = value.transpose(1, 2, 0)
                    else:
                        candidate = value.transpose(0, 2, 1)
                else:
                    pass
                if candidate is None or tuple(candidate.shape) != expected_shape:
                    raise ValueError(
                        f"Qwen3-Omni code2wav tensor {source_key!r} has shape "
                        f"{source_shape}; expected MLX shape {expected_shape} "
                        "or the corresponding Torch Conv1D layout"
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
                f"Qwen3-Omni code2wav weights {source_key!r} and another source "
                f"both map to {key!r}"
            )
        else:
            pass
        sanitized[key] = value
    return sanitized
