# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Prince Canuma and contributors.
# Adapted from MLX-Audio 0.4.6 Qwen3-TTS speech_tokenizer.py and config.py.
#
# MIT License
#
# Copyright (c) 2024 Prince Canuma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Causal Transformer used by the MLX speech decoder."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import MLP
from pydantic import BaseModel, ConfigDict
from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache


class Qwen3TTSMlxDecoderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attention_bias: bool
    latent_dim: int
    codebook_dim: int
    codebook_size: int
    decoder_dim: int
    hidden_size: int
    intermediate_size: int
    layer_scale_initial_scale: float
    head_dim: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    num_quantizers: int
    num_semantic_quantizers: int
    rms_norm_eps: float
    rope_theta: float
    upsample_rates: list[int]
    upsampling_ratios: list[int]


class DecoderRMSNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float) -> None:
        super().__init__()
        self.weight: mx.array = mx.ones((channels,))
        self.epsilon: float = epsilon

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden_float = hidden.astype(mx.float32)
        variance = mx.mean(hidden_float**2, axis=-1, keepdims=True)
        normalized = hidden_float * mx.rsqrt(variance + self.epsilon)
        return (self.weight * normalized).astype(hidden.dtype)


class LayerScale(nn.Module):
    def __init__(self, channels: int, initial_scale: float) -> None:
        super().__init__()
        self.scale: mx.array = mx.ones((channels,)) * initial_scale

    def __call__(self, hidden: mx.array) -> mx.array:
        return hidden * self.scale


def apply_rotary_embedding(
    hidden: mx.array, cosine: mx.array, sine: mx.array
) -> mx.array:
    midpoint = hidden.shape[-1] // 2
    rotated = mx.concatenate([-hidden[..., midpoint:], hidden[..., :midpoint]], axis=-1)
    return (hidden * cosine) + (rotated * sine)


class DecoderAttention(nn.Module):
    def __init__(self, config: Qwen3TTSMlxDecoderConfig) -> None:
        super().__init__()
        self.head_dimension: int = config.head_dim
        self.num_heads: int = config.num_attention_heads
        self.num_key_value_heads: int = config.num_key_value_heads
        self.scale: float = config.head_dim**-0.5
        self.q_proj: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj: nn.Linear = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj: nn.Linear = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def __call__(
        self,
        hidden: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        mask: mx.array | None,
        cache: ContiguousAttentionKVCache | None = None,
    ) -> mx.array:
        batch_size, frame_count, _ = hidden.shape
        queries = (
            self.q_proj(hidden)
            .reshape(batch_size, frame_count, self.num_heads, self.head_dimension)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(hidden)
            .reshape(
                batch_size, frame_count, self.num_key_value_heads, self.head_dimension
            )
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(hidden)
            .reshape(
                batch_size, frame_count, self.num_key_value_heads, self.head_dimension
            )
            .transpose(0, 2, 1, 3)
        )
        cosine, sine = position_embeddings
        queries = apply_rotary_embedding(queries, cosine[:, None], sine[:, None])
        keys = apply_rotary_embedding(keys, cosine[:, None], sine[:, None])
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        else:
            pass
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, frame_count, -1)
        return self.o_proj(attended)


class DecoderTransformerLayer(nn.Module):
    def __init__(self, config: Qwen3TTSMlxDecoderConfig) -> None:
        super().__init__()
        self.self_attn: DecoderAttention = DecoderAttention(config)
        self.mlp: MLP = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm: DecoderRMSNorm = DecoderRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm: DecoderRMSNorm = DecoderRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn_layer_scale: LayerScale = LayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )
        self.mlp_layer_scale: LayerScale = LayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )

    def __call__(
        self,
        hidden: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        mask: mx.array | None,
        cache: ContiguousAttentionKVCache | None = None,
    ) -> mx.array:
        attended = self.self_attn(
            self.input_layernorm(hidden), position_embeddings, mask, cache
        )
        hidden = hidden + self.self_attn_layer_scale(attended)
        return hidden + self.mlp_layer_scale(
            self.mlp(self.post_attention_layernorm(hidden))
        )


class DecoderTransformer(nn.Module):
    def __init__(self, config: Qwen3TTSMlxDecoderConfig) -> None:
        super().__init__()
        self.layers: list[DecoderTransformerLayer] = [
            DecoderTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm: DecoderRMSNorm = DecoderRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        inverse_frequency = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim)
        )
        mx.eval(inverse_frequency)
        self._inverse_frequency: mx.array = (
            inverse_frequency  # noqa: leading-underscore - Not a checkpoint weight.
        )
        self.input_proj: nn.Linear = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj: nn.Linear = nn.Linear(config.hidden_size, config.latent_dim)

    def __call__(
        self,
        embeddings: mx.array,
        cache: list[ContiguousAttentionKVCache] | None = None,
    ) -> mx.array:
        batch_size, frame_count, _ = embeddings.shape
        hidden = self.input_proj(embeddings)
        offset = cache[0].offset if cache is not None else 0
        position_ids = mx.broadcast_to(
            mx.arange(offset, offset + frame_count)[None, :], (batch_size, frame_count)
        )
        inverse_frequency = (
            self._inverse_frequency
        )  # noqa: leading-underscore - Not a checkpoint weight.
        inverse_frequency = inverse_frequency[None, :, None].astype(mx.float32)
        positions = position_ids[:, None, :].astype(mx.float32)
        frequencies = (inverse_frequency * positions).transpose(0, 2, 1)
        rotary_embeddings = mx.concatenate([frequencies, frequencies], axis=-1)
        position_embeddings = (
            mx.cos(rotary_embeddings).astype(hidden.dtype),
            mx.sin(rotary_embeddings).astype(hidden.dtype),
        )
        if frame_count > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(
                frame_count
            ).astype(hidden.dtype)
            if offset > 0:
                mask = mx.pad(mask, [(0, 0), (offset, 0)])
            else:
                pass
        else:
            mask = None
        for i, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                position_embeddings,
                mask,
                cache[i] if cache is not None else None,
            )
        return self.output_proj(self.norm(hidden))
