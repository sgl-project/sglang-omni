# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-ASR (Copyright 2025 Prince Canuma and contributors).
"""Qwen3 text decoder in plain MLX, its key/value cache, and greedy decoding."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from sglang_omni_mlx.transcription import CancelCheck, TranscriptionCancelled

KV_CACHE_STEP_TOKENS = 256


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


def text_decoder_config(text: dict[str, object]) -> TextDecoderConfig:
    """The decoder config from a checkpoint's text_config, either rope spelling."""
    fields = {
        name: text[name]
        for name in TextDecoderConfig.__dataclass_fields__
        if name != "rope_theta"
    }
    if "rope_theta" in text:
        rope_theta = text["rope_theta"]
    else:
        rope_theta = text["rope_parameters"]["rope_theta"]
    return TextDecoderConfig(**fields, rope_theta=rope_theta)


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


def greedy_tokens(
    decoder: TextDecoder,
    embeddings: mx.array,
    caches: list[KVCache],
    cancel: CancelCheck,
) -> Iterator[int]:
    """Greedy token ids after the prompt embeddings, until the caller stops reading."""
    next_token = mx.argmax(decoder(embeddings, caches))
    mx.async_eval(next_token)
    while True:
        if cancel.is_set():
            raise TranscriptionCancelled()
        else:
            pass
        token = next_token
        # Queue the following step before reading this token, so the GPU
        # decodes while the caller checks its stop rules.
        next_token = mx.argmax(
            decoder(decoder.embed_tokens(token.reshape(1, 1)), caches)
        )
        mx.async_eval(next_token)
        yield int(token.item())
