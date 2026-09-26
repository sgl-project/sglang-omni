# SPDX-License-Identifier: Apache-2.0
# Module attribute names mirror the checkpoint keys so weights load without renaming.

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

from .config import ModelConfig


class WhisperAttention(nn.Module):
    """Multi-head attention shared by the Whisper encoder and decoder."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # SGLang's MLX cache sizing reads num_kv_heads and scale off this module.
        self.num_kv_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.scale = self.scaling

        if (self.head_dim * num_heads) != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={embed_dim}"
                f" and num_heads={num_heads})."
            )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def split_heads(self, states: mx.array) -> mx.array:
        bsz, seq_len, _ = states.shape
        return states.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        is_cross: bool = False,
    ) -> mx.array:
        """Run attention; cross-attention keys and values are cached once.

        is_cross is explicit because decode steps reach cross-attention without
        encoder output.
        """
        bsz, seq_len, _ = hidden_states.shape
        query_states = self.split_heads(self.q_proj(hidden_states) * self.scaling)

        if not is_cross:
            key_states = self.split_heads(self.k_proj(hidden_states))
            value_states = self.split_heads(self.v_proj(hidden_states))
            if cache is not None:
                key_states, value_states = cache.update_and_fetch(
                    key_states, value_states
                )
        elif cache is not None and cache[0] is not None:
            key_states, value_states = cache[0], cache[1]
        elif key_value_states is None:
            raise ValueError(
                "cross-attention needs the encoder output until its cache is "
                "populated; pass encoder_hidden_states on the prefill call"
            )
        else:
            key_states = self.split_heads(self.k_proj(key_value_states))
            value_states = self.split_heads(self.v_proj(key_value_states))
            if cache is not None:
                cache[0], cache[1] = key_states, value_states

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=1.0, mask=mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class WhisperEncoderLayer(nn.Module):
    """A single Whisper encoder block (pre-norm residual, GELU feed-forward)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            config.d_model, config.encoder_attention_heads
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    """Whisper audio encoder: two strided convolutions then transformer layers."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins, config.d_model, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.d_model, config.d_model, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, input_features: mx.array) -> mx.array:
        """Encode a (batch, num_mel_bins, frames) mel spectrogram."""
        hidden_states = input_features.transpose(0, 2, 1)
        hidden_states = nn.gelu(self.conv1(hidden_states))
        hidden_states = nn.gelu(self.conv2(hidden_states))

        # Slicing past embed_positions would otherwise fail as an opaque broadcast.
        length = hidden_states.shape[1]
        if length > self.config.max_source_positions:
            raise ValueError(
                f"encoded length {length} exceeds max_source_positions "
                f"{self.config.max_source_positions}; Whisper expects at most "
                f"{self.config.max_source_positions * 2} mel frames per window, "
                f"got {input_features.shape[-1]}"
            )
        hidden_states = hidden_states + self.embed_positions.weight[:length]

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


class WhisperDecoderLayer(nn.Module):
    """A Whisper decoder block: self-attention, cross-attention, feed-forward."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            config.d_model, config.decoder_attention_heads
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = WhisperAttention(
            config.d_model, config.decoder_attention_heads
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        self_cache = None if cache is None else cache[0]
        cross_cache = None if cache is None else cache[1]

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask, cache=self_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            cache=cross_cache,
            is_cross=True,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class WhisperDecoder(nn.Module):
    """Whisper text decoder with learned token and position embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_target_positions, config.d_model)
        self.layers = [
            WhisperDecoderLayer(config) for _ in range(config.decoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        offset: int = 0,
    ) -> mx.array:
        """Decode tokens; offset is the position of input_ids[:, 0] in the sequence."""
        length = input_ids.shape[1]
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = (
            hidden_states + self.embed_positions.weight[offset : offset + length]
        )

        layer_caches = [None] * len(self.layers) if cache is None else cache
        for layer, layer_cache in zip(self.layers, layer_caches):
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                mask=mask,
                cache=layer_cache,
            )
        return self.layer_norm(hidden_states)


class WhisperInnerModel(nn.Module):
    """Container matching the checkpoint's model.* key prefix."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)


class WhisperMlxModel(nn.Module):
    """Native MLX Whisper encoder-decoder model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = WhisperInnerModel(config)

    @property
    def encoder(self) -> WhisperEncoder:
        return self.model.encoder

    @property
    def decoder(self) -> WhisperDecoder:
        return self.model.decoder

    def encode(self, input_features: mx.array) -> mx.array:
        return self.model.encoder(input_features)

    def make_cache(self) -> List[CacheList]:
        """One self-attention and one cross-attention cache per decoder layer."""
        return [
            CacheList(KVCache(), ArraysCache(2))
            for _ in range(self.config.decoder_layers)
        ]

    def __call__(
        self, input_ids: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        """Decode step for SGLang's MLX runner; prefill filled cross-attention."""
        return self.decode(input_ids, None, cache=cache)

    def decode(
        self,
        input_ids: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        offset: Optional[int] = None,
    ) -> mx.array:
        """Return next-token logits for input_ids."""
        if offset is None:
            offset = 0 if cache is None else cache[0][0].offset
        hidden_states = self.model.decoder(
            input_ids,
            encoder_hidden_states,
            mask=create_attention_mask(
                input_ids, None if cache is None else cache[0][0]
            ),
            cache=cache,
            offset=offset,
        )
        return self.model.decoder.embed_tokens.as_linear(hidden_states)

    _CONV_WEIGHTS = ("model.encoder.conv1.weight", "model.encoder.conv2.weight")

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert PyTorch checkpoint layout to MLX layout; idempotent.

        proj_out is dropped: it is tied to model.decoder.embed_tokens.
        """
        sanitized = {}
        for k, v in weights.items():
            if not k.startswith("model."):
                continue
            if k in self._CONV_WEIGHTS:
                v = self.to_mlx_conv1d(k, v)
            sanitized[k] = v
        return sanitized

    def to_mlx_conv1d(self, key: str, value: mx.array) -> mx.array:
        """Transpose a Conv1d kernel into MLX's (out, kernel, in) layout."""
        conv = self.model.encoder.conv1 if "conv1" in key else self.model.encoder.conv2
        expected = conv.weight.shape
        if value.ndim != 3:
            raise ValueError(f"{key}: expected a 3D Conv1d kernel, got {value.shape}")
        if value.shape == expected:
            return value
        transposed = value.transpose(0, 2, 1)
        if transposed.shape != expected:
            raise ValueError(
                f"{key}: cannot map {value.shape} onto MLX Conv1d weight {expected}"
            )
        return transposed


Model = WhisperMlxModel
ModelArgs = ModelConfig
