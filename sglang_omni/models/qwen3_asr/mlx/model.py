# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-ASR (Copyright 2025 Prince Canuma and contributors).

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

from .config import AudioEncoderConfig, ModelConfig, TextConfig


def _rope_safe(rope, x: mx.array, offset: int) -> mx.array:
    """Apply RoPE, working around an mx.fast.rope bug.

    For a 4D tensor (B, heads, L, dim) with L == 1 and B > 1, mx.fast.rope
    (used by nn.RoPE) corrupts every batch row except the first. This only
    bites batched single-token decode (batch generation); single-sequence
    decode has B == 1 and is unaffected. Padding the sequence to length 2 and
    slicing keeps the fast kernel while producing the exact correct result.
    """
    if x.ndim == 4 and x.shape[0] > 1 and x.shape[2] == 1:
        x = mx.concatenate([x, mx.zeros_like(x)], axis=2)
        return rope(x, offset=offset)[:, :, :1, :]
    return rope(x, offset=offset)


def _floor_div(a: mx.array, b: int) -> mx.array:
    """Floor division matching Python semantics."""
    return mx.floor(a.astype(mx.float32) / b).astype(mx.int32)


def _get_feat_extract_output_lengths(input_lengths: mx.array) -> mx.array:
    """Compute output length of the convolutional layers."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = _floor_div(input_lengths_leave - 1, 2) + 1
    output_lengths = (
        _floor_div(_floor_div(feat_lengths - 1, 2) + 1 - 1, 2)
        + 1
        + (input_lengths // 100) * 13
    )
    return output_lengths


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for the audio encoder."""

    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidalPositionEmbedding needs even channels input")

        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = mx.exp(
            -log_timescale_increment * mx.arange(channels // 2, dtype=mx.float32)
        )
        positions = mx.arange(length, dtype=mx.float32)[:, None]
        scaled_time = positions * inv_timescales[None, :]
        self._positional_embedding = mx.concatenate(
            [mx.sin(scaled_time), mx.cos(scaled_time)], axis=1
        )
        mx.eval(self._positional_embedding)

    def __call__(self, seqlen: int) -> mx.array:
        return self._positional_embedding[:seqlen, :]


class AudioAttention(nn.Module):
    """Multi-headed attention for audio encoder."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}"
                f" and num_heads={self.num_heads})."
            )

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        bsz, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=1.0, mask=mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class AudioEncoderLayer(nn.Module):
    """A single transformer encoder layer for audio."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AudioEncoder(nn.Module):
    """Qwen3-ASR Audio Encoder with Conv2d frontend and transformer layers."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        self.conv2d1 = nn.Conv2d(
            1, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1
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

        freq_after_conv = ((((config.num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * freq_after_conv, embed_dim, bias=False
        )
        self.positional_embedding = SinusoidalPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, config.output_dim)

    def _create_block_attention_mask(
        self, seq_len: int, cu_seqlens: List[int], dtype: mx.Dtype
    ) -> mx.array:
        """Create attention mask for ragged/block attention."""
        mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            mask[start:end, start:end] = 0.0
        return mask

    def __call__(
        self,
        input_features: mx.array,
        feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if feature_attention_mask is not None:
            feature_lens = feature_attention_mask.sum(axis=-1).astype(mx.int32)
        else:
            feature_lens = mx.array(
                [input_features.shape[-1]] * input_features.shape[0], dtype=mx.int32
            )

        feature_lens_np = np.array(feature_lens)
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_size = self.n_window * 2
        chunk_num = np.ceil(feature_lens_np / chunk_size).astype(np.int32)

        chunk_lengths = []
        for i in range(len(feature_lens_np)):
            num_chunks = int(chunk_num[i])
            feat_len = int(feature_lens_np[i])
            for j in range(num_chunks):
                if j == num_chunks - 1:
                    remainder = feat_len % chunk_size
                    chunk_lengths.append(chunk_size if remainder == 0 else remainder)
                else:
                    chunk_lengths.append(chunk_size)

        chunk_lengths = np.array(chunk_lengths, dtype=np.int32)

        chunks = []
        for i in range(len(feature_lens_np)):
            feat = input_features[i]
            feat_len = int(feature_lens_np[i])
            num_chunks = int(chunk_num[i])
            pos = 0
            for j in range(num_chunks):
                if j == num_chunks - 1:
                    remainder = feat_len % chunk_size
                    clen = chunk_size if remainder == 0 else remainder
                else:
                    clen = chunk_size
                chunk = feat[:, pos : pos + clen]
                chunks.append(chunk)
                pos += clen

        max_chunk_len = int(max(chunk_lengths))
        padded_chunks = []
        for i, chunk in enumerate(chunks):
            clen = int(chunk_lengths[i])
            if clen < max_chunk_len:
                pad_width = max_chunk_len - clen
                chunk = mx.pad(chunk, [(0, 0), (0, pad_width)])
            padded_chunks.append(chunk)

        padded_feature = mx.stack(padded_chunks, axis=0)

        feature_lens_after_cnn = _get_feat_extract_output_lengths(
            mx.array(chunk_lengths)
        )
        feature_lens_after_cnn_np = np.array(feature_lens_after_cnn)
        max_len_after_cnn = int(feature_lens_after_cnn_np.max())

        x = padded_feature[:, :, :, None]
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))

        b, f, t, c = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
        x = self.conv_out(x)

        pos_emb = self.positional_embedding(x.shape[1])
        x = x + pos_emb[None, :, :]

        hidden_list = []
        for i in range(x.shape[0]):
            valid_len = int(feature_lens_after_cnn_np[i])
            hidden_list.append(x[i, :valid_len])

        hidden_states = mx.concatenate(hidden_list, axis=0)

        aftercnn_lens_np = np.array(aftercnn_lens)
        window_aftercnn = max_len_after_cnn * (
            self.n_window_infer // (self.n_window * 2)
        )

        cu_chunk_lens = [0]
        for cnn_len in aftercnn_lens_np:
            cnn_len = int(cnn_len)
            num_full_windows = cnn_len // window_aftercnn
            for _ in range(num_full_windows):
                cu_chunk_lens.append(window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens.append(remainder)

        cu_seqlens = np.cumsum(cu_chunk_lens).tolist()

        seq_len = hidden_states.shape[0]
        attention_mask = self._create_block_attention_mask(
            seq_len, cu_seqlens, hidden_states.dtype
        )
        attention_mask = attention_mask[None, None, :, :]

        hidden_states = hidden_states[None, :, :]

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=attention_mask)

        hidden_states = hidden_states[0]
        hidden_states = self.ln_post(hidden_states)
        hidden_states = nn.gelu(self.proj1(hidden_states))
        hidden_states = self.proj2(hidden_states)

        return hidden_states


class TextAttention(nn.Module):
    """Multi-headed attention for text decoder with Q/K norms."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[Union[str, mx.array]] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            queries = _rope_safe(self.rope, queries, offset)
            keys = _rope_safe(self.rope, keys, offset)
        else:
            offset = 0
            queries = self.rope(queries)
            keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        query_len = queries.shape[2]
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, query_len, -1)
        return self.o_proj(output)


class TextMLP(nn.Module):
    """MLP for text decoder with SwiGLU activation."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    """A single transformer decoder layer."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TextAttention(config, layer_idx)
        self.mlp = TextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[Union[str, mx.array]] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TextModel(nn.Module):
    """Text decoder model (Qwen3-based)."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(hidden_states, cache[0])

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, mask=mask, cache=cache[i])

        return self.norm(hidden_states)


class Qwen3ASRModel(nn.Module):
    """Qwen3-ASR Model for speech recognition."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.text_config.vocab_size
        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextModel(config.text_config)

        if config.text_config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )

    def get_audio_features(
        self,
        input_features: mx.array,
        feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode audio features."""
        return self.audio_tower(input_features, feature_attention_mask)

    def _build_inputs_embeds(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
        *,
        audio_start: int,
        num_audio_tokens: int,
    ) -> mx.array:
        """Build input embeddings with audio features merged in."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        audio_features = audio_features.astype(inputs_embeds.dtype)
        if num_audio_tokens != audio_features.shape[0]:
            raise ValueError(
                "Qwen3-ASR audio placeholder and feature counts differ: "
                f"{num_audio_tokens} placeholders, {audio_features.shape[0]} features"
            )
        if input_ids.shape[0] != 1:
            raise ValueError("Qwen3-ASR MLX audio prefill supports one request")
        audio_end = audio_start + num_audio_tokens
        if audio_start < 0 or audio_end > input_ids.shape[1]:
            raise ValueError(
                f"Qwen3-ASR audio span [{audio_start}, {audio_end}) is out of bounds"
            )

        inputs_embeds[0, audio_start:audio_end, :] = audio_features
        return inputs_embeds

    def _forward_last_logits(
        self,
        inputs_embeds: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)[:, -1:, :]

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.as_linear(hidden_states)

        return logits

    def __call__(
        self,
        input_ids: mx.array,
        input_embeddings: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if input_embeddings is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            inputs_embeds = input_embeddings

        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.as_linear(hidden_states)

        return logits

    def make_cache(self) -> List[Any]:
        """Create KV cache for generation."""
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.config.text_config.num_hidden_layers)]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights from HuggingFace/PyTorch format to MLX format."""
        sanitized = {}
        is_formatted = not any(k.startswith("thinker.") for k in weights.keys())

        for k, v in weights.items():
            if k.startswith("thinker."):
                k = k[len("thinker.") :]

            if k == "lm_head.weight" and self.config.text_config.tie_word_embeddings:
                continue

            if (
                not is_formatted
                and "conv2d" in k
                and "weight" in k
                and len(v.shape) == 4
            ):
                v = v.transpose(0, 2, 3, 1)

            sanitized[k] = v

        return sanitized

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        """Determine which layers to quantize."""
        return not p.startswith("audio_tower")


Model = Qwen3ASRModel
ModelArgs = ModelConfig
