# SPDX-License-Identifier: MIT AND Apache-2.0
# Text-decoder structure adapted from the Qwen3-ASR MLX implementation, which
# is derived from mlx-audio (Copyright 2025 Prince Canuma and contributors).
# Audio-tower structure adapted from sglang_omni.models.arkasr.audio_tower.
"""Native MLX ARK-ASR audio tower, adapter, and Qwen2 decoder."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

from .config import AudioEncoderConfig, ModelConfig, TextConfig


def rope_safe(rope, x: mx.array, offset: int) -> mx.array:
    """Pad batched single-token decode to avoid an mx.fast.rope kernel bug."""
    if x.ndim == 4 and x.shape[0] > 1 and x.shape[2] == 1:
        x = mx.concatenate([x, mx.zeros_like(x)], axis=2)
        return rope(x, offset=offset)[:, :, :1, :]
    return rope(x, offset=offset)


class ArkRotaryEmbedding(nn.Module):
    """RoPE cache matching the Torch audio tower."""

    def __init__(self, dim: int, rope_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.rope_ratio = rope_ratio

    def get_emb(self, seq_len: int, dtype: mx.Dtype, base: int = 10000) -> mx.array:
        base = base * self.rope_ratio
        inv_freq = 1.0 / (
            base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)  # [seq_len, dim/2]
        emb = mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)
        return emb.astype(dtype)  # [seq_len, dim/2, 2], interleaved cos/sin


def apply_rotary_pos_emb(x: mx.array, rope_cache: mx.array) -> mx.array:
    """Apply an interleaved rotary cache to audio attention states."""
    b, nh, sq, _ = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_shaped = x_rot.reshape(b, nh, sq, rot_dim // 2, 2)
    cos = rope_cache[..., 0].reshape(1, 1, sq, rot_dim // 2)
    sin = rope_cache[..., 1].reshape(1, 1, sq, rot_dim // 2)
    x_out = mx.stack(
        [
            x_shaped[..., 0] * cos - x_shaped[..., 1] * sin,
            x_shaped[..., 1] * cos + x_shaped[..., 0] * sin,
        ],
        axis=-1,
    )
    x_out = x_out.reshape(b, nh, sq, rot_dim)
    return mx.concatenate([x_out, x_pass], axis=-1)


class WhisperRoPESdpaAttention(nn.Module):
    """Whisper self-attention with RoPE (checkpoint bias layout: no k bias)."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        rotary_pos_emb: mx.array | None = None,
    ) -> mx.array:
        bsz, seq_len, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        )

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if rotary_pos_emb is not None:
            queries = apply_rotary_pos_emb(queries, rotary_pos_emb)
            keys = apply_rotary_pos_emb(keys, rotary_pos_emb)

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.head_dim**-0.5, mask=attention_mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class WhisperSpecialEncoderLayer(nn.Module):
    """Whisper encoder layer with RoPE self-attention and GELU FFN."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperRoPESdpaAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        rotary_pos_emb: mx.array | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(nn.gelu(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        # Match the Torch fp16 overflow guard; bf16 does not need clamping.
        if hidden_states.dtype == mx.float16:
            clamp_value = 65504 - 1000
            hidden_states = mx.clip(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class ArkAudioTower(nn.Module):
    """Conv1d mel frontend + RoPE Whisper encoder layers."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        wc = config.audio_config
        self.use_rope = config.use_rope
        self.conv1 = nn.Conv1d(wc.num_mel_bins, wc.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            wc.d_model, wc.d_model, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = nn.Embedding(wc.max_source_positions, wc.d_model)
        self.layers = [WhisperSpecialEncoderLayer(wc) for _ in range(wc.encoder_layers)]
        if self.use_rope:
            self.rotary_embedding = ArkRotaryEmbedding(wc.head_dim // 2)

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        input_frame_mask = None
        if attention_mask is not None:
            expected_shape = (input_features.shape[0], input_features.shape[-1])
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    "ARK-ASR attention_mask shape "
                    f"{tuple(attention_mask.shape)} does not match mel frames "
                    f"{expected_shape}"
                )
            input_frame_mask = attention_mask

        # MLX Conv1d expects channels last.
        hidden_states = input_features.transpose(0, 2, 1)
        if input_frame_mask is not None:
            hidden_states = hidden_states * input_frame_mask[..., None]

        hidden_states = nn.gelu(self.conv1(hidden_states))
        if input_frame_mask is not None:
            hidden_states = hidden_states * input_frame_mask[..., None]
        hidden_states = nn.gelu(self.conv2(hidden_states))

        frame_mask = None
        sdpa_mask = None
        if input_frame_mask is not None:
            frame_mask = input_frame_mask[:, ::2]
            sdpa_mask = mx.where(frame_mask[:, None, None, :], 0.0, -1e9).astype(
                hidden_states.dtype
            )
            hidden_states = hidden_states * frame_mask[..., None]

        if self.use_rope:
            rotary_pos_emb = self.rotary_embedding.get_emb(
                hidden_states.shape[1], hidden_states.dtype
            )[None]
        else:
            rotary_pos_emb = None
            hidden_states = (
                hidden_states + self.embed_positions.weight[: hidden_states.shape[1]]
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=sdpa_mask,
                rotary_pos_emb=rotary_pos_emb,
            )
            if frame_mask is not None:
                hidden_states = hidden_states * frame_mask[..., None]

        # The checkpoint applies its effective LayerNorm in the adapter.
        return hidden_states


class _Gelu(nn.Module):
    """Preserve the checkpoint's adapting layer indices."""

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(x)


class ArkAudioMLPAdapter(nn.Module):
    """Map merged Whisper frames into the language model hidden size."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        wc = config.audio_config
        self.merge_factor = config.merge_factor
        self.whisper = ArkAudioTower(config)
        self.layer_norm = nn.LayerNorm(wc.d_model)
        input_dim = wc.d_model * self.merge_factor
        output_dim = config.text_config.hidden_size
        if config.mlp_adapter_act != "gelu":
            raise ValueError(
                f"ARK-ASR MLX adapter supports gelu only, got {config.mlp_adapter_act}"
            )
        self.adapting = [
            nn.Linear(input_dim, output_dim * 2),
            _Gelu(),
            nn.Linear(output_dim * 2, output_dim),
        ]

    def __call__(
        self,
        audios: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        encoded = self.whisper(audios, attention_mask=attention_mask)
        encoded = self.layer_norm(encoded)
        if attention_mask is not None:
            frame_mask = attention_mask[:, ::2]
            encoded = encoded * frame_mask[..., None]

        bsz = encoded.shape[0]
        seq_len = encoded.shape[1]
        merge = self.merge_factor
        if seq_len % merge != 0:
            target_len = (seq_len // merge) * merge
            if target_len <= 0:
                target_len = merge
                if seq_len < target_len:
                    pad = mx.zeros(
                        (bsz, target_len - seq_len, encoded.shape[-1]),
                        dtype=encoded.dtype,
                    )
                    encoded = mx.concatenate([encoded, pad], axis=1)
            else:
                encoded = encoded[:, :target_len]

        encoded = encoded.reshape(bsz, -1, encoded.shape[-1] * merge)
        encoded = self.adapting[0](encoded)
        encoded = self.adapting[1](encoded)
        return self.adapting[2](encoded)


class TextAttention(nn.Module):
    """Multi-headed self-attention for the Qwen2 text decoder."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: str | mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            queries = rope_safe(self.rope, queries, offset)
            keys = rope_safe(self.rope, keys, offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TextMLP(nn.Module):
    """MLP for the Qwen2 text decoder with SwiGLU activation."""

    def __init__(self, config: TextConfig):
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

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TextDecoderLayer(nn.Module):
    """A single transformer decoder layer."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TextAttention(config)
        self.mlp = TextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: str | mx.array | None = None,
        cache: KVCache | None = None,
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
    """Qwen2 decoder trunk without the language model head."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: list[KVCache] | None = None,
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


class ArkasrModel(nn.Module):
    """ARK-ASR model: Whisper-RoPE audio tower + MLP adapter + Qwen2 LM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        tc = config.text_config
        self.vocab_size = tc.vocab_size
        self.model = TextModel(tc)
        self.audio_encoder = ArkAudioMLPAdapter(config)

        if tc.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.model.embed_tokens

    def apply_lm_head(self, hidden_states: mx.array) -> mx.array:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return self.model.embed_tokens.as_linear(hidden_states)

    def get_audio_features(
        self,
        input_features: mx.array,
        feature_attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Encode mel features to (num_audio_tokens, hidden) for one request."""
        features = self.audio_encoder(
            input_features, attention_mask=feature_attention_mask
        )
        if features.shape[0] != 1:
            raise ValueError("ARK-ASR MLX audio prefill supports one request")
        return features[0]

    def build_inputs_embeds(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
        *,
        audio_start: int,
        num_audio_tokens: int,
    ) -> mx.array:
        """Build input embeddings with audio features merged in."""
        inputs_embeds = self.embed_tokens(input_ids)
        audio_features = audio_features.astype(inputs_embeds.dtype)
        if num_audio_tokens != audio_features.shape[0]:
            raise ValueError(
                "ARK-ASR audio placeholder and feature counts differ: "
                f"{num_audio_tokens} placeholders, {audio_features.shape[0]} features"
            )
        if input_ids.shape[0] != 1:
            raise ValueError("ARK-ASR MLX audio prefill supports one request")
        audio_end = audio_start + num_audio_tokens
        if audio_start < 0 or audio_end > input_ids.shape[1]:
            raise ValueError(
                f"ARK-ASR audio span [{audio_start}, {audio_end}) is out of bounds"
            )

        inputs_embeds[0, audio_start:audio_end, :] = audio_features
        return inputs_embeds

    def forward_last_logits(
        self,
        inputs_embeds: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)[:, -1:, :]
        return self.apply_lm_head(hidden_states)

    def __call__(
        self,
        input_ids: mx.array,
        input_embeddings: mx.array | None = None,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        if input_embeddings is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            inputs_embeds = input_embeddings

        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)
        return self.apply_lm_head(hidden_states)

    def make_cache(self) -> list[KVCache]:
        """Create KV cache for generation."""
        return [KVCache() for _ in range(self.config.text_config.num_hidden_layers)]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize checkpoint weights for MLX."""
        sanitized = {}
        for k, v in weights.items():
            if k == "lm_head.weight" and self.config.text_config.tie_word_embeddings:
                continue
            if (
                k.startswith("audio_encoder.whisper.conv")
                and k.endswith(".weight")
                and len(v.shape) == 3
            ):
                v = v.transpose(0, 2, 1)
            sanitized[k] = v
        return sanitized

    def quant_predicate(self, p: str, m: nn.Module) -> bool:
        """Quantize the text stack only; the audio tower stays full precision."""
        return not p.startswith("audio_encoder")


Model = ArkasrModel
ModelArgs = ModelConfig
