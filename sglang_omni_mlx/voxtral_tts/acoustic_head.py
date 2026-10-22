# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS flow-matching acoustic code predictor in MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from sglang_omni_mlx.voxtral_tts.config import AcousticConfig, AudioModelConfig

ACOUSTIC_DECODE_ITERATIONS = 8
CLASSIFIER_FREE_GUIDANCE_ALPHA = 1.2
NOISE_SCALE = 1.0
SEMANTIC_VOCAB_ALIGNMENT = 128
SPECIAL_TOKEN_COUNT = 2
EMPTY_AUDIO_TOKEN_ID = 0
END_AUDIO_TOKEN_ID = 1


def padded_size(codebook_size: int, alignment: int) -> int:
    return alignment * ((codebook_size + alignment - 1) // alignment)


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def __call__(self, timesteps: mx.array) -> mx.array:
        inverse_frequencies = mx.exp(
            -mx.log(10000.0)
            * mx.arange(self.dimension // 2, dtype=mx.float32)
            / (self.dimension // 2)
        )
        angles = timesteps * inverse_frequencies[None, :]
        return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


class AcousticFeedForward(nn.Module):
    def __init__(self, config: AcousticConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=config.use_biases)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class BidirectionalAttention(nn.Module):
    def __init__(self, config: AcousticConfig) -> None:
        super().__init__()
        self.head_count = config.n_heads
        self.kv_head_count = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_repeats = self.head_count // self.kv_head_count
        self.wq = nn.Linear(
            config.dim, self.head_count * self.head_dim, bias=config.use_biases
        )
        self.wk = nn.Linear(config.dim, self.kv_head_count * self.head_dim, bias=False)
        self.wv = nn.Linear(
            config.dim, self.kv_head_count * self.head_dim, bias=config.use_biases
        )
        self.wo = nn.Linear(
            self.head_count * self.head_dim, config.dim, bias=config.use_biases
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, token_count, _ = hidden_states.shape
        queries = self.wq(hidden_states).reshape(
            batch_size, token_count, self.head_count, self.head_dim
        )
        keys = self.wk(hidden_states).reshape(
            batch_size, token_count, self.kv_head_count, self.head_dim
        )
        values = self.wv(hidden_states).reshape(
            batch_size, token_count, self.kv_head_count, self.head_dim
        )
        if self.kv_repeats > 1:
            keys = mx.repeat(keys, self.kv_repeats, axis=2)
            values = mx.repeat(values, self.kv_repeats, axis=2)
        else:
            pass
        queries = queries.transpose(0, 2, 1, 3) * (self.head_dim**-0.5)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        weights = mx.softmax(
            (queries @ keys.transpose(0, 1, 3, 2)).astype(mx.float32), axis=-1
        ).astype(values.dtype)
        attended = weights @ values
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, token_count, self.head_count * self.head_dim
        )
        return self.wo(attended)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, config: AcousticConfig) -> None:
        super().__init__()
        self.attention = BidirectionalAttention(config)
        self.feed_forward = AcousticFeedForward(config)
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class FlowMatchingAudioTransformer(nn.Module):
    def __init__(self, config: AudioModelConfig) -> None:
        super().__init__()
        acoustic_config = config.acoustic_transformer
        self.config = config
        self.acoustic_levels = config.acoustic_codebook_size
        self.acoustic_codebook_count = config.n_acoustic_codebook
        self.time_embedding = TimeEmbedding(acoustic_config.dim)
        self.input_projection = nn.Linear(
            self.acoustic_codebook_count, acoustic_config.dim, bias=False
        )
        self.time_projection = nn.Linear(
            acoustic_config.dim, acoustic_config.dim, bias=False
        )
        self.llm_projection = nn.Linear(
            acoustic_config.input_dim, acoustic_config.dim, bias=False
        )
        semantic_size = padded_size(
            config.semantic_codebook_size + SPECIAL_TOKEN_COUNT,
            SEMANTIC_VOCAB_ALIGNMENT,
        )
        self.semantic_codebook_output = nn.Linear(
            acoustic_config.dim, semantic_size, bias=acoustic_config.use_biases
        )
        self.acoustic_codebook_output = nn.Linear(
            acoustic_config.dim, self.acoustic_codebook_count, bias=False
        )
        self.layers = [
            AcousticTransformerBlock(acoustic_config)
            for _ in range(acoustic_config.n_layers)
        ]
        self.norm = nn.RMSNorm(acoustic_config.dim, eps=acoustic_config.norm_eps)

    def predict_velocity(
        self,
        acoustic_latents: mx.array,
        language_hidden_states: mx.array,
        timestep_embedding: mx.array,
    ) -> mx.array:
        acoustic_features = self.input_projection(
            acoustic_latents.astype(language_hidden_states.dtype)
        )
        time_features = self.time_projection(
            timestep_embedding.astype(language_hidden_states.dtype)
        )
        language_features = self.llm_projection(language_hidden_states)
        sequence = mx.stack(
            [acoustic_features, time_features, language_features], axis=1
        )
        for layer in self.layers:
            sequence = layer(sequence)
        return self.acoustic_codebook_output(self.norm(sequence)[:, 0, :])

    def decode_one_frame(
        self, semantic_codes: mx.array, language_hidden_states: mx.array
    ) -> mx.array:
        batch_size = semantic_codes.shape[0]
        should_decode = semantic_codes != END_AUDIO_TOKEN_ID
        sampled = NOISE_SCALE * mx.random.normal(
            (batch_size, self.acoustic_codebook_count),
            dtype=language_hidden_states.dtype,
        )
        unconditional_hidden_states = mx.zeros_like(language_hidden_states)
        timesteps = mx.linspace(
            0.0, 1.0, ACOUSTIC_DECODE_ITERATIONS, dtype=language_hidden_states.dtype
        )
        for index in range(ACOUSTIC_DECODE_ITERATIONS - 1):
            timestep = timesteps[index]
            step_size = timesteps[index + 1] - timestep
            timestep_embedding = self.time_embedding(
                mx.broadcast_to(timestep, (batch_size, 1))
            ).astype(language_hidden_states.dtype)
            batched_latents = mx.concatenate([sampled, sampled], axis=0)
            batched_hidden_states = mx.concatenate(
                [language_hidden_states, unconditional_hidden_states], axis=0
            )
            batched_timestep_embedding = mx.concatenate(
                [timestep_embedding, timestep_embedding], axis=0
            )
            velocity = self.predict_velocity(
                batched_latents, batched_hidden_states, batched_timestep_embedding
            )
            conditional_velocity = velocity[:batch_size]
            unconditional_velocity = velocity[batch_size:]
            guided_velocity = (
                CLASSIFIER_FREE_GUIDANCE_ALPHA * conditional_velocity
                + (1.0 - CLASSIFIER_FREE_GUIDANCE_ALPHA) * unconditional_velocity
            )
            sampled = sampled + guided_velocity * step_size
        sampled = mx.clip(sampled, -1.0, 1.0)
        quantized = mx.round((sampled + 1.0) * 0.5 * (self.acoustic_levels - 1))
        acoustic_codes = quantized.astype(mx.int32) + SPECIAL_TOKEN_COUNT
        return mx.where(
            should_decode[:, None],
            acoustic_codes,
            EMPTY_AUDIO_TOKEN_ID + SPECIAL_TOKEN_COUNT,
        )

    def __call__(self, language_hidden_states: mx.array) -> mx.array:
        semantic_logits = self.semantic_codebook_output(language_hidden_states).astype(
            mx.float32
        )
        semantic_logits[:, EMPTY_AUDIO_TOKEN_ID] = -float("inf")
        semantic_logits[
            :, self.config.semantic_codebook_size + SPECIAL_TOKEN_COUNT :
        ] = -float("inf")
        semantic_codes = mx.argmax(semantic_logits, axis=-1)
        acoustic_codes = self.decode_one_frame(semantic_codes, language_hidden_states)
        return mx.concatenate([semantic_codes[:, None], acoustic_codes], axis=1)
