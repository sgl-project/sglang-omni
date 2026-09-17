# SPDX-License-Identifier: Apache-2.0
"""MLX equivalent of Fun-ASR's Torch SANM encoder and audio adaptor.

Consumes the same HF checkpoint and CPU-extracted features as the Torch path.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import Qwen3Model

from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)

from .config import ModelConfig


def _activation(name):
    if name == "relu":
        return nn.relu
    if name == "gelu":
        return nn.gelu
    if name == "silu":
        return nn.silu
    raise ValueError(f"Unsupported Fun-ASR activation: {name}")


class Attention(nn.Module):
    def __init__(self, in_size, size, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = size // heads
        self.q_proj = nn.Linear(in_size, size)
        self.k_proj = nn.Linear(in_size, size)
        self.v_proj = nn.Linear(in_size, size)
        self.out_proj = nn.Linear(size, size)

    def __call__(self, x):
        batch, length, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        shape = (batch, length, self.heads, self.head_dim)
        out = mx.fast.scaled_dot_product_attention(
            q.reshape(shape).transpose(0, 2, 1, 3),
            k.reshape(shape).transpose(0, 2, 1, 3),
            v.reshape(shape).transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
        )
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1)), v


class FSMN(nn.Module):
    def __init__(self, size, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(size, size, kernel_size, groups=size, bias=False)

    def __call__(self, x):
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        return x + self.conv(mx.pad(x, ((0, 0), (left, right), (0, 0))))


class EncoderLayer(nn.Module):
    def __init__(self, in_size, config):
        super().__init__()
        size = config.d_model
        self.in_size, self.size = in_size, size
        self.self_attn = Attention(in_size, size, config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(in_size, eps=1e-5)
        self.final_layer_norm = nn.LayerNorm(size, eps=1e-5)
        self.fc1 = nn.Linear(size, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, size)
        self.fsmn = FSMN(size, config.kernel_size)
        self.activation = _activation(config.activation_function)

    def __call__(self, x):
        residual = x
        attn, values = self.self_attn(self.self_attn_layer_norm(x))
        x = attn + self.fsmn(values)
        if self.in_size == self.size:
            x = residual + x
        x = x + self.fc2(self.activation(self.fc1(self.final_layer_norm(x))))
        if x.dtype == mx.float16:
            x = mx.clip(x, -64504, 64504)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_size = config.d_model
        self.stem = EncoderLayer(config.input_size, config)
        self.layers = [
            EncoderLayer(config.d_model, config)
            for _ in range(config.encoder_layers - 1)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.timestamp_prediction_layers = [
            EncoderLayer(config.d_model, config)
            for _ in range(config.num_timestamp_prediction_blocks)
        ]
        self.timestamp_prediction_layer_norm = nn.LayerNorm(config.d_model, eps=1e-5)

    def __call__(self, x):
        x = x * self.output_size**0.5
        _, timesteps, dim = x.shape
        positions = mx.arange(1, timesteps + 1).astype(x.dtype)
        inv = mx.exp(
            mx.arange(dim // 2).astype(x.dtype) * (-math.log(10000.0) / (dim / 2 - 1))
        )
        phase = positions[:, None] * inv[None, :]
        x = x + mx.concatenate([mx.sin(phase), mx.cos(phase)], axis=-1)[None, :, :]
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        for layer in self.timestamp_prediction_layers:
            x = layer(x)
        return self.timestamp_prediction_layer_norm(x)


class AdaptorLayer(nn.Module):
    def __init__(self, size, heads, activation):
        super().__init__()
        self.self_attn = Attention(size, size, heads)
        self.self_attn_layer_norm = nn.LayerNorm(size, eps=1e-5)
        self.final_layer_norm = nn.LayerNorm(size, eps=1e-5)
        self.fc1 = nn.Linear(size, size // 4)
        self.fc2 = nn.Linear(size // 4, size)
        self.activation = _activation(activation)

    def __call__(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))[0]
        return x + self.fc2(self.activation(self.fc1(self.final_layer_norm(x))))


class AudioAdaptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        size = config.text_config.hidden_size
        self.linear_1 = nn.Linear(
            config.encoder_config.d_model, config.adaptor_intermediate_size
        )
        self.linear_2 = nn.Linear(config.adaptor_intermediate_size, size)
        self.activation = _activation(config.activation_function)
        self.blocks = [
            AdaptorLayer(
                size, config.adaptor_num_attention_heads, config.activation_function
            )
            for _ in range(config.adaptor_num_hidden_layers)
        ]

    def __call__(self, x):
        x = self.linear_2(self.activation(self.linear_1(x)))
        for layer in self.blocks:
            x = layer(x)
        return x


class FunASRModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.audio_tower = AudioEncoder(config.encoder_config)
        self.multi_modal_projector = AudioAdaptor(config)
        self.model = Qwen3Model(config.text_config)
        self.lm_head = (
            None
            if config.text_config.tie_word_embeddings
            else nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        )

    @property
    def layers(self):
        return self.model.layers

    def get_audio_features(self, features, mask=None):
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError(
                "Fun-ASR MLX expects one audio item shaped [1, features, time]"
            )
        valid = int(mx.sum(mask).item()) if mask is not None else features.shape[-1]
        if valid < 1 or valid > features.shape[-1]:
            raise ValueError("Fun-ASR MLX audio feature length is out of bounds")
        x = (
            features[:, :, :valid]
            .transpose(0, 2, 1)
            .astype(self.audio_tower.stem.fc1.weight.dtype)
        )
        x = self.multi_modal_projector(self.audio_tower(x))
        return x[0, : fun_asr_low_frame_rate_length(valid)]

    def _build_inputs_embeds(
        self, input_ids, audio_features, *, audio_start, num_audio_tokens
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("Fun-ASR MLX prefill requires one request")
        if audio_features.shape != (
            num_audio_tokens,
            self.config.text_config.hidden_size,
        ):
            raise ValueError(
                "Fun-ASR MLX audio embeddings do not match the placeholder span"
            )
        end = audio_start + num_audio_tokens
        if audio_start < 0 or end > input_ids.shape[1]:
            raise ValueError("Fun-ASR MLX audio placeholder span is out of bounds")
        embeds = self.model.embed_tokens(input_ids)
        embeds[0, audio_start:end] = audio_features.astype(embeds.dtype)
        return embeds

    def _project(self, hidden):
        return (
            self.lm_head(hidden)
            if self.lm_head is not None
            else self.model.embed_tokens.as_linear(hidden)
        )

    def _forward_last_logits(self, inputs_embeds, cache=None):
        hidden = self.model(None, cache=cache, input_embeddings=inputs_embeds)
        return self._project(hidden[:, -1:, :])

    def __call__(self, input_ids, cache=None):
        return self._project(self.model(input_ids, cache=cache))

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights):
        result = {}
        for key, value in weights.items():
            is_torch_audio = key.startswith(
                (
                    "model.audio_tower.",
                    "model.audio_adaptor.",
                    "model.multi_modal_projector.",
                )
            )
            key = key.replace("model.audio_adaptor.", "multi_modal_projector.", 1)
            key = key.replace(
                "model.multi_modal_projector.", "multi_modal_projector.", 1
            )
            key = key.replace("model.audio_tower.", "audio_tower.", 1)
            key = key.replace("model.language_model.", "model.", 1)
            key = key.replace(".feedforward_sequential_memory.", ".fsmn.")
            if key == "lm_head.weight" and self.lm_head is None:
                continue
            if "rotary_emb.inv_freq" in key:
                continue
            if is_torch_audio and key.endswith(".conv.weight"):
                value = value.transpose(0, 2, 1)
            result[key] = value
        return result
