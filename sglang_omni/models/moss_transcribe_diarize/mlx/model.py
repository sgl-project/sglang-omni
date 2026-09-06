# SPDX-License-Identifier: MIT
# The Whisper encoder is derived from mlx-audio (Copyright 2023 Apple Inc.).

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sglang_omni.models.qwen3_asr.mlx.model import TextModel

from .config import AudioEncoderConfig, ModelConfig


class WhisperAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        queries = queries.reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(batch, length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
        )
        return self.out_proj(
            output.transpose(0, 2, 1, 3).reshape(batch, length, self.embed_dim)
        )


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        self.self_attn = WhisperAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.self_attn_layer_norm(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.fc2(
            nn.gelu(self.fc1(self.final_layer_norm(hidden_states)))
        )
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins, config.d_model, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, input_features: mx.array) -> mx.array:
        input_features = input_features.astype(self.conv1.weight.dtype)
        hidden_states = input_features.transpose(0, 2, 1)
        hidden_states = nn.gelu(self.conv1(hidden_states))
        hidden_states = nn.gelu(self.conv2(hidden_states))
        if hidden_states.shape[1] > self.config.max_source_positions:
            raise ValueError(
                "Whisper encoder input exceeds max_source_positions: "
                f"{hidden_states.shape[1]} > {self.config.max_source_positions}"
            )
        hidden_states = (
            hidden_states + self.embed_positions.weight[: hidden_states.shape[1]]
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


class VQAdaptor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, norm_eps: float):
        self.linear1 = nn.Linear(input_dim, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=norm_eps)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear1(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return self.layer_norm(hidden_states)


class MossTranscribeDiarizeModel(nn.Module):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.vocab_size = config.text_config.vocab_size
        self.whisper_encoder = WhisperEncoder(config.audio_config)
        self.vq_adaptor = VQAdaptor(
            int(config.adaptor_input_dim),
            config.text_config.hidden_size,
            config.text_config.rms_norm_eps,
        )
        self.model = TextModel(config.text_config)
        self.lm_head = (
            None
            if config.text_config.tie_word_embeddings
            else nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        )

    def get_audio_features(
        self,
        input_features: mx.array,
        audio_feature_lengths: mx.array,
        audio_chunk_mapping: mx.array | None = None,
    ) -> list[mx.array]:
        if input_features is None or audio_feature_lengths is None:
            raise ValueError("MOSS-TD requires input features and feature lengths")
        if audio_feature_lengths.size != input_features.shape[0]:
            raise ValueError("MOSS-TD requires one feature length per audio chunk")

        encoded = self.whisper_encoder(input_features)
        lengths = np.asarray(audio_feature_lengths).astype(np.int64).tolist()
        if audio_chunk_mapping is None:
            mapping = [0] * len(lengths)
        else:
            mapping = np.asarray(audio_chunk_mapping).astype(np.int64).tolist()
        if len(mapping) != len(lengths):
            raise ValueError("MOSS-TD requires one audio mapping per audio chunk")

        merge_size = int(self.config.audio_merge_size)
        num_audios = max(mapping) + 1 if mapping else 0
        parts: list[list[mx.array]] = [[] for _ in range(num_audios)]
        for chunk_index, (token_length, audio_index) in enumerate(
            zip(lengths, mapping)
        ):
            parts[audio_index].append(
                encoded[chunk_index : chunk_index + 1, : token_length * merge_size]
            )

        outputs: list[mx.array] = []
        for audio_parts in parts:
            joined = mx.concatenate(audio_parts, axis=1)
            trimmed_length = joined.shape[1] // merge_size * merge_size
            merged = joined[:, :trimmed_length].reshape(
                1,
                trimmed_length // merge_size,
                joined.shape[2] * merge_size,
            )
            outputs.append(self.vq_adaptor(merged)[0])
        return outputs

    def _build_inputs_embeds(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
        *,
        audio_positions: list[int],
    ) -> mx.array:
        if input_ids.shape[0] != 1:
            raise ValueError("MOSS-TD MLX audio prefill supports one request")
        if len(audio_positions) != audio_features.shape[0]:
            raise ValueError(
                "MOSS-TD audio placeholder and feature counts differ: "
                f"{len(audio_positions)} placeholders, "
                f"{audio_features.shape[0]} features"
            )
        inputs_embeds = self.model.embed_tokens(input_ids)
        inputs_embeds[0, mx.array(audio_positions), :] = audio_features.astype(
            inputs_embeds.dtype
        )
        return inputs_embeds

    def _project(self, hidden_states: mx.array) -> mx.array:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return self.model.embed_tokens.as_linear(hidden_states)

    def _forward_last_logits(
        self,
        inputs_embeds: mx.array,
        cache: list[Any] | None = None,
    ) -> mx.array:
        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)
        return self._project(hidden_states[:, -1:, :])

    def __call__(
        self,
        input_ids: mx.array,
        input_embeddings: mx.array | None = None,
        cache: list[Any] | None = None,
    ) -> mx.array:
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=input_embeddings,
            cache=cache,
        )
        return self._project(hidden_states)

    def make_cache(self) -> list[Any]:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.config.text_config.num_hidden_layers)]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized: dict[str, mx.array] = {}
        is_hf = any(key.startswith("model.") for key in weights)
        adaptor_names = {
            "vq_adaptor.layers.0.": "vq_adaptor.linear1.",
            "vq_adaptor.layers.2.": "vq_adaptor.linear2.",
            "vq_adaptor.layers.3.": "vq_adaptor.layer_norm.",
        }
        for name, value in weights.items():
            if name == "lm_head.weight" and self.config.text_config.tie_word_embeddings:
                continue
            if name.startswith("model.language_model."):
                name = "model." + name.removeprefix("model.language_model.")
            elif name.startswith("model.whisper_encoder."):
                name = "whisper_encoder." + name.removeprefix("model.whisper_encoder.")
            elif name.startswith("model.vq_adaptor."):
                name = "vq_adaptor." + name.removeprefix("model.vq_adaptor.")
            for old, new in adaptor_names.items():
                if old in name:
                    name = name.replace(old, new)
                    break
            if is_hf and name in {
                "whisper_encoder.conv1.weight",
                "whisper_encoder.conv2.weight",
            }:
                value = value.transpose(0, 2, 1)
            sanitized[name] = value
        return sanitized

    def model_quant_predicate(self, path: str, module: nn.Module) -> bool:
        return not path.startswith(("whisper_encoder", "vq_adaptor"))


Model = MossTranscribeDiarizeModel
ModelArgs = ModelConfig

__all__ = ["MossTranscribeDiarizeModel", "Model", "ModelArgs"]
