# SPDX-License-Identifier: MIT
# The Whisper encoder is derived from mlx-audio (Copyright 2023 Apple Inc.).
"""MOSS-TD Whisper encoder and adaptor around the shared Qwen3 decoder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sglang_omni_mlx.checkpoint import load_weights, read_weights
from sglang_omni_mlx.text_decoder import (
    KVCache,
    TextDecoder,
    TextDecoderConfig,
    text_decoder_config,
)


@dataclass(frozen=True, kw_only=True)
class AudioEncoderConfig:
    num_mel_bins: int
    d_model: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    max_source_positions: int


class WhisperAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.hidden_size = config.d_model
        self.head_count = config.encoder_attention_heads
        self.head_dim = self.hidden_size // self.head_count
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, token_count, _ = hidden_states.shape
        queries, keys, values = (
            projection(hidden_states)
            .reshape(batch_size, token_count, self.head_count, self.head_dim)
            .transpose(0, 2, 1, 3)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.head_dim**-0.5
        )
        return self.out_proj(
            attended.transpose(0, 2, 1, 3).reshape(
                batch_size, token_count, self.hidden_size
            )
        )


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.self_attn = WhisperAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states)
        )
        return hidden_states + self.fc2(
            nn.gelu(self.fc1(self.final_layer_norm(hidden_states)))
        )


class WhisperEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
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
        hidden_states = input_features.astype(self.conv1.weight.dtype).transpose(
            0, 2, 1
        )
        hidden_states = nn.gelu(self.conv1(hidden_states))
        hidden_states = nn.gelu(self.conv2(hidden_states))
        assert hidden_states.shape[1] <= self.config.max_source_positions
        hidden_states = (
            hidden_states + self.embed_positions.weight[: hidden_states.shape[1]]
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


class VQAdaptor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, norm_epsilon: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=norm_epsilon)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear1(hidden_states)
        hidden_states = nn.silu(hidden_states)
        return self.layer_norm(self.linear2(hidden_states))


class MossTranscribeDiarize(nn.Module):
    ENCODER_WINDOW_BATCH_SIZE = 2

    def __init__(
        self,
        audio_config: AudioEncoderConfig,
        text_config: TextDecoderConfig,
        audio_merge_size: int,
        adaptor_input_size: int,
    ) -> None:
        super().__init__()
        self.audio_merge_size = audio_merge_size
        self.whisper_encoder = WhisperEncoder(audio_config)
        self.vq_adaptor = VQAdaptor(
            adaptor_input_size, text_config.hidden_size, text_config.rms_norm_eps
        )
        self.model = TextDecoder(text_config)

    def encode_audio(
        self, input_features: mx.array, audio_token_lengths: np.ndarray
    ) -> mx.array:
        """Encode padded Whisper windows in bounded batches, then trim and merge."""
        assert len(audio_token_lengths) == input_features.shape[0]
        audio_parts: list[mx.array] = []
        for start in range(0, input_features.shape[0], self.ENCODER_WINDOW_BATCH_SIZE):
            encoded = self.whisper_encoder(
                input_features[start : start + self.ENCODER_WINDOW_BATCH_SIZE]
            )
            for local_index in range(encoded.shape[0]):
                token_length = int(audio_token_lengths[start + local_index])
                window = encoded[
                    local_index : local_index + 1,
                    : token_length * self.audio_merge_size,
                ]
                merged = window.reshape(
                    1,
                    token_length,
                    window.shape[2] * self.audio_merge_size,
                )
                projected = self.vq_adaptor(merged)[0]
                mx.eval(projected)
                audio_parts.append(projected)
        return mx.concatenate(audio_parts, axis=0)

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]


def checkpoint_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Official checkpoint names and convolution layouts to this module tree."""
    adaptor_names = {
        "vq_adaptor.layers.0.": "vq_adaptor.linear1.",
        "vq_adaptor.layers.2.": "vq_adaptor.linear2.",
        "vq_adaptor.layers.3.": "vq_adaptor.layer_norm.",
    }
    renamed: dict[str, mx.array] = {}
    for name, value in weights.items():
        if name == "lm_head.weight" or "rotary_emb.inv_freq" in name:
            continue
        else:
            pass
        if name in {
            "model.whisper_encoder.conv1.weight",
            "model.whisper_encoder.conv2.weight",
        }:
            value = value.transpose(0, 2, 1)
        else:
            pass
        if name.startswith("model.language_model."):
            name = "model." + name.removeprefix("model.language_model.")
        elif name.startswith("model.whisper_encoder."):
            name = "whisper_encoder." + name.removeprefix("model.whisper_encoder.")
        elif name.startswith("model.vq_adaptor."):
            name = "vq_adaptor." + name.removeprefix("model.vq_adaptor.")
        else:
            pass
        for checkpoint_prefix, model_prefix in adaptor_names.items():
            name = name.replace(checkpoint_prefix, model_prefix, 1)
        renamed[name] = value
    return renamed


def load_moss_transcribe_diarize(model_directory: Path) -> MossTranscribeDiarize:
    """Build MOSS-TD from an official checkpoint directory."""
    config = json.loads((model_directory / "config.json").read_text())
    if not config["text_config"].get("tie_word_embeddings", True):
        raise ValueError("MOSS-TD MLX expects tied input and output embeddings")
    else:
        pass
    audio = config["audio_config"]
    model = MossTranscribeDiarize(
        AudioEncoderConfig(
            **{name: audio[name] for name in AudioEncoderConfig.__dataclass_fields__}
        ),
        text_decoder_config(config["text_config"]),
        audio_merge_size=config["audio_merge_size"],
        adaptor_input_size=config["adaptor_input_dim"],
    )
    load_weights(
        model,
        checkpoint_weights(read_weights(model_directory)),
        config.get("quantization"),
    )
    return model
