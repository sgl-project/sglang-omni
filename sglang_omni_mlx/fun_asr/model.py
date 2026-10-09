# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR-Nano in plain MLX: SANM audio encoder, audio adaptor, and the model built from a checkpoint."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from sglang_omni_mlx.checkpoint import load_weights, read_weights
from sglang_omni_mlx.text_decoder import (
    KVCache,
    TextDecoder,
    TextDecoderConfig,
    text_decoder_config,
)

LAYER_NORM_EPS = 1e-5
FLOAT16_LIMIT = 64504.0


@dataclass(frozen=True, kw_only=True)
class AudioEncoderConfig:
    input_size: int
    d_model: int
    attention_heads: int
    ffn_dim: int
    # The stem layer, then encoder_layers - 1 layers, then the timestamp layers.
    encoder_layers: int
    timestamp_layers: int
    kernel_size: int


@dataclass(frozen=True, kw_only=True)
class AudioAdaptorConfig:
    hidden_size: int
    projector_hidden_size: int
    attention_heads: int
    layers: int


def read_configs(
    config: dict[str, object],
) -> tuple[AudioEncoderConfig, AudioAdaptorConfig, TextDecoderConfig]:
    """Model configs from either Hub layout of FunAudioLLM/Fun-ASR-Nano-2512-hf.

    Revision 972ee603 has encoder_config and flat adaptor fields; later
    revisions have audio_config and adaptor_config with transformers names.
    """
    decoder = text_decoder_config(config["text_config"])
    if "encoder_config" in config:
        encoder = config["encoder_config"]
        audio = AudioEncoderConfig(
            input_size=encoder["num_mel_bins"] * encoder["num_stacked_frames"],
            d_model=encoder["d_model"],
            attention_heads=encoder["encoder_attention_heads"],
            ffn_dim=encoder["encoder_ffn_dim"],
            encoder_layers=encoder["encoder_layers"],
            timestamp_layers=encoder["num_timestamp_prediction_blocks"],
            kernel_size=encoder["kernel_size"],
        )
        adaptor = AudioAdaptorConfig(
            hidden_size=decoder.hidden_size,
            projector_hidden_size=config["adaptor_intermediate_size"],
            attention_heads=config["adaptor_num_attention_heads"],
            layers=config["adaptor_num_hidden_layers"],
        )
    else:
        encoder = config["audio_config"]
        timestamp_layers = encoder["num_timestamp_prediction_layers"]
        audio = AudioEncoderConfig(
            input_size=encoder["num_mel_bins"] * encoder["num_stacked_frames"],
            d_model=encoder["hidden_size"],
            attention_heads=encoder["num_attention_heads"],
            ffn_dim=encoder["intermediate_size"],
            encoder_layers=encoder["num_hidden_layers"] - timestamp_layers,
            timestamp_layers=timestamp_layers,
            kernel_size=encoder["fsmn_kernel_size"],
        )
        projector = config["adaptor_config"]
        adaptor = AudioAdaptorConfig(
            hidden_size=projector["hidden_size"],
            projector_hidden_size=projector["projector_hidden_size"],
            attention_heads=projector["num_attention_heads"],
            layers=projector["num_hidden_layers"],
        )
    return audio, adaptor, decoder


class Attention(nn.Module):
    def __init__(self, input_size: int, size: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = size // heads
        self.q_proj = nn.Linear(input_size, size)
        self.k_proj = nn.Linear(input_size, size)
        self.v_proj = nn.Linear(input_size, size)
        self.out_proj = nn.Linear(size, size)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Attention output and the values, which SANM's memory block reuses."""
        batch, length, _ = x.shape
        shape = (batch, length, self.heads, self.head_dim)
        values = self.v_proj(x)
        out = mx.fast.scaled_dot_product_attention(
            self.q_proj(x).reshape(shape).transpose(0, 2, 1, 3),
            self.k_proj(x).reshape(shape).transpose(0, 2, 1, 3),
            values.reshape(shape).transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
        )
        return (
            self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1)),
            values,
        )


class MemoryBlock(nn.Module):
    """SANM's FSMN memory: a depthwise convolution over time, added back."""

    def __init__(self, size: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(size, size, kernel_size, groups=size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        return x + self.conv(mx.pad(x, ((0, 0), (left, right), (0, 0))))


class EncoderLayer(nn.Module):
    def __init__(self, input_size: int, config: AudioEncoderConfig) -> None:
        super().__init__()
        size = config.d_model
        self.has_residual = input_size == size
        self.self_attn = Attention(input_size, size, config.attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(input_size, eps=LAYER_NORM_EPS)
        self.final_layer_norm = nn.LayerNorm(size, eps=LAYER_NORM_EPS)
        self.fc1 = nn.Linear(size, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, size)
        self.fsmn = MemoryBlock(size, config.kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        attention, values = self.self_attn(self.self_attn_layer_norm(x))
        hidden = attention + self.fsmn(values)
        if self.has_residual:
            hidden = x + hidden
        else:
            pass
        hidden = hidden + self.fc2(nn.relu(self.fc1(self.final_layer_norm(hidden))))
        if hidden.dtype == mx.float16:
            hidden = mx.clip(hidden, -FLOAT16_LIMIT, FLOAT16_LIMIT)
        else:
            pass
        return hidden


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.stem = EncoderLayer(config.input_size, config)
        self.layers = [
            EncoderLayer(config.d_model, config)
            for _ in range(config.encoder_layers - 1)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model, eps=LAYER_NORM_EPS)
        self.timestamp_prediction_layers = [
            EncoderLayer(config.d_model, config) for _ in range(config.timestamp_layers)
        ]
        self.timestamp_prediction_layer_norm = nn.LayerNorm(
            config.d_model, eps=LAYER_NORM_EPS
        )

    def __call__(self, features: mx.array) -> mx.array:
        """[1, frames, input_size] stacked features to [1, frames, d_model]."""
        x = features * self.input_scale()
        _, frame_count, size = x.shape
        positions = mx.arange(1, frame_count + 1).astype(x.dtype)
        inverse_frequencies = mx.exp(
            mx.arange(size // 2).astype(x.dtype) * (-math.log(10000.0) / (size / 2 - 1))
        )
        phase = positions[:, None] * inverse_frequencies[None, :]
        x = x + mx.concatenate([mx.sin(phase), mx.cos(phase)], axis=-1)[None]
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        for layer in self.timestamp_prediction_layers:
            x = layer(x)
        return self.timestamp_prediction_layer_norm(x)

    def input_scale(self) -> float:
        return self.d_model**0.5


class AdaptorLayer(nn.Module):
    def __init__(self, size: int, heads: int) -> None:
        super().__init__()
        self.self_attn = Attention(size, size, heads)
        self.self_attn_layer_norm = nn.LayerNorm(size, eps=LAYER_NORM_EPS)
        self.final_layer_norm = nn.LayerNorm(size, eps=LAYER_NORM_EPS)
        self.fc1 = nn.Linear(size, size // 4)
        self.fc2 = nn.Linear(size // 4, size)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.self_attn_layer_norm(x))[0]
        return x + self.fc2(nn.relu(self.fc1(self.final_layer_norm(x))))


class AudioAdaptor(nn.Module):
    def __init__(self, encoder_size: int, config: AudioAdaptorConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(encoder_size, config.projector_hidden_size)
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.hidden_size)
        self.blocks = [
            AdaptorLayer(config.hidden_size, config.attention_heads)
            for _ in range(config.layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_2(nn.relu(self.linear_1(x)))
        for layer in self.blocks:
            x = layer(x)
        return x


class FunASR(nn.Module):
    def __init__(
        self,
        audio_config: AudioEncoderConfig,
        adaptor_config: AudioAdaptorConfig,
        text_config: TextDecoderConfig,
    ) -> None:
        super().__init__()
        self.audio_tower = AudioEncoder(audio_config)
        self.multi_modal_projector = AudioAdaptor(audio_config.d_model, adaptor_config)
        self.model = TextDecoder(text_config)

    def encode_audio(self, features: mx.array, audio_token_count: int) -> mx.array:
        """[stacked_frames, input_size] features to [audio_token_count, hidden_size] embeddings.

        The adaptor keeps the frame rate; the decoder sees only the first
        audio_token_count rows, as the reference model does.
        """
        x = features[None].astype(self.audio_tower.stem.fc1.weight.dtype)
        embeddings = self.multi_modal_projector(self.audio_tower(x))[0]
        return embeddings[:audio_token_count]

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]


AUDIO_SUBLAYER_RENAMES = (
    (".input_layernorm.", ".self_attn_layer_norm."),
    (".post_attention_layernorm.", ".final_layer_norm."),
    (".mlp.", "."),
    (".self_attn.o_proj.", ".self_attn.out_proj."),
    (".self_attn.fsmn.", ".fsmn."),
)
FLAT_AUDIO_LAYER = re.compile(r"^audio_tower\.layers\.(\d+)\.(.*)$")


def flat_audio_layer_name(
    index: int, rest: str, encoder_layers: int, timestamp_layers: int
) -> str:
    """Later revisions number the stem, encoder and timestamp layers as one list."""
    if index == 0:
        return f"audio_tower.stem.{rest}"
    elif rest.startswith("final_layernorm."):
        suffix = rest.removeprefix("final_layernorm.")
        if index == encoder_layers - 1:
            return f"audio_tower.layer_norm.{suffix}"
        elif index == encoder_layers + timestamp_layers - 1:
            return f"audio_tower.timestamp_prediction_layer_norm.{suffix}"
        else:
            raise ValueError(f"Unexpected final_layernorm on audio layer {index}")
    elif index < encoder_layers:
        return f"audio_tower.layers.{index - 1}.{rest}"
    else:
        return (
            f"audio_tower.timestamp_prediction_layers.{index - encoder_layers}.{rest}"
        )


def checkpoint_weights(
    weights: dict[str, mx.array], audio_config: AudioEncoderConfig
) -> dict[str, mx.array]:
    """Checkpoint names and layouts to this module tree's."""
    renamed: dict[str, mx.array] = {}
    flat_audio_layers = any(
        name.startswith("model.audio_tower.layers.") and ".input_layernorm." in name
        for name in weights
    )
    for name, value in weights.items():
        if "rotary_emb.inv_freq" in name or name == "lm_head.weight":
            continue
        else:
            pass
        name = name.removeprefix("model.")
        name = name.replace("language_model.", "model.", 1)
        if name.startswith(("audio_tower.", "multi_modal_projector.")):
            name = name.replace(
                "multi_modal_projector.layers.", "multi_modal_projector.blocks.", 1
            )
            for old, new in AUDIO_SUBLAYER_RENAMES:
                name = name.replace(old, new)
            match = FLAT_AUDIO_LAYER.match(name) if flat_audio_layers else None
            if match is not None:
                name = flat_audio_layer_name(
                    int(match.group(1)),
                    match.group(2),
                    audio_config.encoder_layers,
                    audio_config.timestamp_layers,
                )
            else:
                pass
            if name.endswith(".conv.weight"):
                # PyTorch Conv1d is [out, in, kernel]; MLX is [out, kernel, in].
                value = value.transpose(0, 2, 1)
            else:
                pass
        else:
            pass
        renamed[name] = value
    return renamed


def load_fun_asr(model_directory: Path) -> FunASR:
    """Build the model from a Fun-ASR-Nano-2512-hf checkpoint directory, quantizing as the checkpoint was."""
    config = json.loads((model_directory / "config.json").read_text())
    if not config["text_config"].get("tie_word_embeddings", True):
        raise ValueError("Fun-ASR MLX expects tied input and output embeddings")
    else:
        pass
    audio_config, adaptor_config, text_config = read_configs(config)
    model = FunASR(audio_config, adaptor_config, text_config)
    load_weights(
        model,
        checkpoint_weights(read_weights(model_directory), audio_config),
        config.get("quantization"),
    )
    return model
