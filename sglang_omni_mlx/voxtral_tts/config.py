# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS checkpoint configuration for the MLX runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ACOUSTIC_NORM_EPSILON = 1e-5
DEFAULT_ENCODER_CONVOLUTION_STRIDES_TEXT = "2, 2, 2, 1"


@dataclass(frozen=True, kw_only=True)
class TextConfig:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    rope_theta: float
    norm_eps: float


@dataclass(frozen=True, kw_only=True)
class AcousticConfig:
    input_dim: int
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    use_biases: bool
    norm_eps: float


@dataclass(frozen=True, kw_only=True)
class AudioModelConfig:
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer: AcousticConfig
    audio_token_id: int


@dataclass(frozen=True, kw_only=True)
class VoxtralConfig:
    text: TextConfig
    audio: AudioModelConfig
    audio_tokenizer: dict[str, object]
    voices: dict[str, int]
    checkpoint_directory: Path


@dataclass(frozen=True, kw_only=True)
class AudioTokenizerConfig:
    sampling_rate: int
    patch_size: int
    patch_projection_kernel_size: int
    semantic_codebook_size: int
    semantic_dimension: int
    acoustic_codebook_size: int
    acoustic_dimension: int
    causal: bool
    sliding_window_size: int
    half_attention_window_on_downsample: bool
    encoder_convolution_strides: tuple[int, ...]
    dimension: int
    hidden_dimension: int
    head_dimension: int
    head_count: int
    kv_head_count: int
    qk_norm_epsilon: float
    qk_norm: bool
    use_biases: bool
    norm_epsilon: float
    layer_scale: bool
    layer_scale_initial: float | None
    transformer_lengths: tuple[int, ...]
    convolution_kernels: tuple[int, ...]
    convolution_strides: tuple[int, ...]

    @classmethod
    def from_voxtral_config(cls, config: VoxtralConfig) -> AudioTokenizerConfig:
        values = config.audio_tokenizer
        transformer_lengths = tuple(
            int(value)
            for value in str(values["decoder_transformer_lengths_str"]).split(",")
        )
        convolution_kernels = tuple(
            int(value) for value in str(values["decoder_convs_kernels_str"]).split(",")
        )
        convolution_strides = tuple(
            int(value) for value in str(values["decoder_convs_strides_str"]).split(",")
        )
        encoder_convolution_strides = tuple(
            int(value)
            for value in str(
                values.get(
                    "encoder_convs_strides_str",
                    DEFAULT_ENCODER_CONVOLUTION_STRIDES_TEXT,
                )
            ).split(",")
        )
        if not (
            len(transformer_lengths)
            == len(convolution_kernels)
            == len(convolution_strides)
            == len(encoder_convolution_strides)
        ):
            raise ValueError("Voxtral audio tokenizer configuration lengths differ")
        else:
            pass
        layer_scale_initial_value = values["layer_scale_init"]
        if layer_scale_initial_value is None:
            layer_scale_initial = None
        else:
            layer_scale_initial = float(layer_scale_initial_value)
        return cls(
            sampling_rate=int(values["sampling_rate"]),
            patch_size=int(values["pretransform_patch_size"]),
            patch_projection_kernel_size=int(values["patch_proj_kernel_size"]),
            semantic_codebook_size=int(values["semantic_codebook_size"]),
            semantic_dimension=int(values["semantic_dim"]),
            acoustic_codebook_size=int(values["acoustic_codebook_size"]),
            acoustic_dimension=int(values["acoustic_dim"]),
            causal=bool(values["causal"]),
            sliding_window_size=int(values["attn_sliding_window_size"]),
            half_attention_window_on_downsample=bool(
                values["half_attn_window_upon_downsampling"]
            ),
            encoder_convolution_strides=encoder_convolution_strides,
            dimension=int(values["dim"]),
            hidden_dimension=int(values["hidden_dim"]),
            head_dimension=int(values["head_dim"]),
            head_count=int(values["n_heads"]),
            kv_head_count=int(values["n_kv_heads"]),
            qk_norm_epsilon=float(values["qk_norm_eps"]),
            qk_norm=bool(values["qk_norm"]),
            use_biases=bool(values["use_biases"]),
            norm_epsilon=float(values["norm_eps"]),
            layer_scale=bool(values["layer_scale"]),
            layer_scale_initial=layer_scale_initial,
            transformer_lengths=transformer_lengths,
            convolution_kernels=convolution_kernels,
            convolution_strides=convolution_strides,
        )


def read_voxtral_config(checkpoint_directory: Path) -> VoxtralConfig:
    params_path = checkpoint_directory / "params.json"
    params = json.loads(params_path.read_text())
    if not isinstance(params, dict):
        raise ValueError(f"Voxtral params must be a JSON object: {params_path}")
    else:
        pass
    multimodal = params["multimodal"]
    if not isinstance(multimodal, dict):
        raise ValueError("Voxtral params.multimodal must be an object")
    else:
        pass
    audio_values = multimodal["audio_model_args"]
    tokenizer_values = multimodal["audio_tokenizer_args"]
    if not isinstance(audio_values, dict) or not isinstance(tokenizer_values, dict):
        raise ValueError("Voxtral audio configuration values must be objects")
    else:
        pass
    acoustic_values = audio_values["acoustic_transformer_args"]
    if not isinstance(acoustic_values, dict):
        raise ValueError("Voxtral acoustic_transformer_args must be an object")
    else:
        pass
    tokenizer_values = tokenizer_values.copy()
    voice_values = tokenizer_values.pop("voice")
    if not isinstance(voice_values, dict):
        raise ValueError("Voxtral audio_tokenizer_args.voice must be an object")
    else:
        pass
    return VoxtralConfig(
        text=TextConfig(
            dim=int(params["dim"]),
            n_layers=int(params["n_layers"]),
            head_dim=int(params["head_dim"]),
            hidden_dim=int(params["hidden_dim"]),
            n_heads=int(params["n_heads"]),
            n_kv_heads=int(params["n_kv_heads"]),
            vocab_size=int(params["vocab_size"]),
            rope_theta=float(params["rope_theta"]),
            norm_eps=float(params["norm_eps"]),
        ),
        audio=AudioModelConfig(
            semantic_codebook_size=int(audio_values["semantic_codebook_size"]),
            acoustic_codebook_size=int(audio_values["acoustic_codebook_size"]),
            n_acoustic_codebook=int(audio_values["n_acoustic_codebook"]),
            acoustic_transformer=AcousticConfig(
                input_dim=int(acoustic_values["input_dim"]),
                dim=int(acoustic_values["dim"]),
                n_layers=int(acoustic_values["n_layers"]),
                head_dim=int(acoustic_values["head_dim"]),
                hidden_dim=int(acoustic_values["hidden_dim"]),
                n_heads=int(acoustic_values["n_heads"]),
                n_kv_heads=int(acoustic_values["n_kv_heads"]),
                use_biases=bool(acoustic_values["use_biases"]),
                norm_eps=float(
                    acoustic_values.get("norm_eps", DEFAULT_ACOUSTIC_NORM_EPSILON)
                ),
            ),
            audio_token_id=int(audio_values["audio_token_id"]),
        ),
        audio_tokenizer=dict(tokenizer_values),
        voices={str(name): int(token_id) for name, token_id in voice_values.items()},
        checkpoint_directory=checkpoint_directory,
    )
