# SPDX-License-Identifier: MIT
# Adapted from the Qwen3-ASR MLX configuration, which is derived from mlx-audio
# (Copyright 2025 Prince Canuma and contributors).
"""MLX configuration for ARK-ASR."""

from __future__ import annotations

import inspect
from dataclasses import dataclass


@dataclass
class AudioEncoderConfig:
    """ARK-ASR audio encoder checkpoint configuration."""

    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    activation_function: str = "gelu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    scale_embedding: bool = False

    @classmethod
    def from_dict(cls, params: dict[str, object]) -> AudioEncoderConfig:
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads


@dataclass
class TextConfig:
    """ARK-ASR Qwen2 text decoder configuration."""

    model_type: str = "qwen2"
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, object] | None = None
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    use_cache: bool = True

    @classmethod
    def from_dict(cls, params: dict[str, object]) -> TextConfig:
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class ModelConfig:
    """Configuration for the ARK-ASR model."""

    audio_config: AudioEncoderConfig | dict[str, object] | None = None
    text_config: TextConfig | dict[str, object] | None = None
    model_type: str = "arkasr"
    audio_token_id: int = 151663
    adapter_type: str = "mlp"
    merge_factor: int = 4
    mlp_adapter_act: str = "gelu"
    use_rope: bool = True
    max_whisper_length: int = 1500
    spec_aug: bool = False

    def __post_init__(self):
        if self.audio_config is None:
            self.audio_config = AudioEncoderConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioEncoderConfig.from_dict(self.audio_config)

        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

    @classmethod
    def from_dict(cls, params: dict[str, object]) -> ModelConfig:
        params = params.copy()

        # ARK nests the audio encoder config under whisper_config and keeps
        # the Qwen2 LM parameters at the top level.
        if "whisper_config" in params:
            params["audio_config"] = params.pop("whisper_config")

        if "audio_config" in params and isinstance(params["audio_config"], dict):
            params["audio_config"] = AudioEncoderConfig.from_dict(
                params["audio_config"]
            )
        elif "audio_config" not in params:
            params["audio_config"] = AudioEncoderConfig()

        text_keys = set(inspect.signature(TextConfig).parameters) - {"model_type"}
        text_params = {k: params.pop(k) for k in list(params) if k in text_keys}
        if "text_config" not in params:
            params["text_config"] = TextConfig.from_dict(text_params)
        elif isinstance(params["text_config"], dict):
            params["text_config"] = TextConfig.from_dict(params["text_config"])

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
