# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from sglang_omni.models.qwen3_asr.mlx.config import TextConfig


@dataclass
class AudioEncoderConfig:
    num_mel_bins: int = 80
    d_model: int = 1024
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    max_source_positions: int = 1500
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    encoder_layerdrop: float = 0.0
    scale_embedding: bool = False

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> AudioEncoderConfig:
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    audio_config: AudioEncoderConfig | dict[str, Any] | None = None
    text_config: TextConfig | dict[str, Any] | None = None
    model_type: str = "moss_transcribe_diarize"
    audio_token_id: int = 151671
    audio_merge_size: int = 4
    adaptor_input_dim: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.audio_config is None:
            self.audio_config = AudioEncoderConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioEncoderConfig.from_dict(self.audio_config)

        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

        if self.adaptor_input_dim is None:
            self.adaptor_input_dim = self.audio_config.d_model * self.audio_merge_size
        self.text_config.tie_word_embeddings = self.tie_word_embeddings

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelConfig:
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            }
        )


__all__ = ["AudioEncoderConfig", "ModelConfig", "TextConfig"]
