# SPDX-License-Identifier: Apache-2.0
"""MLX configuration for the official Fun-ASR HF checkpoint."""

from dataclasses import dataclass, fields

from mlx_lm.models.qwen3 import ModelArgs as TextConfig


def known_fields(cls, values):
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in values.items() if k in names}


@dataclass
class AudioConfig:
    num_mel_bins: int
    num_stacked_frames: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_hidden_layers: int
    num_timestamp_prediction_layers: int
    fsmn_kernel_size: int
    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5

    @property
    def input_size(self):
        return self.num_mel_bins * self.num_stacked_frames

    @property
    def num_transcription_layers(self):
        return self.num_hidden_layers - self.num_timestamp_prediction_layers


@dataclass
class AdaptorConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    projector_hidden_size: int
    hidden_act: str = "relu"
    projector_hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5


@dataclass
class ModelConfig:
    audio_config: AudioConfig
    adaptor_config: AdaptorConfig
    text_config: TextConfig
    model_type: str = "fun_asr_nano"
    audio_token_id: int = 151646

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        config["audio_config"] = AudioConfig(
            **known_fields(AudioConfig, config["audio_config"])
        )
        config["adaptor_config"] = AdaptorConfig(
            **known_fields(AdaptorConfig, config["adaptor_config"])
        )
        text = dict(config["text_config"])
        rope = text.get("rope_parameters") or {}
        text.setdefault("rope_theta", rope.get("rope_theta", 1000000.0))
        text.setdefault("tie_word_embeddings", config.get("tie_word_embeddings", True))
        config["text_config"] = TextConfig.from_dict(text)
        return cls(**known_fields(cls, config))
