# SPDX-License-Identifier: Apache-2.0
"""MLX configuration for the official Fun-ASR HF checkpoint."""

from dataclasses import dataclass, fields

from mlx_lm.models.qwen3 import ModelArgs as TextConfig


@dataclass
class EncoderConfig:
    num_mel_bins: int = 80
    num_stacked_frames: int = 7
    d_model: int = 512
    encoder_attention_heads: int = 4
    encoder_ffn_dim: int = 2048
    encoder_layers: int = 50
    num_timestamp_prediction_blocks: int = 20
    kernel_size: int = 11
    activation_function: str = "relu"

    @property
    def input_size(self):
        return self.num_mel_bins * self.num_stacked_frames


@dataclass
class ModelConfig:
    encoder_config: EncoderConfig
    text_config: TextConfig
    model_type: str = "fun_asr_nano"
    audio_token_id: int = 151646
    adaptor_intermediate_size: int = 2048
    adaptor_num_hidden_layers: int = 2
    adaptor_num_attention_heads: int = 8
    activation_function: str = "relu"

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        enc = config.get("encoder_config", {})
        config["encoder_config"] = EncoderConfig(
            **{
                k: v
                for k, v in enc.items()
                if k in {f.name for f in fields(EncoderConfig)}
            }
        )
        text = dict(config["text_config"])
        rope = text.get("rope_parameters") or {}
        text.setdefault("rope_theta", rope.get("rope_theta", 1000000.0))
        text.setdefault("tie_word_embeddings", config.get("tie_word_embeddings", True))
        config["text_config"] = TextConfig.from_dict(text)
        return cls(
            **{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}}
        )
