from __future__ import annotations

from sglang_omni.models.dots_tts.components.config_base import (
    ConfigBase,
    StrictConfigBase,
)
from sglang_omni.models.dots_tts.components.vocoder.config import AudioVAEConfig


class _EncoderConfig(ConfigBase):
    num_layers: int = 6
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    modulation: bool = False
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_dropout: float = 0.0
    dropout: float = 0.0
    norm_layer: str = "LayerNorm"
    alibi_bias: bool = False
    rotary_bias: bool = False
    rotary_theta: float | None = 10000
    input_dim: int = 1024
    causal: bool = True


class _DiTConfig(ConfigBase):
    num_layers: int = 18
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    modulation: bool = True
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_dropout: float = 0.0
    dropout: float = 0.0
    norm_layer: str = "LayerNorm"
    alibi_bias: bool = False
    rotary_bias: bool = True
    rotary_theta: float | None = 10000


class LossConfig(StrictConfigBase):
    ce_weight: float = 1.0
    fm_weight: float = 1.0
    eos_weight: float = 1.0


class MeanFlowConfig(ConfigBase):
    enabled: bool = False
    use_duration_embedding: bool = True


class ModelConfig(ConfigBase):
    model_type: str = "dots_tts"
    latent_dim: int
    patch_size: int
    cfg_droprate: float = 0.2
    PatchEncoder: _EncoderConfig
    DiT: _DiTConfig
    vocoder: AudioVAEConfig
    fm_sigma: float = 0.0
    xvec_drop_rate: float = 0.2
    campplus_embedding_size: int | None = 512
    xvec_max_audio_seconds: float = 10.0
    meanflow: MeanFlowConfig | None = None


__all__ = [
    "LossConfig",
    "MeanFlowConfig",
    "ModelConfig",
]
