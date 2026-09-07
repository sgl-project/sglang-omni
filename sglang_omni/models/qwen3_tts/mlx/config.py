# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-TTS (Copyright 2025 Prince Canuma and contributors).

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any


def _filtered(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    allowed = inspect.signature(cls).parameters
    return {k: v for k, v in params.items() if k in allowed}


@dataclass
class SpeakerEncoderConfig:
    """Configuration for the ECAPA-TDNN speaker encoder (Base voice cloning)."""

    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    enc_kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> SpeakerEncoderConfig:
        return cls(**_filtered(cls, params))


@dataclass
class CodePredictorConfig:
    """Configuration for the Qwen3-TTS talker code predictor (MTP) sub-model."""

    vocab_size: int = 2048
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_code_groups: int = 16

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> CodePredictorConfig:
        return cls(**_filtered(cls, params))


@dataclass
class TalkerConfig:
    """Configuration for the Qwen3-TTS talker transformer."""

    code_predictor_config: CodePredictorConfig | dict[str, Any] | None = None
    vocab_size: int = 3072
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, Any] | None = field(
        default_factory=lambda: {
            "interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        }
    )
    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_code_groups: int = 16
    text_hidden_size: int = 2048
    text_vocab_size: int = 151936
    codec_eos_token_id: int = 2150
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149
    codec_language_id: dict[str, int] | None = None
    spk_id: dict[str, list[int]] | None = None
    spk_is_dialect: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.code_predictor_config is None:
            self.code_predictor_config = CodePredictorConfig()
        elif isinstance(self.code_predictor_config, dict):
            self.code_predictor_config = CodePredictorConfig.from_dict(
                self.code_predictor_config
            )

    @property
    def mrope_section(self) -> list[int] | None:
        if not self.rope_scaling:
            return None
        return self.rope_scaling.get("mrope_section")

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> TalkerConfig:
        return cls(**_filtered(cls, params))


@dataclass
class TokenizerDecoderConfig:
    """Configuration for the speech-tokenizer decoder (codes -> waveform)."""

    latent_dim: int = 1024
    codebook_dim: int = 512
    codebook_size: int = 2048
    decoder_dim: int = 1536
    hidden_act: str = "silu"
    hidden_size: int = 512
    intermediate_size: int = 1024
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    head_dim: int = 64
    num_attention_heads: int = 16
    num_hidden_layers: int = 8
    num_key_value_heads: int = 16
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    semantic_codebook_size: int = 4096
    attention_bias: bool = False
    attention_dropout: float = 0.0
    # Every code2wav transformer layer is sliding-window attention; the
    # official config exposes this as a computed property rather than data,
    # so it is not present in config.json and must not be read from there.
    sliding_window: int = 72
    upsample_rates: list[int] = field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: list[int] = field(default_factory=lambda: [2, 2])
    vector_quantization_hidden_dimension: int = 512

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> TokenizerDecoderConfig:
        return cls(**_filtered(cls, params))


@dataclass
class TokenizerEncoderConfig:
    """Configuration for the speech-tokenizer encoder (waveform -> codes)."""

    frame_rate: float = 12.5
    audio_channels: int = 1
    codebook_dim: int = 256
    codebook_size: int = 2048
    compress: int = 2
    dilation_growth_rate: int = 2
    head_dim: int = 64
    hidden_act: str = "gelu"
    hidden_size: int = 512
    intermediate_size: int = 2048
    kernel_size: int = 7
    last_kernel_size: int = 3
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    norm_eps: float = 1e-5
    num_attention_heads: int = 8
    num_filters: int = 64
    num_hidden_layers: int = 8
    num_key_value_heads: int = 8
    num_quantizers: int = 32
    num_residual_layers: int = 1
    num_semantic_quantizers: int = 1
    pad_mode: str = "constant"
    residual_kernel_size: int = 3
    rope_theta: float = 10000.0
    sampling_rate: int = 24000
    sliding_window: int = 250
    upsampling_ratios: list[int] = field(default_factory=lambda: [8, 6, 5, 4])
    use_causal_conv: bool = True
    use_conv_shortcut: bool = False
    vector_quantization_hidden_dimension: int = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> TokenizerEncoderConfig:
        params = dict(params)
        # The checkpoint spells the frame rate as a private field.
        if "frame_rate" not in params and "_frame_rate" in params:
            params["frame_rate"] = params["_frame_rate"]
        return cls(**_filtered(cls, params))


@dataclass
class TokenizerConfig:
    """Configuration for the Qwen3-TTS speech tokenizer.

    Loaded from the ``speech_tokenizer/config.json`` subdirectory of a
    checkpoint, not from the top-level model config.
    """

    encoder_config: TokenizerEncoderConfig | dict[str, Any] | None = None
    decoder_config: TokenizerDecoderConfig | dict[str, Any] | None = None
    encoder_valid_num_quantizers: int = 16
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    decode_upsample_rate: int = 1920
    encode_downsample_rate: int = 1920

    def __post_init__(self) -> None:
        if isinstance(self.encoder_config, dict):
            self.encoder_config = TokenizerEncoderConfig.from_dict(self.encoder_config)
        if self.decoder_config is None:
            self.decoder_config = TokenizerDecoderConfig()
        elif isinstance(self.decoder_config, dict):
            self.decoder_config = TokenizerDecoderConfig.from_dict(self.decoder_config)

    @property
    def has_encoder(self) -> bool:
        return self.encoder_config is not None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> TokenizerConfig:
        return cls(**_filtered(cls, params))


@dataclass
class ModelConfig:
    """Configuration for the MLX Qwen3-TTS stack.

    ``tokenizer_config`` stays a raw mapping: the speech tokenizer ships its
    own ``config.json`` in a subdirectory and is configured from there.
    """

    talker_config: TalkerConfig | dict[str, Any] | None = None
    speaker_encoder_config: SpeakerEncoderConfig | dict[str, Any] | None = None
    tokenizer_config: dict[str, Any] | None = None
    model_type: str = "qwen3_tts"
    tokenizer_type: str = "qwen3_tts_tokenizer_12hz"
    tts_model_size: str = "0b6"
    tts_model_type: str = "base"
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    sample_rate: int = 24000

    def __post_init__(self) -> None:
        if self.talker_config is None:
            self.talker_config = TalkerConfig()
        elif isinstance(self.talker_config, dict):
            self.talker_config = TalkerConfig.from_dict(self.talker_config)
        # Only Base checkpoints carry a speaker encoder; CustomVoice and
        # VoiceDesign select a preset voice instead and omit the section.
        if isinstance(self.speaker_encoder_config, dict):
            self.speaker_encoder_config = SpeakerEncoderConfig.from_dict(
                self.speaker_encoder_config
            )

    @property
    def has_speaker_encoder(self) -> bool:
        return self.speaker_encoder_config is not None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelConfig:
        return cls(**_filtered(cls, params))
