# SPDX-License-Identifier: Apache-2.0

"""Parse Mistral-format params.json into structured config for Voxtral realtime ASR."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VoxtralTextConfig:
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 131072
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    max_seq_len: int = 131072
    tied_embeddings: bool = True
    ada_rms_norm_t_cond_dim: int | None = None


@dataclass
class VoxtralAudioEncodingArgs:
    num_mel_bins: int = 128
    window_size: int = 400
    hop_length: int = 160
    sampling_rate: int = 16000
    global_log_mel_max: float | None = None


@dataclass
class VoxtralAudioConfig:
    dim: int = 1280
    n_layers: int = 32
    hidden_dim: int = 5120
    n_heads: int = 20
    head_dim: int = 64
    vocab_size: int = 51864
    max_source_positions: int = 1500
    encoder_layers: int = 32  # alias for n_layers
    encoder_ffn_dim: int = 5120  # alias for hidden_dim
    encoder_attention_heads: int = 20  # alias for n_heads
    downsample_factor: int = 2
    block_pool_size: int = 4
    is_causal: bool = True
    sliding_window: int | None = 750
    pos_embed: str = "sinusoidal"
    audio_encoding_args: VoxtralAudioEncodingArgs = field(
        default_factory=VoxtralAudioEncodingArgs
    )


@dataclass
class VoxtralRealtimeConfig:
    text_config: VoxtralTextConfig = field(default_factory=VoxtralTextConfig)
    audio_config: VoxtralAudioConfig = field(default_factory=VoxtralAudioConfig)
    model_path: str = ""

    @staticmethod
    def from_model_path(model_path: str) -> "VoxtralRealtimeConfig":
        params_path = os.path.join(model_path, "params.json")
        with open(params_path) as f:
            params = json.load(f)

        text_config = VoxtralTextConfig(
            dim=params.get("dim", 3072),
            n_layers=params.get("n_layers", 26),
            head_dim=params.get("head_dim", 128),
            hidden_dim=params.get("hidden_dim", 9216),
            n_heads=params.get("n_heads", 32),
            n_kv_heads=params.get("n_kv_heads", 8),
            vocab_size=params.get("vocab_size", 131072),
            rope_theta=params.get("rope_theta", 1000000.0),
            norm_eps=params.get("norm_eps", 1e-5),
            max_seq_len=params.get("max_seq_len", 131072),
            tied_embeddings=params.get("tied_embeddings", True),
            ada_rms_norm_t_cond_dim=(
                params.get("ada_rms_norm_t_cond_dim")
                if params.get("ada_rms_norm_t_cond", False)
                else None
            ),
        )

        mm = params.get("multimodal", {})
        whisper_args = mm.get("whisper_model_args", {})
        encoder_args = whisper_args.get("encoder_args", {})
        downsample_args = whisper_args.get("downsample_args", {})
        audio_enc_args = encoder_args.get("audio_encoding_args", {})

        downsample_factor = downsample_args.get("downsample_factor", 2)
        is_causal = encoder_args.get("causal", False)
        block_pool_size = downsample_factor if is_causal else 1

        audio_config = VoxtralAudioConfig(
            dim=encoder_args.get("dim", 1280),
            n_layers=encoder_args.get("n_layers", 32),
            hidden_dim=encoder_args.get("hidden_dim", 5120),
            n_heads=encoder_args.get("n_heads", 20),
            head_dim=encoder_args.get("head_dim", 64),
            vocab_size=encoder_args.get("vocab_size", 51864),
            max_source_positions=encoder_args.get("max_source_positions") or 1500,
            encoder_layers=encoder_args.get("n_layers", 32),
            encoder_ffn_dim=encoder_args.get("hidden_dim", 5120),
            encoder_attention_heads=encoder_args.get("n_heads", 20),
            downsample_factor=downsample_factor,
            block_pool_size=block_pool_size,
            is_causal=is_causal,
            sliding_window=encoder_args.get("sliding_window", None),
            pos_embed=encoder_args.get("pos_embed", "sinusoidal"),
            audio_encoding_args=VoxtralAudioEncodingArgs(
                num_mel_bins=audio_enc_args.get("num_mel_bins", 128),
                window_size=audio_enc_args.get("window_size", 400),
                hop_length=audio_enc_args.get("hop_length", 160),
                sampling_rate=audio_enc_args.get("sampling_rate", 16000),
                global_log_mel_max=audio_enc_args.get("global_log_mel_max", None),
            ),
        )

        return VoxtralRealtimeConfig(
            text_config=text_config,
            audio_config=audio_config,
            model_path=model_path,
        )

    def to_hf_mistral_config(self) -> Any:
        """Build a HuggingFace MistralConfig for loading transformers.MistralModel."""
        from transformers import MistralConfig

        return MistralConfig(
            hidden_size=self.text_config.dim,
            intermediate_size=self.text_config.hidden_dim,
            num_hidden_layers=self.text_config.n_layers,
            num_attention_heads=self.text_config.n_heads,
            num_key_value_heads=self.text_config.n_kv_heads,
            head_dim=self.text_config.head_dim,
            vocab_size=self.text_config.vocab_size,
            max_position_embeddings=self.text_config.max_seq_len,
            rope_theta=self.text_config.rope_theta,
            rms_norm_eps=self.text_config.norm_eps,
            tie_word_embeddings=self.text_config.tied_embeddings,
            sliding_window=None,
        )
