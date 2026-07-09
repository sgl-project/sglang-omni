# SPDX-License-Identifier: Apache-2.0
"""Native HF config for bosonai/higgs-audio-v3-stt.

The checkpoint ships custom remote code (``auto_map``) targeting the
transformers-4 layer API, which breaks on transformers 5. Registering a
native config class under the checkpoint's ``model_type`` lets
``AutoConfig.from_pretrained`` resolve locally without
``trust_remote_code``.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class HiggsAudioEncoderConfig(PretrainedConfig):
    """Whisper-large-v3-style tower (+ AvgPool1d(2) applied by the model)."""

    model_type = "higgs_audio_encoder"

    def __init__(
        self,
        d_model=1280,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        num_mel_bins=128,
        max_source_positions=1500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions


class HiggsAudio3Config(PretrainedConfig):
    model_type = "higgs_audio_3"
    sub_configs = {
        "audio_encoder_config": HiggsAudioEncoderConfig,
        "text_config": Qwen3Config,
    }

    def __init__(
        self,
        audio_encoder_config=None,
        text_config=None,
        audio_in_token_idx=151672,
        audio_eos_token_id=151670,
        projector_temporal_downsample=2,
        chunk_size_seconds=4.0,
        **kwargs,
    ):
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config = HiggsAudioEncoderConfig(**audio_encoder_config)
        elif audio_encoder_config is None:
            audio_encoder_config = HiggsAudioEncoderConfig()
        self.audio_encoder_config = audio_encoder_config

        if isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)
        elif text_config is None:
            text_config = Qwen3Config()
        # The checkpoint's text_config claims tie_word_embeddings=True, but
        # it ships a distinct audio_decoder_proj.text_lm_head whose weights
        # differ from embed_tokens (finetuned apart). Force untied so the
        # LLM allocates a real lm_head param for load_weights to fill.
        text_config.tie_word_embeddings = False
        self.text_config = text_config

        self.audio_in_token_idx = audio_in_token_idx
        self.audio_eos_token_id = audio_eos_token_id
        self.projector_temporal_downsample = projector_temporal_downsample
        self.chunk_size_seconds = chunk_size_seconds
        super().__init__(**kwargs)

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.text_config


def higgs_audio_token_lengths(mel_lengths: Any) -> torch.Tensor:
    """Audio-embedding count per chunk for ``mel_lengths`` valid mel frames.

    Mirrors the tower + projector shape math: conv2 (k3 s2 p1) ->
    AvgPool1d(2, s2) -> projector depthwise conv (k3 s2 p1).
    """
    if not isinstance(mel_lengths, torch.Tensor):
        mel_lengths = torch.tensor(mel_lengths)
    after_conv = (mel_lengths - 1) // 2 + 1
    after_pool = after_conv // 2
    return (after_pool - 1) // 2 + 1


def higgs_num_audio_tokens(num_mel_frames: int) -> int:
    return int(higgs_audio_token_lengths(num_mel_frames).item())


AutoConfig.register("higgs_audio_3", HiggsAudio3Config, exist_ok=True)

__all__ = [
    "HiggsAudio3Config",
    "HiggsAudioEncoderConfig",
    "higgs_audio_token_lengths",
    "higgs_num_audio_tokens",
]
