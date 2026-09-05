# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-derived runtime values shared by MOSS-TTS-Realtime tests."""

from __future__ import annotations

from types import SimpleNamespace

from transformers import PretrainedConfig, Qwen3Config

from sglang_omni.models.moss_tts_realtime.sglang_model import _normalize_config
from sglang_omni.models.moss_tts_realtime.stages import (
    bind_moss_tts_realtime_processor_config,
)


def _build_runtime_config():
    config = PretrainedConfig(
        architectures=["MossTTSRealtime"],
        language_config=Qwen3Config(
            vocab_size=151936,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=40960,
            tie_word_embeddings=False,
        ),
        local_config=PretrainedConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=33,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            rope_theta=1_000_000.0,
            audio_vocab_size=1027,
            audio_pad_token=1024,
            rvq=16,
        ),
        rvq=16,
        audio_pad_token=1024,
        audio_vocab_size=1027,
        reference_audio_pad=151654,
        text_pad=151655,
    )
    _normalize_config(config)
    processor = SimpleNamespace(
        channels=16,
        delay_tokens_len=12,
        audio_channel_pad=config.audio_pad_token,
        audio_bos_token=1025,
        audio_eos_token=1026,
        audio_pad_token_id=config.reference_audio_pad,
        text_pad_token_id=config.text_pad,
    )
    return bind_moss_tts_realtime_processor_config(config, processor)


MODEL_CONFIG = _build_runtime_config()
AUDIO_PAD_TOKEN_ID = int(MODEL_CONFIG.audio_pad_token)
AUDIO_BOS_TOKEN_ID = int(MODEL_CONFIG.audio_bos_token)
AUDIO_EOS_TOKEN_ID = int(MODEL_CONFIG.audio_eos_token)
AUDIO_VOCAB_SIZE = int(MODEL_CONFIG.audio_vocab_size)
REFERENCE_AUDIO_PAD_TOKEN_ID = int(MODEL_CONFIG.reference_audio_pad)
TEXT_PAD_TOKEN_ID = int(MODEL_CONFIG.text_pad)
