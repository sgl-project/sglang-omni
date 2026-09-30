# SPDX-License-Identifier: Apache-2.0
"""Tiny EasyMagpie configs so the talker heads and codec run on CPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from sglang_omni.models.easymagpie_tts.hf_config import EasyMagpieTTSConfig
from sglang_omni.models.easymagpie_tts.local_transformer import EasyMagpieTTSHeads
from sglang_omni.models.easymagpie_tts.sglang_model import (
    EasyMagpieTTSForConditionalGeneration,
)

TINY_TTS_CONFIG = {
    "text_vocab_size": 64,
    "text_eos_id": 63,
    "embedding_dim": 8,
    "audio_embedding_dim": 8,
    "num_audio_codebooks": 2,
    "codebook_size": 16,
    "frame_stacking_factor": 2,
    "phoneme_stacking_factor": 1,
    "phoneme_vocab_size": 20,
    "phoneme_bos_id": 17,
    "phoneme_eos_id": 18,
    "phoneme_unk_id": 19,
    "streaming_phonemes_delay": 3,
    "streaming_speech_delay": 5,
    "local_transformer_n_layers": 1,
    "local_transformer_n_heads": 2,
    "local_transformer_hidden_dim": 8,
}


@pytest.fixture
def tiny_raw_config() -> dict:
    return dict(TINY_TTS_CONFIG)


@pytest.fixture
def tiny_tts_config() -> EasyMagpieTTSConfig:
    return EasyMagpieTTSConfig.from_dict(TINY_TTS_CONFIG)


@pytest.fixture
def talker(tiny_tts_config) -> EasyMagpieTTSForConditionalGeneration:
    """The talker with its TTS heads but without the SGLang backbone."""
    model = EasyMagpieTTSForConditionalGeneration.__new__(
        EasyMagpieTTSForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=2, eos_token_id=1)
    model.tts_config = tiny_tts_config
    model.heads = EasyMagpieTTSHeads(tiny_tts_config)
    model.decode_step = None
    model.last_audio_codes = None
    model.last_phoneme_tokens = None
    model.last_audio_eos = None
    return model
