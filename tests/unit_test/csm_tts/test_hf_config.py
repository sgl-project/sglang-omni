# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the synthetic CSM backbone LlamaConfig."""

from __future__ import annotations

from sglang_omni.models.csm_tts.hf_config import CsmTtsHfConfig


def test_text_config_is_backbone_not_audio_vocab():
    config = CsmTtsHfConfig()
    text = config.get_text_config()
    assert text.vocab_size == 128256
    assert text.hidden_size == 2048
    assert text.num_hidden_layers == 16
    assert text.num_attention_heads == 32
    assert text.num_key_value_heads == 8
    assert text.vocab_size != 2051
