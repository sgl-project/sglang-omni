# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the synthetic CSM backbone LlamaConfig."""

from __future__ import annotations

from sglang_omni.models.csm_tts.hf_config import (
    CsmTtsHfConfig,
    build_backbone_llama_config,
    build_depth_decoder_config,
)


def test_text_config_is_backbone_not_audio_vocab():
    cfg = CsmTtsHfConfig()
    text = cfg.get_text_config()
    assert text.vocab_size == 128256
    assert text.hidden_size == 2048
    assert text.num_hidden_layers == 16
    assert text.num_attention_heads == 32
    assert text.num_key_value_heads == 8
    assert text.vocab_size != 2051


def test_depth_decoder_config_shape():
    depth = build_depth_decoder_config(CsmTtsHfConfig())
    assert int(depth.num_codebooks) == 32
    assert int(depth.vocab_size) == 2051
    assert int(depth.hidden_size) == 1024
    assert int(depth.num_hidden_layers) == 4


def test_backbone_builder_ties_embeddings():
    llama = build_backbone_llama_config(CsmTtsHfConfig())
    assert llama.tie_word_embeddings is True
