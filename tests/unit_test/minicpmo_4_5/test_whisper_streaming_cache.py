# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib

import torch
from torch import nn
from transformers import WhisperConfig


def _native_model_module():
    # Importing SGLang on a CPU-only macOS test host reaches decorators that
    # eagerly initialize Inductor/Triton. The model under test does not use
    # torch.compile; make those decorators inert only during module import.
    original_compile = torch.compile
    torch.compile = lambda *args, **kwargs: (lambda function: function)
    try:
        return importlib.import_module("sglang_omni.models.minicpmo_4_5.sglang_model")
    finally:
        torch.compile = original_compile


def _tiny_whisper_config(*, max_source_positions: int) -> WhisperConfig:
    config = WhisperConfig(
        num_mel_bins=4,
        d_model=8,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        max_source_positions=max_source_positions,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return config


def test_transformers_5_whisper_cache_grows_across_two_chunks() -> None:
    module = _native_model_module()
    encoder = module.MiniCPMO45WhisperEncoder(
        _tiny_whisper_config(max_source_positions=8)
    ).eval()
    features = torch.randn(1, 4, 4)

    first = encoder(
        features,
        attention_mask=torch.zeros(1, 1, 2, 2),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    assert first.past_key_values.self_attention_cache.get_seq_length() == 2

    second = encoder(
        features,
        attention_mask=torch.zeros(1, 1, 2, 4),
        past_key_values=first.past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    assert second.past_key_values.self_attention_cache.get_seq_length() == 4


def test_streaming_audio_resets_whisper_cache_at_position_boundary() -> None:
    module = _native_model_module()
    encoder = module.MiniCPMO45WhisperEncoder(
        _tiny_whisper_config(max_source_positions=4)
    ).eval()

    class Harness:
        apm = encoder
        audio_projection_layer = nn.Identity()
        audio_avg_pooler = nn.AvgPool1d(1, stride=1)
        audio_encoder_layer = -1

        @staticmethod
        def _get_feat_extract_output_lengths(lengths):
            return lengths, lengths

    data = {
        "audio_features": torch.randn(1, 4, 4),
        "audio_feature_lens": [torch.tensor([2])],
    }
    _, first_cache = module.MiniCPMO45ForCausalLM.encode_audio_streaming(
        Harness(),
        data,
        past_key_values=None,
        use_extra_context=False,
        prefix_extra_frames=0,
        suffix_extra_frames=0,
    )
    assert first_cache.self_attention_cache.get_seq_length() == 2

    _, reset_cache = module.MiniCPMO45ForCausalLM.encode_audio_streaming(
        Harness(),
        data,
        past_key_values=first_cache,
        use_extra_context=False,
        prefix_extra_frames=0,
        suffix_extra_frames=0,
    )
    # 2 old + 2 new reaches the learned boundary, so the official behavior is
    # to start a fresh audio-encoder cache rather than grow it to four.
    assert reset_cache.self_attention_cache.get_seq_length() == 2
