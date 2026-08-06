# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from sglang_omni.models.mimo_v25_asr.sglang_model import (
    MiMoInputPatchEncoder,
    MiMoV2ASRForCausalLM,
    is_output_only_weight,
    parse_channel_values,
)


def _tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        group_size=4,
        audio_channels=8,
        input_local_dim=32,
        input_local_layers=1,
        local_attn_heads=4,
        speech_vocab_size="8-8-8-8-8-8-8-8",
        speech_zeroemb_idx="7-7-7-7-7-7-7-7",
    )


def test_parse_channel_values() -> None:
    assert parse_channel_values("1-2-3-4-5-6-7-8") == (1, 2, 3, 4, 5, 6, 7, 8)
    assert parse_channel_values(9) == (9,) * 8


def test_output_only_weight_prefixes() -> None:
    assert is_output_only_weight("local_transformer.layers.0.weight")
    assert not is_output_only_weight("speech_embeddings.0.weight")


def test_input_patch_encoder_groups_frames() -> None:
    encoder = MiMoInputPatchEncoder(_tiny_config())
    codes = torch.zeros((8, 8), dtype=torch.long)
    embeds = encoder(codes)
    assert embeds.shape == (2, 64)


def test_model_class_name_matches_hf_architecture() -> None:
    assert MiMoV2ASRForCausalLM.__name__ == "MiMoV2ASRForCausalLM"
