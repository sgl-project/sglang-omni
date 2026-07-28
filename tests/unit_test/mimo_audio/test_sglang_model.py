# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from sglang_omni.models.mimo_audio.sglang_model import (
    MiMoAudioModel,
    MiMoInputPatchEncoder,
    is_output_only_weight,
    make_input_local_config,
    parse_channel_values,
)


def _tiny_config() -> Qwen2Config:
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config.group_size = 4
    config.audio_channels = 8
    config.input_local_dim = 16
    config.input_local_layers = 1
    config.local_attn_heads = 4
    config.local_attn_dropout = 0.0
    config.speech_vocab_size = "17-17-9-9-9-9-9-9"
    config.speech_zeroemb_idx = "16-16-8-8-8-8-8-8"
    config.input_full_attention = True
    return config


def test_channel_config_parsing_is_exact() -> None:
    assert parse_channel_values("1024-1024-128-128-128-128-128-128") == (
        1024,
        1024,
        128,
        128,
        128,
        128,
        128,
        128,
    )


def test_input_local_config_matches_checkpoint_contract() -> None:
    local = make_input_local_config(_tiny_config())

    assert local.hidden_size == 16
    assert local.num_hidden_layers == 1
    assert local.num_attention_heads == 4
    assert local.num_key_value_heads == 4
    assert local.head_dim == 4
    assert local.intermediate_size == 64
    assert local._attn_implementation == "eager"


def test_patch_encoder_shape_dtype_and_full_attention() -> None:
    torch.manual_seed(0)
    encoder = MiMoInputPatchEncoder(_tiny_config()).eval()
    codes = torch.zeros((8, 8), dtype=torch.long)
    first = encoder(codes)

    changed = codes.clone()
    changed[3, 0] = 1
    second = encoder(changed)

    assert first.shape == (2, 32)
    assert first.dtype == encoder.speech_embeddings[0].weight.dtype
    # Frame 3 is in the future relative to frame 0. Under official full
    # attention it can affect the entire first patch representation.
    assert not torch.equal(first[0], second[0])
    # It cannot affect the independently batched second patch.
    assert torch.equal(first[1], second[1])


def test_patch_encoder_rejects_out_of_range_codes() -> None:
    encoder = MiMoInputPatchEncoder(_tiny_config())
    codes = torch.zeros((4, 8), dtype=torch.long)
    codes[0, 2] = 9

    try:
        encoder(codes)
    except ValueError as exc:
        assert "channel 2" in str(exc)
    else:
        raise AssertionError("out-of-range MiMo code was accepted")


def test_audio_output_weights_are_explicitly_classified() -> None:
    assert is_output_only_weight("local_transformer.layers.0.self_attn.q_proj.weight")
    assert is_output_only_weight("local_transformer_lm_heads.0.weight")
    assert is_output_only_weight("hidden_states_downcast.weight")
    assert not is_output_only_weight("input_local_transformer.layers.0.norm.weight")
    assert not is_output_only_weight("speech_group_downcast.weight")


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded: list[str] = []

    def load_weights(self, weights) -> None:
        self.loaded = [name for name, _ in weights]


def _loader_fixture() -> MiMoAudioModel:
    model = MiMoAudioModel.__new__(MiMoAudioModel)
    nn.Module.__init__(model)
    model.language_model = _FakeLanguageModel()
    model.input_patch_encoder = nn.Module()
    model.input_patch_encoder.input_local_transformer = nn.Linear(2, 2, bias=False)
    model.input_patch_encoder.speech_embeddings = nn.ModuleList([nn.Embedding(2, 2)])
    model.input_patch_encoder.speech_group_downcast = nn.Linear(2, 2, bias=False)
    model.loaded_weight_names = set()
    model.skipped_output_weight_names = set()
    model.unexpected_weight_names = set()
    return model


def test_weight_mapping_loads_all_input_weights_and_skips_output_weights() -> None:
    model = _loader_fixture()
    weights = [
        ("model.embed_tokens.weight", torch.zeros(2, 2)),
        ("lm_head.weight", torch.zeros(2, 2)),
        ("input_local_transformer.weight", torch.ones(2, 2)),
        ("speech_embeddings.0.weight", torch.ones(2, 2)),
        ("speech_group_downcast.weight", torch.ones(2, 2)),
        ("local_transformer.layers.0.weight", torch.ones(2, 2)),
    ]

    loaded = model.load_weights(weights)

    assert model.language_model.loaded == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    assert model.skipped_output_weight_names == {"local_transformer.layers.0.weight"}
    assert {
        "input_local_transformer.weight",
        "speech_embeddings.0.weight",
        "speech_group_downcast.weight",
    } <= loaded


def test_weight_mapping_rejects_missing_required_input_weight() -> None:
    model = _loader_fixture()
    weights = [
        ("model.embed_tokens.weight", torch.zeros(2, 2)),
        ("input_local_transformer.weight", torch.ones(2, 2)),
        ("speech_embeddings.0.weight", torch.ones(2, 2)),
    ]

    try:
        model.load_weights(weights)
    except ValueError as exc:
        assert "Missing required MiMo input weights" in str(exc)
        assert "speech_group_downcast.weight" in str(exc)
    else:
        raise AssertionError("incomplete MiMo input checkpoint was accepted")


def test_weight_mapping_rejects_unexpected_input_weight() -> None:
    model = _loader_fixture()
    weights = [
        ("model.embed_tokens.weight", torch.zeros(2, 2)),
        ("input_local_transformer.weight", torch.ones(2, 2)),
        ("speech_embeddings.0.weight", torch.ones(2, 2)),
        ("speech_group_downcast.weight", torch.ones(2, 2)),
        ("unknown_input_module.weight", torch.ones(2, 2)),
    ]

    try:
        model.load_weights(weights)
    except ValueError as exc:
        assert "Unexpected MiMo checkpoint weights" in str(exc)
        assert "unknown_input_module.weight" in str(exc)
    else:
        raise AssertionError("unexpected MiMo input weight was accepted")
