# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni_mlx.fun_asr.model import (  # noqa: E402
    FunASR,
    checkpoint_weights,
    read_configs,
)

TEXT_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "vocab_size": 50,
    "max_position_embeddings": 128,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "tie_word_embeddings": True,
}
# Revision 972ee603: encoder_config, flat adaptor fields, stem/layers/timestamp names.
FIRST_CONFIG = {
    "encoder_config": {
        "num_mel_bins": 4,
        "num_stacked_frames": 7,
        "d_model": 16,
        "encoder_attention_heads": 2,
        "encoder_ffn_dim": 32,
        "encoder_layers": 3,
        "num_timestamp_prediction_blocks": 2,
        "kernel_size": 3,
    },
    "adaptor_intermediate_size": 24,
    "adaptor_num_attention_heads": 2,
    "adaptor_num_hidden_layers": 2,
    "text_config": TEXT_CONFIG,
}
# Later revisions: audio_config/adaptor_config and one flat list of audio layers.
FLAT_CONFIG = {
    "audio_config": {
        "num_mel_bins": 4,
        "num_stacked_frames": 7,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "num_hidden_layers": 5,
        "num_timestamp_prediction_layers": 2,
        "fsmn_kernel_size": 3,
    },
    "adaptor_config": {
        "hidden_size": 32,
        "projector_hidden_size": 24,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
    },
    "text_config": TEXT_CONFIG,
}


def torch_conv_layout(name: str, value: mx.array) -> mx.array:
    return value.transpose(0, 2, 1) if name.endswith(".conv.weight") else value


def first_revision_name(name: str) -> str:
    if name.startswith("model."):
        return name.replace("model.", "model.language_model.", 1)
    else:
        return "model." + name


def flat_revision_name(name: str) -> str:
    """This module tree's name as the later Hub revision spells it (3 encoder, 2 timestamp layers)."""
    if name.startswith("model."):
        return name.replace("model.", "model.language_model.", 1)
    else:
        pass
    sublayer = (
        name.replace(".self_attn_layer_norm.", ".input_layernorm.")
        .replace(".final_layer_norm.", ".post_attention_layernorm.")
        .replace(".fc1.", ".mlp.fc1.")
        .replace(".fc2.", ".mlp.fc2.")
        .replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        .replace(".fsmn.", ".self_attn.fsmn.")
    )
    if name.startswith("multi_modal_projector.blocks."):
        return "model." + sublayer.replace(".blocks.", ".layers.", 1)
    elif name.startswith("audio_tower.stem."):
        return "model." + sublayer.replace("audio_tower.stem.", "audio_tower.layers.0.")
    elif name.startswith("audio_tower.layers."):
        index, rest = sublayer.removeprefix("audio_tower.layers.").split(".", 1)
        return f"model.audio_tower.layers.{int(index) + 1}.{rest}"
    elif name.startswith("audio_tower.timestamp_prediction_layers."):
        index, rest = sublayer.removeprefix(
            "audio_tower.timestamp_prediction_layers."
        ).split(".", 1)
        return f"model.audio_tower.layers.{int(index) + 3}.{rest}"
    elif name.startswith("audio_tower.layer_norm."):
        return "model.audio_tower.layers.2.final_layernorm." + name.rsplit(".", 1)[1]
    elif name.startswith("audio_tower.timestamp_prediction_layer_norm."):
        return "model.audio_tower.layers.4.final_layernorm." + name.rsplit(".", 1)[1]
    else:
        return "model." + name


@pytest.mark.parametrize(
    ("config", "checkpoint_name"),
    [(FIRST_CONFIG, first_revision_name), (FLAT_CONFIG, flat_revision_name)],
)
def test_both_hub_layouts_load_every_parameter(config, checkpoint_name) -> None:
    audio_config, adaptor_config, text_config = read_configs(config)
    assert (audio_config.encoder_layers, audio_config.timestamp_layers) == (3, 2)
    source = FunASR(audio_config, adaptor_config, text_config)
    expected = dict(tree_flatten(source.parameters()))
    checkpoint = {
        checkpoint_name(name): torch_conv_layout(name, value)
        for name, value in expected.items()
    }
    checkpoint["model.language_model.rotary_emb.inv_freq"] = mx.zeros((8,))

    target = FunASR(audio_config, adaptor_config, text_config)
    target.load_weights(
        list(checkpoint_weights(checkpoint, audio_config).items()), strict=True
    )
    for name, value in tree_flatten(target.parameters()):
        assert mx.array_equal(value, expected[name]), name


def test_encode_audio_returns_one_row_per_audio_token() -> None:
    audio_config, adaptor_config, text_config = read_configs(FIRST_CONFIG)
    model = FunASR(audio_config, adaptor_config, text_config)
    embeddings = model.encode_audio(mx.zeros((9, 28)), audio_token_count=2)
    assert embeddings.shape == (2, 32)
