# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.fun_asr.sglang_model import (
    FunAsrNanoAdaptor,
    FunAsrNanoAudioEncoder,
    FunAsrNanoForConditionalGeneration,
)


def test_fun_asr_audio_modules_match_current_checkpoint_parameter_names() -> None:
    encoder = FunAsrNanoAudioEncoder(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
    )
    encoder_names = set(dict(encoder.named_parameters()))

    assert "stem.self_attn.q_proj.weight" in encoder_names
    assert "stem.self_attn.k_proj.weight" in encoder_names
    assert "stem.self_attn.v_proj.weight" in encoder_names
    assert "stem.self_attn.out_proj.weight" in encoder_names
    assert "stem.fsmn.conv.weight" in encoder_names
    assert "stem.fc1.weight" in encoder_names
    assert "layers.0.self_attn_layer_norm.weight" in encoder_names
    assert "layers.0.final_layer_norm.weight" in encoder_names
    assert "layer_norm.weight" in encoder_names
    assert "timestamp_prediction_layers.0.fc2.weight" in encoder_names
    assert "timestamp_prediction_layer_norm.weight" in encoder_names

    projector = FunAsrNanoAdaptor(
        encoder_dim=8,
        llm_dim=8,
        ffn_dim=16,
        num_layers=1,
        attention_heads=2,
    )
    projector_names = set(dict(projector.named_parameters()))

    assert "linear_1.weight" in projector_names
    assert "linear_2.weight" in projector_names
    assert "blocks.0.self_attn.q_proj.weight" in projector_names
    assert "blocks.0.self_attn_layer_norm.weight" in projector_names
    assert "blocks.0.fc1.weight" in projector_names
    assert "blocks.0.final_layer_norm.weight" in projector_names


def _weight_loader_target() -> FunAsrNanoForConditionalGeneration:
    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=False)
    )
    model.audio_tower = nn.Module()
    model.audio_tower.layer_norm = nn.LayerNorm(2)
    model.multi_modal_projector = nn.Module()
    model.multi_modal_projector.linear_1 = nn.Linear(2, 2)
    return model


def test_fun_asr_weight_loader_loads_current_audio_prefixes() -> None:
    model = _weight_loader_target()
    expected = torch.tensor([2.0, 3.0])

    model.load_weights([("model.audio_tower.layer_norm.weight", expected.clone())])

    assert torch.equal(model.audio_tower.layer_norm.weight, expected)


def test_fun_asr_weight_loader_rejects_unknown_audio_weights() -> None:
    model = _weight_loader_target()

    with pytest.raises(ValueError, match=r"model\.audio_tower\.missing\.weight"):
        model.load_weights([("model.audio_tower.missing.weight", torch.ones(2))])
