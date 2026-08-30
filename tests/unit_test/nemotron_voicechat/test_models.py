# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from sglang_omni.models.nemotron_voicechat.configuration import EarTTSConfig
from sglang_omni.models.nemotron_voicechat.convert_duplex import (
    _configure_duplex_config,
)
from sglang_omni.models.nemotron_voicechat.registration import register_voicechat_models
from sglang_omni.models.nemotron_voicechat.talker import (
    EarTTSForCausalLM,
    MaskGITSampler,
)
from sglang_omni.models.nemotron_voicechat.thinker import NemotronDuplexHForCausalLM


def test_voicechat_models_register_with_sglang() -> None:
    from sglang.srt.models.registry import ModelRegistry

    register_voicechat_models()

    assert ModelRegistry.models["NemotronDuplexHForCausalLM"] is (
        NemotronDuplexHForCausalLM
    )
    assert ModelRegistry.models["EarTTSForCausalLM"] is EarTTSForCausalLM


def test_eartts_auto_config_roundtrip(tmp_path) -> None:
    register_voicechat_models()
    config = EarTTSConfig(
        architectures=["EarTTSForCausalLM"],
        hidden_size=16,
        num_hidden_layers=2,
    )
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))

    loaded = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, EarTTSConfig)
    assert loaded.architectures == ["EarTTSForCausalLM"]
    assert loaded.rope_parameters["full_attention"]["rope_theta"] == 1_000_000


def test_maskgit_outputs_one_valid_code_per_quantizer() -> None:
    torch.manual_seed(1)
    config = EarTTSConfig(
        hidden_size=8,
        intermediate_size=16,
        num_quantizers=3,
        codebook_size=4,
        latent_size=4,
        num_iter=2,
        exponent=3.0,
        mog_low_rank=2,
        mog_num_layers=1,
        mog_num_predictions=5,
        top_p_or_k=1.0,
    )
    sampler = MaskGITSampler(config)

    codes = sampler(torch.randn(2, config.hidden_size))

    assert codes.shape == (2, config.num_quantizers)
    assert codes.dtype == torch.long
    assert torch.all(codes >= 0)
    assert torch.all(codes < config.codebook_size)


class _FakeEarTTSBackbone:
    def load_weights(self, weights):
        [(name, _)] = weights
        return {name}


def _fake_eartts_model():
    model = SimpleNamespace(backbone=_FakeEarTTSBackbone())
    params = [
        ("backbone.model.embed_tokens.weight", torch.nn.Parameter(torch.zeros(1))),
        ("backbone.model.layers.0.weight", torch.nn.Parameter(torch.zeros(1))),
        ("total_emb.bos_emb", torch.nn.Parameter(torch.zeros(1))),
    ]
    buffers = [("sil_tokens", torch.zeros(1, dtype=torch.int32))]
    model.named_parameters = lambda: iter(params)
    model.named_buffers = lambda: iter(buffers)
    return model


def test_eartts_weight_loader_requires_complete_checkpoint() -> None:
    model = _fake_eartts_model()

    with pytest.raises(RuntimeError, match="total_emb.bos_emb"):
        EarTTSForCausalLM.load_weights(
            model,
            [
                ("model.backbone.layers.0.weight", torch.ones(1)),
                ("model.sil_tokens", torch.ones(1, dtype=torch.int32)),
            ],
        )


def test_eartts_weight_loader_accepts_complete_checkpoint() -> None:
    model = _fake_eartts_model()

    loaded = EarTTSForCausalLM.load_weights(
        model,
        [
            ("model.backbone.layers.0.weight", torch.ones(1)),
            ("model.total_emb.bos_emb", torch.ones(1)),
            ("model.sil_tokens", torch.ones(1, dtype=torch.int32)),
        ],
    )

    assert loaded == {
        "backbone.model.layers.0.weight",
        "total_emb.bos_emb",
        "sil_tokens",
    }


def test_duplex_unified_checkpoint_name_mapping() -> None:
    mapper = NemotronDuplexHForCausalLM._map_voicechat_weight_name

    assert mapper("stt_model.llm.backbone.layers.0.weight") == ("model.layers.0.weight")
    assert mapper("stt_model.llm.layers.0.weight") == "model.layers.0.weight"
    assert mapper("stt_model.function_head.weight") == "function_head.weight"


def test_duplex_conversion_pins_voicechat_runtime_config() -> None:
    config = _configure_duplex_config(
        SimpleNamespace(),
        {
            "duplex_text_channel_weight": 1,
            "duplex_user_channel_weight": 1,
            "duplex_function_channel_weight": 2,
        },
    )

    assert config.architectures == ["NemotronDuplexHForCausalLM"]
    assert config.predict_user_text is False
    assert config.use_function_head is True
    assert config.mamba_ssm_dtype == "float32"
    assert config.duplex_text_channel_weight == 1.0
    assert config.duplex_user_channel_weight == 1.0
    assert config.duplex_function_channel_weight == 2.0
    assert config.fuse_method == "add"
