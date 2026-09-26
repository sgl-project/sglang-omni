# SPDX-License-Identifier: Apache-2.0
"""The Llama-shaped config the checkpoint shim writes for SGLang."""

import json

from sglang_omni.models.personaplex.hf_config import build_backbone_config


def test_backbone_config_is_a_moshi_shaped_llama():
    config = build_backbone_config(context_length=4096)
    assert config["architectures"] == ["PersonaPlexForCausalLM"]
    assert config["model_type"] == "llama"
    assert config["rope_is_neox_style"] is False
    assert config["rms_norm_eps"] == 1e-8
    assert config["intermediate_size"] == 11264
    assert config["vocab_size"] == 32000
    assert config["max_position_embeddings"] == 4096
    json.dumps(config)
