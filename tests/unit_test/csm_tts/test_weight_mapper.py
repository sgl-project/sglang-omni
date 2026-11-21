# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the CSM HF-export weight remap."""

from __future__ import annotations

import pytest

from sglang_omni.models.csm_tts.weight_loader import CsmWeightMapper


def test_lm_head_routes_to_codebook0_head_not_text_lm():
    mapped = CsmWeightMapper().map("lm_head.weight")
    assert mapped == "codebook0_head.weight"
    assert "lm_head" not in mapped


def test_backbone_and_depth_prefixes():
    mapper = CsmWeightMapper()
    assert (
        mapper.map("backbone_model.layers.3.self_attn.q_proj.weight")
        == "backbone.model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        mapper.map("depth_decoder.model.layers.1.mlp.up_proj.weight")
        == "depth_decoder.layers.1.mlp.up_proj.weight"
    )
    assert (
        mapper.map("embed_text_tokens.weight") == "backbone.model.embed_tokens.weight"
    )


def test_tied_embed_and_codec_are_skipped():
    mapper = CsmWeightMapper()
    assert mapper.map("depth_decoder.model.embed_tokens.weight") is None
    assert mapper.map("codec_model.encoder.layers.0.weight") is None


def test_unexpected_name_is_strict():
    with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
        CsmWeightMapper().map("not_a_real_csm_tensor.weight")
