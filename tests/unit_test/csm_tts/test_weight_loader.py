# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CsmWeightMapper, the HF-transformers shard name remap.

The mapper guards documented traps: the checkpoint ``lm_head.weight`` is the
codebook-0 head (NOT a text head); the depth tied-embed duplicate and the whole
``codec_model.*`` subtree are deliberate skips; any tensor outside the census is
a hard error. CPU-only (no sglang, no GPU)."""

from __future__ import annotations

import pytest

from sglang_omni.models.csm_tts.weight_loader import CsmWeightMapper


def test_exact_map_routes() -> None:
    m = CsmWeightMapper()
    assert m.map("embed_text_tokens.weight") == "backbone.model.embed_tokens.weight"
    assert (
        m.map("backbone_model.embed_tokens.embed_audio_tokens.weight")
        == "frame_embedding.weight"
    )
    assert m.map("backbone_model.norm.weight") == "backbone.model.norm.weight"
    # The checkpoint lm_head IS the codebook-0 head, never a text head.
    assert m.map("lm_head.weight") == "codebook0_head.weight"
    assert m.map("depth_decoder.codebooks_head.weight") == "codebooks_head.weight"
    assert m.map("depth_decoder.model.norm.weight") == "depth_decoder.norm.weight"
    assert (
        m.map("depth_decoder.model.inputs_embeds_projector.weight")
        == "depth_decoder.inputs_embeds_projector.weight"
    )


def test_prefix_routes() -> None:
    m = CsmWeightMapper()
    assert (
        m.map("backbone_model.layers.3.self_attn.q_proj.weight")
        == "backbone.model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        m.map("depth_decoder.model.layers.0.mlp.gate_proj.weight")
        == "depth_decoder.layers.0.mlp.gate_proj.weight"
    )


def test_deliberate_skips_map_to_none() -> None:
    m = CsmWeightMapper()
    # Tied alias of the audio table (present in the 538-tensor census).
    assert m.map("depth_decoder.model.embed_tokens.weight") is None
    # The Mimi codec subtree is loaded by audio_codec.py, not here.
    assert m.map("codec_model.encoder.layers.0.conv.weight") is None


def test_unexpected_name_raises() -> None:
    m = CsmWeightMapper()
    with pytest.raises(ValueError):
        m.map("totally.unknown.tensor")


def test_census_zero_unexpected() -> None:
    m = CsmWeightMapper()
    names = [
        "embed_text_tokens.weight",
        "lm_head.weight",
        "backbone_model.embed_tokens.embed_audio_tokens.weight",
        "backbone_model.norm.weight",
        "backbone_model.layers.0.self_attn.q_proj.weight",
        "depth_decoder.model.inputs_embeds_projector.weight",
        "depth_decoder.model.norm.weight",
        "depth_decoder.codebooks_head.weight",
        "depth_decoder.model.layers.0.mlp.gate_proj.weight",
        "depth_decoder.model.embed_tokens.weight",  # skip: tied alias
        "codec_model.encoder.conv.weight",  # skip: codec subtree
    ]
    census = m.census(names)
    assert census["unexpected"] == 0
    assert census["skipped"] >= 2
    assert census["mapped"] >= 8
    assert census["mapped"] + census["skipped"] == len(names)
