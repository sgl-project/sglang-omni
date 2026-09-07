# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the native sglang talker: weight mapping and condition math."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
    _MiniCPMTTSProjector,
)

HIDDEN = 8
LLM_DIM = 16
NUM_AUDIO = 12
NUM_TEXT = 20
TEXT_EOS = 5
AUDIO_BOS = 6


class _RecordingBackbone:
    def __init__(self):
        self.received: list[tuple[str, torch.Tensor]] = []

    def load_weights(self, weights):
        self.received.extend(weights)


def _bare_model() -> MiniCPMOTalkerForCausalLM:
    model = object.__new__(MiniCPMOTalkerForCausalLM)
    nn.Module.__init__(model)
    model.num_audio_tokens = NUM_AUDIO
    model.codec_eos_id = NUM_AUDIO - 1
    model.text_eos_token_id = TEXT_EOS
    model.audio_bos_token_id = AUDIO_BOS
    model.normalize_projected_hidden = True
    model.llama = _RecordingBackbone()
    model.emb_text = nn.Embedding(NUM_TEXT, HIDDEN)
    model.projector_semantic = _MiniCPMTTSProjector(LLM_DIM, HIDDEN)
    model.emb_code = nn.Embedding(NUM_AUDIO, HIDDEN)
    model.head_code = nn.Linear(HIDDEN, NUM_AUDIO, bias=False)
    return model


def _checkpoint_weights():
    g = torch.rand(NUM_AUDIO, 1) + 0.5
    v = torch.randn(NUM_AUDIO, HIDDEN)
    return {
        "tts.model.embed_tokens.weight": torch.randn(NUM_AUDIO, HIDDEN),
        "tts.model.norm.weight": torch.randn(HIDDEN),
        "tts.emb_text.weight": torch.randn(NUM_TEXT, HIDDEN),
        "tts.emb_code.0.weight": torch.randn(NUM_AUDIO, HIDDEN),
        "tts.head_code.0.parametrizations.weight.original0": g,
        "tts.head_code.0.parametrizations.weight.original1": v,
        "tts.projector_semantic.linear1.weight": torch.randn(HIDDEN, LLM_DIM),
        "tts.projector_semantic.linear1.bias": torch.randn(HIDDEN),
        "tts.projector_semantic.linear2.weight": torch.randn(HIDDEN, HIDDEN),
        "tts.projector_semantic.linear2.bias": torch.randn(HIDDEN),
        "tts.projector_spk.linear1.weight": torch.randn(HIDDEN, LLM_DIM),
        "llm.model.embed_tokens.weight": torch.randn(4, 4),
    }


def test_load_weights_mapping_and_weight_norm_collapse():
    model = _bare_model()
    ckpt = _checkpoint_weights()
    loaded = model.load_weights(list(ckpt.items()))

    # backbone gets tts.model.* with only the tts. prefix stripped
    backbone_names = [name for name, _ in model.llama.received]
    assert backbone_names == ["model.embed_tokens.weight", "model.norm.weight"]

    torch.testing.assert_close(model.emb_text.weight, ckpt["tts.emb_text.weight"])
    torch.testing.assert_close(model.emb_code.weight, ckpt["tts.emb_code.0.weight"])
    torch.testing.assert_close(
        model.projector_semantic.linear1.weight,
        ckpt["tts.projector_semantic.linear1.weight"],
    )

    # weight_norm collapse: g * v / ||v|| row-wise (dim=0)
    g = ckpt["tts.head_code.0.parametrizations.weight.original0"]
    v = ckpt["tts.head_code.0.parametrizations.weight.original1"]
    expected = g * v / v.norm(dim=1, keepdim=True)
    torch.testing.assert_close(model.head_code.weight, expected)

    assert "head_code.weight" in loaded
    assert "emb_code.weight" in loaded
    # projector_spk and non-tts weights are skipped, not loaded
    assert not any("projector_spk" in name for name in loaded)


def test_load_weights_missing_head_norm_raises():
    model = _bare_model()
    ckpt = _checkpoint_weights()
    del ckpt["tts.head_code.0.parametrizations.weight.original0"]
    with pytest.raises(ValueError, match="weight-norm"):
        model.load_weights(list(ckpt.items()))


def test_condition_matches_reference_math():
    model = _bare_model()
    tokens = torch.tensor([3, 7, 1], dtype=torch.long)
    hidden = torch.randn(3, LLM_DIM)

    condition = model.build_condition_embeddings(tokens, hidden)

    # reference: emb_text(t) + l2norm(projector(h)), then [text_eos, audio_bos]
    ref = model.emb_text(tokens) + F.normalize(
        model.projector_semantic(hidden), p=2, dim=-1
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, torch.cat([ref, boundary], dim=0))
    assert condition.shape == (5, HIDDEN)


def test_condition_empty_span_is_boundary_only():
    model = _bare_model()
    condition = model.build_condition_embeddings(
        torch.empty(0, dtype=torch.long), torch.empty(0, LLM_DIM)
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, boundary)


def test_condition_length_mismatch_raises():
    model = _bare_model()
    with pytest.raises(ValueError, match="length mismatch"):
        model.build_condition_embeddings(torch.tensor([1, 2]), torch.randn(3, LLM_DIM))
