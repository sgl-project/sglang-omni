# SPDX-License-Identifier: Apache-2.0
"""Checkpoint tensors land on the right Llama and Moshi parameters, without a GPU."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.personaplex.architecture import (
    AUDIO_CARD,
    NUM_AUDIO_STREAMS,
    NUM_STREAMS,
    TEMPORAL_TRANSFORMER,
    TEXT_CARD,
)
from sglang_omni.models.personaplex.sglang_model import (
    PersonaPlexForCausalLM,
    backbone_weight,
)

DIM = 4


def test_backbone_tensors_map_onto_llama_names():
    in_proj = torch.arange(12.0).view(6, 2)
    mapped = dict(
        backbone_weight("transformer.layers.3.self_attn.in_proj_weight", in_proj)
    )
    q, k, v = in_proj.chunk(3)
    assert torch.equal(mapped["model.layers.3.self_attn.q_proj.weight"], q)
    assert torch.equal(mapped["model.layers.3.self_attn.k_proj.weight"], k)
    assert torch.equal(mapped["model.layers.3.self_attn.v_proj.weight"], v)

    linear_in = torch.arange(8.0).view(4, 2)
    mapped = dict(
        backbone_weight("transformer.layers.0.gating.linear_in.weight", linear_in)
    )
    assert torch.equal(mapped["model.layers.0.mlp.gate_proj.weight"], linear_in[:2])
    assert torch.equal(mapped["model.layers.0.mlp.up_proj.weight"], linear_in[2:])

    alpha = torch.ones(1, 1, DIM)
    ((name, tensor),) = backbone_weight("transformer.layers.1.norm2.alpha", alpha)
    assert name == "model.layers.1.post_attention_layernorm.weight"
    assert tensor.shape == (DIM,)
    ((name, _),) = backbone_weight(
        "transformer.layers.1.self_attn.out_proj.weight", torch.zeros(DIM, DIM)
    )
    assert name == "model.layers.1.self_attn.o_proj.weight"
    with pytest.raises(KeyError):
        list(backbone_weight("transformer.layers.1.unknown.weight", alpha))


def fake_model() -> SimpleNamespace:
    loaded = SimpleNamespace(backbone=None, depformer=None)
    return SimpleNamespace(
        loaded=loaded,
        temporal=TEMPORAL_TRANSFORMER,
        llm=SimpleNamespace(load_weights=lambda w: setattr(loaded, "backbone", w)),
        audio_emb=nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, DIM) for _ in range(NUM_AUDIO_STREAMS)
        ),
        text_emb=nn.Embedding(TEXT_CARD + 1, DIM),
        depformer=SimpleNamespace(
            load_reference_weights=lambda w: setattr(loaded, "depformer", w)
        ),
    )


def test_load_weights_routes_every_checkpoint_group():
    model = fake_model()
    text_emb = torch.randn(TEXT_CARD + 1, DIM)
    audio = {f"emb.{k}.weight": torch.randn(AUDIO_CARD + 1, DIM) for k in range(16)}
    weights = {
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(DIM, DIM),
        "out_norm.alpha": torch.ones(1, 1, DIM),
        "text_linear.weight": torch.randn(TEXT_CARD, DIM),
        "text_emb.weight": text_emb,
        "depformer_in.0.weight": torch.zeros(1),
        "linears.0.weight": torch.zeros(1),
        **audio,
    }

    PersonaPlexForCausalLM.load_weights(model, weights.items())

    backbone = dict(model.loaded.backbone)
    assert set(backbone) == {
        "model.layers.0.self_attn.o_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    }
    assert torch.equal(backbone["model.embed_tokens.weight"], text_emb[:TEXT_CARD])
    assert torch.equal(model.text_emb.weight, text_emb)
    assert torch.equal(model.audio_emb[5].weight, audio["emb.5.weight"])
    assert set(model.loaded.depformer) == {"depformer_in.0.weight", "linears.0.weight"}

    with pytest.raises(KeyError, match="unexpected PersonaPlex tensor"):
        PersonaPlexForCausalLM.load_weights(fake_model(), [("mystery", torch.zeros(1))])


def test_embed_rows_reads_text_from_column_zero_and_codebook_k_from_column_k_plus_one():
    model = SimpleNamespace(
        audio_emb=nn.ModuleList(
            nn.Embedding(AUDIO_CARD + 1, NUM_STREAMS) for _ in range(NUM_AUDIO_STREAMS)
        ),
        text_emb=nn.Embedding(TEXT_CARD + 1, NUM_STREAMS),
    )
    with torch.no_grad():
        for k, table in enumerate(model.audio_emb):
            table.weight.zero_()
            table.weight[:, 1 + k] = torch.arange(AUDIO_CARD + 1, dtype=torch.float32)
        model.text_emb.weight.zero_()
        model.text_emb.weight[:, 0] = torch.arange(TEXT_CARD + 1, dtype=torch.float32)

    rows = torch.randint(0, AUDIO_CARD + 1, (3, NUM_STREAMS))
    rows[:, 0] = torch.tensor([5, 31999, 32000])
    with torch.no_grad():
        embedded = PersonaPlexForCausalLM.embed_rows(model, rows)
    assert torch.equal(embedded, rows.float())


def test_sliding_window_covers_the_reference_context():
    model = SimpleNamespace(temporal=TEMPORAL_TRANSFORMER)
    window = PersonaPlexForCausalLM.get_attention_sliding_window_size(model)
    assert window == TEMPORAL_TRANSFORMER.context - 1
