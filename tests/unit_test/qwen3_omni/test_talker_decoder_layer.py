# SPDX-License-Identifier: Apache-2.0

"""Construction contract for the talker decoder layer (#2358).

The talker layer must stay "the thinker layer with the sparse MoE class
swapped": no copied construction or forward (fixes land once, in the
thinker), and no thinker MoE block built only to be discarded — the
discarded experts stayed alive until cyclic gc.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import sglang_omni.models.qwen3_omni.components.thinker_model as thinker_module
from sglang_omni.models.qwen3_omni.components.talker import (
    Qwen3OmniMoeTalkerDecoderLayer,
    Qwen3OmniMoeTalkerSparseMoeBlock,
)
from sglang_omni.models.qwen3_omni.components.thinker_model import (
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
)


def fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=128,
        head_dim=8,
        rms_norm_eps=1e-6,
        attention_bias=False,
        num_attention_heads=2,
        num_key_value_heads=1,
        dual_chunk_attention_config=None,
        num_hidden_layers=4,
    )


def test_talker_layer_swaps_only_the_sparse_moe_class() -> None:
    """The class-attribute hook routes each layer family to its own MoE block."""
    assert (
        Qwen3OmniMoeThinkerTextDecoderLayer.sparse_moe_block_cls
        is Qwen3OmniMoeThinkerTextSparseMoeBlock
    )
    assert (
        Qwen3OmniMoeTalkerDecoderLayer.sparse_moe_block_cls
        is Qwen3OmniMoeTalkerSparseMoeBlock
    )
    assert issubclass(
        Qwen3OmniMoeTalkerSparseMoeBlock, Qwen3OmniMoeThinkerTextSparseMoeBlock
    )
    # The talker layer must not grow local copies of the thinker's
    # construction or forward; behavior stays inherited.
    assert (
        Qwen3OmniMoeTalkerDecoderLayer.__init__
        is Qwen3OmniMoeThinkerTextDecoderLayer.__init__
    )
    assert (
        Qwen3OmniMoeTalkerDecoderLayer.forward
        is Qwen3OmniMoeThinkerTextDecoderLayer.forward
    )


def test_talker_layer_builds_exactly_one_talker_moe_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction instantiates the subclass MoE exactly once and never builds
    a thinker MoE block on the side (the pre-#2358 discarded-experts path)."""

    built: list = []

    class RecordingMoe:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            built.append(self)

    class FakeLayerCommunicator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeScatterModes:
        @staticmethod
        def init_new(**kwargs) -> object:
            return object()

    config = fake_config()
    monkeypatch.setattr(
        thinker_module, "Qwen3OmniMoeThinkerTextAttention", lambda **kw: object()
    )
    monkeypatch.setattr(
        thinker_module,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=1, attn_tp_rank=0),
    )
    monkeypatch.setattr(thinker_module, "LayerScatterModes", FakeScatterModes)
    monkeypatch.setattr(thinker_module, "LayerCommunicator", FakeLayerCommunicator)
    monkeypatch.setattr(thinker_module, "RMSNorm", lambda *a, **kw: object())
    monkeypatch.setattr(
        Qwen3OmniMoeTalkerDecoderLayer, "sparse_moe_block_cls", RecordingMoe
    )

    layer = Qwen3OmniMoeTalkerDecoderLayer(config=config, layer_id=1)

    assert len(built) == 1
    assert type(layer.mlp) is RecordingMoe
    assert layer.mlp.kwargs["layer_id"] == 1
    assert layer.mlp.kwargs["config"] is config
    assert layer.mlp.kwargs["prefix"] == "mlp"
