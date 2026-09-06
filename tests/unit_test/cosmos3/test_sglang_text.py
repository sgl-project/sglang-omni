# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from torch import nn

from sglang_omni.models.cosmos3.components import sglang_text
from sglang_omni.models.cosmos3.components.sglang_text import (
    Cosmos3TextForCausalLM,
    map_cosmos3_text_weight_name,
)


def test_maps_cosmos_attention_names_to_sglang_qwen3() -> None:
    assert (
        map_cosmos3_text_weight_name("layers.0.self_attn.to_q.weight")
        == "layers.0.self_attn.q_proj.weight"
    )
    assert (
        map_cosmos3_text_weight_name("layers.0.self_attn.to_out.weight")
        == "layers.0.self_attn.o_proj.weight"
    )
    assert (
        map_cosmos3_text_weight_name("layers.0.self_attn.norm_k.weight")
        == "layers.0.self_attn.k_norm.weight"
    )
    assert map_cosmos3_text_weight_name("embed_tokens.weight") == (
        "embed_tokens.weight"
    )
    assert map_cosmos3_text_weight_name("model.norm.weight") == "norm.weight"


def test_filters_gen_and_vision_weights_from_text_model() -> None:
    assert map_cosmos3_text_weight_name("layers.0.mlp_moe_gen.gate_proj.weight") is None
    assert map_cosmos3_text_weight_name("layers.0.self_attn.add_q_proj.weight") is None
    assert map_cosmos3_text_weight_name("layers.0.self_attn.to_add_out.weight") is None
    assert map_cosmos3_text_weight_name("norm_moe_gen.weight") is None
    assert map_cosmos3_text_weight_name("blocks.0.attn.qkv.weight") is None


def test_load_weights_forwards_only_mapped_text_tensors(monkeypatch) -> None:
    captured: list[tuple[str, torch.Tensor]] = []

    def fake_load_weights(self, weights):
        del self
        captured.extend(weights)
        return {name for name, _ in captured}

    monkeypatch.setattr(Qwen3ForCausalLM, "load_weights", fake_load_weights)
    model = object.__new__(Cosmos3TextForCausalLM)
    tensor = torch.ones(1)

    result = Cosmos3TextForCausalLM.load_weights(
        model,
        [
            ("layers.0.self_attn.to_q.weight", tensor),
            ("layers.0.self_attn.add_q_proj.weight", tensor),
            ("blocks.0.attn.qkv.weight", tensor),
            ("lm_head.weight", tensor),
        ],
    )

    assert [name for name, _ in captured] == [
        "layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ]
    assert result == {
        "layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    }


def test_text_wrapper_uses_nested_text_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeTextModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(2, 2)

    def fake_init(*, config, quant_config=None, prefix=""):
        captured.update(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )
        return _FakeTextModel()

    monkeypatch.setattr(sglang_text, "Qwen3LLMModel", fake_init)
    monkeypatch.setattr(
        sglang_text,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True, world_size=1),
    )
    monkeypatch.setattr(
        sglang_text, "ParallelLMHead", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(sglang_text, "LogitsProcessor", lambda config: object())
    monkeypatch.setattr(sglang_text, "Pooler", lambda **kwargs: object())
    monkeypatch.setattr(
        sglang_text,
        "get_server_args",
        lambda: SimpleNamespace(enable_dp_lm_head=False),
    )
    text_config = SimpleNamespace(
        tie_word_embeddings=True,
        vocab_size=10,
        hidden_size=4,
    )
    root_config = SimpleNamespace(
        text_config=text_config,
        vision_config=SimpleNamespace(deepstack_visual_indexes=[1, 2, 3]),
        tie_word_embeddings=False,
    )

    model = Cosmos3TextForCausalLM(root_config, quant_config="quant", prefix="p")

    assert captured["config"] is text_config
    assert captured["quant_config"] == "quant"
    assert captured["prefix"] == "p.model"
    assert text_config.tie_word_embeddings is False
    assert model.root_config is root_config


def test_text_forward_prefers_scheduler_mrope_positions(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_forward(self, **kwargs):
        del self
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(Qwen3ForCausalLM, "forward", fake_forward)
    model = object.__new__(Cosmos3TextForCausalLM)
    mrope_positions = torch.tensor(
        [
            [4, 5],
            [4, 5],
            [4, 5],
        ],
        dtype=torch.long,
    )
    forward_batch = SimpleNamespace(mrope_positions=mrope_positions)

    result = Cosmos3TextForCausalLM.forward(
        model,
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([4, 5]),
        forward_batch=forward_batch,
    )

    assert result == "output"
    assert captured["positions"] is mrope_positions
    assert captured["forward_batch"] is forward_batch


def test_multimodal_forward_passes_deepstack_to_qwen3_vl_text_model() -> None:
    captured: dict[str, object] = {}

    class _TextModel(nn.Module):
        def forward(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return torch.ones((2, 4))

    model = object.__new__(Cosmos3TextForCausalLM)
    nn.Module.__init__(model)
    model.model = _TextModel()
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.capture_aux_hidden_states = False
    model.lm_head = nn.Identity()
    model.logits_processor = lambda *args: "logits"
    deepstack = torch.ones((2, 12))
    positions = torch.arange(2).unsqueeze(0).expand(3, -1)
    forward_batch = SimpleNamespace(mrope_positions=positions)

    result = model.forward(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        forward_batch=forward_batch,
        input_embeds=torch.ones((2, 4)),
        input_deepstack_embeds=deepstack,
    )

    assert result == "logits"
    assert captured["kwargs"]["input_deepstack_embeds"] is deepstack
    assert captured["args"][1] is positions
