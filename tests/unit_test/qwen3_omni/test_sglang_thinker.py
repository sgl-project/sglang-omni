# SPDX-License-Identifier: Apache-2.0
"""CPU tests for Qwen3-Omni M-RoPE graph capability plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)

from sglang_omni.models.qwen3_omni.components import (
    sglang_thinker as sglang_thinker_module,
)
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
    _config_uses_mrope,
)
from sglang_omni.models.qwen3_omni.hf_config import (
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(rope_scaling={"mrope_section": [16, 24, 24]}), True),
        (SimpleNamespace(rope_parameters={"mrope_section": [16, 24, 24]}), True),
        (SimpleNamespace(rope_scaling={"rope_type": "default"}), False),
        (SimpleNamespace(), False),
    ],
)
def test_qwen_text_config_declares_mrope_only_for_mrope_sections(config, expected):
    assert _config_uses_mrope(config) is expected


@pytest.mark.parametrize(
    ("rope_scaling", "expected_mrope"),
    [
        (
            {"rope_type": "default", "mrope_section": [16, 24, 24]},
            True,
        ),
        (None, False),
    ],
)
def test_real_qwen_config_drives_prefill_positions(
    monkeypatch: pytest.MonkeyPatch,
    rope_scaling: dict[str, object] | None,
    expected_mrope: bool,
):
    class LightweightTextModel(torch.nn.Module):
        def __init__(self, *, config, quant_config, prefix):
            super().__init__()
            del quant_config, prefix
            self.embed_tokens = torch.nn.Embedding(
                config.vocab_size,
                config.hidden_size,
            )

    class LightweightLogitsProcessor(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    monkeypatch.setattr(
        sglang_thinker_module,
        "Qwen3MoeLLMModel",
        LightweightTextModel,
    )
    monkeypatch.setattr(
        sglang_thinker_module,
        "LogitsProcessor",
        LightweightLogitsProcessor,
    )

    text_config = Qwen3OmniMoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        rope_scaling=rope_scaling,
    )
    root_config = Qwen3OmniMoeThinkerConfig(text_config=text_config)
    wrapper = Qwen3OmniThinkerForCausalLM(root_config)

    assert wrapper.config is text_config
    assert wrapper.is_mrope_enabled is expected_mrope

    runner = object.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(model=wrapper)
    ordinary_positions = torch.arange(4, dtype=torch.long)
    mrope_positions = ordinary_positions.repeat(3, 1)
    forward_batch = SimpleNamespace(
        positions=ordinary_positions,
        mrope_positions=mrope_positions,
    )

    selected = runner._get_layer_model_positions(forward_batch)
    assert selected is (mrope_positions if expected_mrope else ordinary_positions)


def test_outer_thinker_forward_accepts_sidecar_request_identity():
    seen: dict[str, object] = {}

    def fake_model(**kwargs):
        seen.update(kwargs)
        return torch.zeros((2, 3))

    wrapper = object.__new__(Qwen3OmniThinkerForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = fake_model
    wrapper.logits_processor = lambda *args: args[0]
    wrapper.lm_head = object()
    input_ids = torch.tensor([1, 2], dtype=torch.long)
    positions = torch.tensor([0, 1], dtype=torch.long)
    input_embeds = torch.ones((2, 3))
    forward_batch = SimpleNamespace(mrope_positions=None)

    result = wrapper.forward(
        input_ids=input_ids,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        omni_prefill_rids=("request-1",),
    )

    assert result is input_ids
    assert seen["input_embeds"] is input_embeds
    assert seen["positions"] is positions
