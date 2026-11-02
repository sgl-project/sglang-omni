# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang_omni.models.llada2_uni.components.thinker import (
    LLaDA2MoeGate,
    LLaDA2MoeTextModel,
)


def test_router_computes_logits_in_fp32() -> None:
    gate = LLaDA2MoeGate(
        SimpleNamespace(num_experts=8, hidden_size=16),
        params_dtype=torch.bfloat16,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        gate.weight.data.normal_()
        hidden_states = torch.randn(5, 16, dtype=torch.bfloat16)

    result = gate(hidden_states)

    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result,
        F.linear(hidden_states.float(), gate.weight.float()),
        rtol=0,
        atol=0,
    )


def test_expert_bias_loads_without_a_config_flag() -> None:
    gate = LLaDA2MoeGate(SimpleNamespace(num_experts=8, hidden_size=16))

    assert "expert_bias" in dict(gate.named_buffers())
    assert torch.count_nonzero(gate.expert_bias) == 0

    expert_bias = torch.arange(8, dtype=torch.float32)
    model = LLaDA2MoeTextModel.__new__(LLaDA2MoeTextModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=8)
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].mlp = torch.nn.Module()
    model.layers[0].mlp.gate = gate
    model.load_weights([("model.layers.0.mlp.gate.expert_bias", expert_bias)])
    torch.testing.assert_close(gate.expert_bias, expert_bias, rtol=0, atol=0)
