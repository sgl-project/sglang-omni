# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang_omni.models.llada2_uni.components.thinker import LLaDA2MoeGate


def test_router_accumulates_and_returns_fp32():
    gate = LLaDA2MoeGate(
        SimpleNamespace(num_experts=8, hidden_size=16),
        params_dtype=torch.bfloat16,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        gate.weight.data.normal_()
        hidden = torch.randn(5, 16, dtype=torch.bfloat16)
    result = gate(hidden)
    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result, F.linear(hidden.float(), gate.weight.float()), rtol=0, atol=0
    )


def test_expert_bias_is_checkpoint_loadable_without_config_flag():
    gate = LLaDA2MoeGate(SimpleNamespace(num_experts=8, hidden_size=16))
    assert "expert_bias" in dict(gate.named_buffers())
    assert torch.count_nonzero(gate.expert_bias) == 0
    bias = torch.arange(8, dtype=torch.float32)
    gate.load_state_dict({"weight": torch.zeros_like(gate.weight), "expert_bias": bias})
    torch.testing.assert_close(gate.expert_bias, bias, rtol=0, atol=0)
