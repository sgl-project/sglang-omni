# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang_omni.models.llada2_uni.components import thinker
from sglang_omni.models.llada2_uni.components.thinker import LLaDA2MoeGate
from sglang_omni.vendor.sglang.layers import StandardTopKOutput


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
    gate.load_state_dict(
        {"weight": torch.zeros_like(gate.weight), "expert_bias": expert_bias}
    )
    torch.testing.assert_close(gate.expert_bias, expert_bias, rtol=0, atol=0)


def test_moe_expert_projections_return_rank_partials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed_experts = Mock(return_value=nn.Identity())
    down_projection = Mock(return_value=nn.Identity())
    monkeypatch.setattr(thinker, "get_moe_impl_class", lambda config: routed_experts)
    monkeypatch.setattr(
        thinker, "MergedColumnParallelLinear", Mock(return_value=nn.Identity())
    )
    monkeypatch.setattr(thinker, "RowParallelLinear", down_projection)
    config = PretrainedConfig(
        num_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        hidden_size=8,
        moe_intermediate_size=16,
        num_shared_experts=1,
    )

    thinker.LLaDA2MoeSparseMoeBlock(config, layer_id=0)

    assert routed_experts.call_args.kwargs["reduce_results"] is False
    assert down_projection.call_args.kwargs["reduce_results"] is False


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("with_shared_expert", [False, True])
def test_moe_combines_rank_partials_before_cast(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, with_shared_expert: bool
) -> None:
    class RoutedExperts(nn.Module):
        def forward(
            self, hidden_states: torch.Tensor, topk_output: StandardTopKOutput
        ) -> torch.Tensor:
            return hidden_states.fill_(256)

    block = thinker.LLaDA2MoeSparseMoeBlock.__new__(thinker.LLaDA2MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.gate = LLaDA2MoeGate(PretrainedConfig(num_experts=2, hidden_size=2))
    block.gate.weight.data.zero_()
    block.topk = Mock(return_value=None)
    block.experts = RoutedExperts()
    block.shared_experts = nn.Identity() if with_shared_expert else None
    block.num_experts = 2
    block.num_experts_per_tok = 1
    block.n_group = 1
    block.topk_group = 1
    block.routed_scaling_factor = 1.0
    monkeypatch.setattr(
        thinker, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    reduced_inputs: list[torch.Tensor] = []

    def all_reduce(value: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(value.clone())
        return value - 256

    monkeypatch.setattr(thinker, "tensor_model_parallel_all_reduce", all_reduce)
    output = block(torch.ones((3, 2), dtype=torch.bfloat16))

    assert len(reduced_inputs) == (tp_size > 1)
    if tp_size > 1:
        expected_dtype = torch.float32 if with_shared_expert else torch.bfloat16
        assert reduced_inputs[0].dtype == expected_dtype
        torch.testing.assert_close(
            reduced_inputs[0],
            torch.full(
                (3, 2), 257 if with_shared_expert else 256, dtype=expected_dtype
            ),
            rtol=0,
            atol=0,
        )
    expected = int(with_shared_expert) if tp_size > 1 else 256
    torch.testing.assert_close(
        output, torch.full((3, 2), expected, dtype=torch.bfloat16), rtol=0, atol=0
    )
