# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from sglang_omni.models.llada2_uni import stages


def test_thinker_factory_propagates_tp_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    bootstrap_module = ModuleType("sglang_omni.models.llada2_uni.bootstrap")

    def fake_create_scheduler(
        server_args,
        gpu_id,
        *,
        tp_rank,
        nccl_port,
    ):
        captured["scheduler"] = {
            "server_args": server_args,
            "gpu_id": gpu_id,
            "tp_rank": tp_rank,
            "nccl_port": nccl_port,
        }
        return "scheduler"

    bootstrap_module.create_dllm_thinker_scheduler = fake_create_scheduler
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.llada2_uni.bootstrap",
        bootstrap_module,
    )

    backend_module = ModuleType("sglang_omni.scheduling.sglang_backend")

    def fake_build_server_args(model_path, **kwargs):
        captured["build_kwargs"] = {"model_path": model_path, **kwargs}
        server_args = SimpleNamespace(
            dllm_algorithm=kwargs["dllm_algorithm"],
            mem_fraction_static=kwargs.get("mem_fraction_static"),
        )
        captured["built_server_args"] = server_args
        return server_args

    backend_module.build_sglang_server_args = fake_build_server_args
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.sglang_backend",
        backend_module,
    )

    result = stages.create_sglang_dllm_thinker_executor_from_config(
        "model",
        gpu_id=3,
        tp_rank=2,
        tp_size=4,
        nccl_port=29500,
    )

    assert result == "scheduler"
    assert captured["build_kwargs"]["tp_size"] == 4
    scheduler_args = captured["scheduler"]
    assert scheduler_args["server_args"] is captured["built_server_args"]
    assert scheduler_args["gpu_id"] == 3
    assert scheduler_args["tp_rank"] == 2
    assert scheduler_args["nccl_port"] == 29500


def test_sparse_moe_disables_inner_tp_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sglang")
    from torch import nn

    from sglang_omni.models.llada2_uni.components import thinker as thinker_module

    captured: dict[str, Any] = {}

    class RoutedExperts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured["routed_reduce_results"] = kwargs["reduce_results"]

    class SharedExperts(nn.Module):
        def __init__(
            self,
            config,
            intermediate_size,
            quant_config=None,
            reduce_results=True,
        ):
            super().__init__()
            captured["shared_reduce_results"] = reduce_results

    monkeypatch.setattr(
        thinker_module,
        "get_moe_impl_class",
        lambda quant_config: RoutedExperts,
    )
    monkeypatch.setattr(thinker_module, "LLaDA2MoeMLP", SharedExperts)

    config = SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        router_dtype=None,
        hidden_size=8,
        moe_router_enable_expert_bias=False,
        moe_intermediate_size=16,
        num_shared_experts=1,
    )
    thinker_module.LLaDA2MoeSparseMoeBlock(config, layer_id=3)

    assert captured == {
        "routed_reduce_results": False,
        "shared_reduce_results": False,
    }


@pytest.mark.parametrize(
    ("tp_size", "with_shared_expert", "expected_collectives"),
    [(1, True, 0), (4, True, 1), (4, False, 1)],
)
def test_sparse_moe_reduces_tp_partials_once(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    with_shared_expert: bool,
    expected_collectives: int,
) -> None:
    pytest.importorskip("sglang")
    import torch
    from torch import nn

    from sglang_omni.models.llada2_uni.components import thinker as thinker_module
    from sglang_omni.models.llada2_uni.components.thinker import (
        LLaDA2MoeSparseMoeBlock,
    )

    class RoutedExperts(nn.Module):
        def forward(self, hidden_states, topk_output):
            return hidden_states * 2

    class SharedExperts(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 3

    class Gate(nn.Module):
        expert_bias = None

        def forward(self, hidden_states):
            return torch.zeros((hidden_states.shape[0], 1))

    block = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.experts = RoutedExperts()
    block.shared_experts = SharedExperts() if with_shared_expert else None
    block.gate = Gate()
    block.num_experts_per_tok = 1
    block.routed_scaling_factor = 1.0
    block._group_limited_topk = lambda scores: (
        torch.ones_like(scores),
        torch.zeros_like(scores, dtype=torch.int64),
    )

    reduced_inputs: list[torch.Tensor] = []
    monkeypatch.setattr(
        thinker_module,
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )

    def fake_all_reduce(value: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(value.clone())
        return value + 1

    monkeypatch.setattr(
        thinker_module,
        "tensor_model_parallel_all_reduce",
        fake_all_reduce,
    )

    hidden_states = torch.randn((6, 8), dtype=torch.bfloat16)
    output = block(hidden_states)

    routed_output = (hidden_states * 2).float()
    expected_partial = (
        routed_output + (hidden_states * 3).float()
        if with_shared_expert
        else routed_output.bfloat16()
    )
    expected_output = expected_partial + (1 if tp_size > 1 else 0)

    assert len(reduced_inputs) == expected_collectives
    if reduced_inputs:
        expected_reduce_dtype = torch.float32 if with_shared_expert else torch.bfloat16
        assert reduced_inputs[0].dtype == expected_reduce_dtype
        torch.testing.assert_close(reduced_inputs[0], expected_partial)
    torch.testing.assert_close(output, expected_output.bfloat16())
