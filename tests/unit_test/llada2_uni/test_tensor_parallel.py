# SPDX-License-Identifier: Apache-2.0
"""LLaDA2-Uni tensor-parallel MoE reduction and thinker factory wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.arg_groups import model_override_base
from torch import nn

from sglang_omni.models.llada2_uni import bootstrap, stages
from sglang_omni.models.llada2_uni.components import thinker
from sglang_omni.platforms.xpu import XPUOmniPlatform
from sglang_omni.scheduling import sglang_backend
from sglang_omni.vendor.sglang.layers import QuantizationConfig, StandardTopKOutput

TP_RANK_COUNT = 2


class ColumnLinearStandIn(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool,
        quant_config: QuantizationConfig | None,
    ) -> None:
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states, None


class RowLinearStandIn(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        quant_config: QuantizationConfig | None,
        reduce_results: bool,
    ) -> None:
        super().__init__()
        self.reduce_results = reduce_results

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self.reduce_results:
            return TP_RANK_COUNT * hidden_states, None
        else:
            return hidden_states, None


class RoutedExpertsStandIn(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        reduce_results: bool,
    ) -> None:
        super().__init__()

    def forward(
        self, hidden_states: torch.Tensor, topk_output: StandardTopKOutput
    ) -> torch.Tensor:
        return torch.full_like(hidden_states, 3.0)


def test_moe_block_reduces_routed_plus_shared_output_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        thinker, "get_moe_impl_class", lambda quant_config: RoutedExpertsStandIn
    )
    monkeypatch.setattr(thinker, "MergedColumnParallelLinear", ColumnLinearStandIn)
    monkeypatch.setattr(thinker, "RowParallelLinear", RowLinearStandIn)
    monkeypatch.setattr(thinker, "SiluAndMul", nn.Identity)
    monkeypatch.setattr(
        thinker,
        "reduce_moe_output",
        lambda hidden_states: TP_RANK_COUNT * hidden_states,
    )
    config = SimpleNamespace(
        hidden_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        moe_intermediate_size=4,
        num_shared_experts=1,
    )

    block = thinker.LLaDA2MoeSparseMoeBlock(config, layer_id=1)
    output = block(torch.ones(3, 4))

    assert torch.equal(output, torch.full((3, 4), (3.0 + 1.0) * TP_RANK_COUNT))


def test_thinker_factory_passes_the_platform_backend_and_tp_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_args_kwargs = {}
    scheduler_kwargs = {}

    def build_server_args(
        model_path: str,
        *,
        context_length: int,
        dllm_algorithm: str,
        dllm_algorithm_config: str | None,
        **overrides: str | bool | int,
    ) -> None:
        server_args_kwargs.update(overrides)

    def create_scheduler(
        server_args: None,
        gpu_id: int,
        *,
        tp_rank: int,
        nccl_port: int | None,
        total_gpu_memory_fraction: float | None,
    ) -> None:
        scheduler_kwargs.update(
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
        )

    monkeypatch.setattr(stages, "current_platform", XPUOmniPlatform())
    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", build_server_args)
    monkeypatch.setattr(
        model_override_base,
        "resolved_view",
        lambda server_args: SimpleNamespace(
            dllm_algorithm=None, mem_fraction_static=None
        ),
    )
    monkeypatch.setattr(bootstrap, "create_dllm_thinker_scheduler", create_scheduler)

    stages.create_sglang_dllm_thinker_executor_from_config(
        "model",
        device="cpu",
        tp_rank=1,
        tp_size=2,
        nccl_port=29500,
        total_gpu_memory_fraction=0.8,
    )

    assert server_args_kwargs["attention_backend"] == "triton"
    assert server_args_kwargs["tp_size"] == 2
    assert scheduler_kwargs == {
        "tp_rank": 1,
        "nccl_port": 29500,
        "total_gpu_memory_fraction": 0.8,
    }
