# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.llada2_uni.components import thinker

HIDDEN = 4
INTERMEDIATE = 6
NUM_EXPERTS = 8
TOP_K = 2

ROUTED = 3.0
SHARED = 5.0
REDUCE_MARK = 100.0


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=HIDDEN,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=INTERMEDIATE,
        num_shared_experts=1,
        routed_scaling_factor=2.5,
        moe_router_enable_expert_bias=False,
        router_dtype=None,
    )


class _FakeExperts(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, hidden_states, topk_output):
        del topk_output
        return torch.full_like(hidden_states, ROUTED)


class _FakeColumnParallel(nn.Module):
    def __init__(self, input_size, output_sizes, **kwargs) -> None:
        super().__init__()
        del input_size
        self.width = sum(output_sizes)

    def forward(self, hidden_states):
        shape = (*hidden_states.shape[:-1], self.width)
        return hidden_states.new_full(shape, 1.0), None


class _FakeRowParallel(nn.Module):

    def __init__(
        self, input_size, output_size, *, reduce_results=True, **kwargs
    ) -> None:
        super().__init__()
        del input_size
        self.output_size = output_size
        self.reduce_results = reduce_results

    def forward(self, hidden_states):
        shape = (*hidden_states.shape[:-1], self.output_size)
        return hidden_states.new_full(shape, SHARED), None


class _FakeSiluAndMul(nn.Module):
    def forward(self, hidden_states):
        return hidden_states.chunk(2, dim=-1)[0]


def _build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tp_size: int,
    skip_all_reduce: bool = False,
) -> tuple[thinker.LLaDA2MoeSparseMoeBlock, list[torch.Tensor], list[bool]]:
    reduced: list[torch.Tensor] = []
    consulted: list[bool] = []

    def fake_all_reduce(tensor):
        reduced.append(tensor.clone())
        return tensor + REDUCE_MARK

    def fake_skip(*, is_tp_path):
        consulted.append(is_tp_path)
        return skip_all_reduce

    monkeypatch.setattr(
        thinker, "get_moe_impl_class", lambda quant_config: _FakeExperts
    )
    monkeypatch.setattr(thinker, "MergedColumnParallelLinear", _FakeColumnParallel)
    monkeypatch.setattr(thinker, "RowParallelLinear", _FakeRowParallel)
    monkeypatch.setattr(thinker, "SiluAndMul", _FakeSiluAndMul)
    monkeypatch.setattr(
        thinker, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(thinker, "tensor_model_parallel_all_reduce", fake_all_reduce)
    monkeypatch.setattr(thinker, "should_skip_post_experts_all_reduce", fake_skip)

    block = thinker.LLaDA2MoeSparseMoeBlock(_config(), layer_id=1)
    with torch.no_grad():
        block.gate.weight.fill_(0.1)
    return block, reduced, consulted


def test_the_moe_block_all_reduces_its_partial_sum_under_tp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block, reduced, consulted = _build(monkeypatch, tp_size=2)

    out = block(torch.ones(3, HIDDEN))

    assert len(reduced) == 1, "the combined output must be reduced exactly once"
    assert consulted == [True], "the TP path's skip conditions must be consulted"
    assert torch.equal(reduced[0], torch.full((3, HIDDEN), ROUTED + SHARED))
    assert torch.equal(out, torch.full((3, HIDDEN), ROUTED + SHARED + REDUCE_MARK))


def test_the_shared_expert_does_not_reduce_its_own_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block, _, _ = _build(monkeypatch, tp_size=2)

    assert block.shared_experts.down_proj.reduce_results is False
    assert block.experts.kwargs["reduce_results"] is False


def test_a_single_rank_owns_every_expert_and_stays_off_the_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block, reduced, consulted = _build(monkeypatch, tp_size=1)

    out = block(torch.ones(3, HIDDEN))

    assert reduced == []
    assert consulted == []
    assert torch.equal(out, torch.full((3, HIDDEN), ROUTED + SHARED))


def test_a_downstream_reduce_suppresses_this_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block, reduced, _ = _build(monkeypatch, tp_size=2, skip_all_reduce=True)

    out = block(torch.ones(3, HIDDEN))

    assert reduced == []
    assert torch.equal(out, torch.full((3, HIDDEN), ROUTED + SHARED))
