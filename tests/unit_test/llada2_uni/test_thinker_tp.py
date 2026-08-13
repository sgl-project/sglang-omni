# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch import nn

from sglang_omni.models.llada2_uni.components import thinker


@pytest.mark.parametrize(
    ("tp_size", "expected"),
    [
        (1, (16, 4)),
        (2, (8, 2)),
        (4, (4, 1)),
        (8, (2, 1)),
        (16, (1, 1)),
    ],
)
def test_attention_head_partitioning(tp_size, expected) -> None:
    assert thinker._get_local_attention_head_counts(16, 4, tp_size) == expected


@pytest.mark.parametrize("tp_size", [0, 3, 6, 32])
def test_attention_head_partitioning_rejects_invalid_tp_size(tp_size) -> None:
    with pytest.raises(ValueError):
        thinker._get_local_attention_head_counts(16, 4, tp_size)


class _Gate(nn.Module):
    expert_bias = None

    def forward(self, hidden_states):
        return torch.zeros(
            hidden_states.shape[0],
            2,
            dtype=hidden_states.dtype,
        )


class _Experts(nn.Module):
    def forward(self, hidden_states, topk_output):
        return hidden_states + 1


class _SharedExperts(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 2


def test_sparse_moe_reduces_combined_routed_and_shared_output(
    monkeypatch,
) -> None:
    block = object.__new__(thinker.LLaDA2MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.tp_size = 2
    block.num_experts_per_tok = 1
    block.routed_scaling_factor = 1.0
    block.gate = _Gate()
    block.experts = _Experts()
    block.shared_experts = _SharedExperts()
    block._group_limited_topk = MethodType(
        lambda self, scores: (
            torch.ones(scores.shape[0], 1),
            torch.zeros(scores.shape[0], 1, dtype=torch.long),
        ),
        block,
    )
    reduced = []

    monkeypatch.setattr(
        thinker,
        "should_skip_post_experts_all_reduce",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        thinker,
        "tensor_model_parallel_all_reduce",
        lambda value: reduced.append(value.clone()) or value * 2,
    )

    output = block(torch.tensor([[3.0, 4.0]]))

    assert len(reduced) == 1
    assert torch.equal(reduced[0], torch.tensor([[9.0, 11.0]]))
    assert torch.equal(output, torch.tensor([[18.0, 22.0]]))
