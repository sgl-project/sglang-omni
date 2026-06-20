# SPDX-License-Identifier: Apache-2.0
"""RL rollout request-data helpers."""

from __future__ import annotations

import math

import torch

from sglang_omni.scheduling.types import sampled_logprobs_to_list


def test_sampled_logprobs_to_list_preserves_sampler_values() -> None:
    logits = torch.tensor([[2.0, 1.0]])
    raw_logprob = torch.log_softmax(logits, dim=-1)[0, 0].item()
    sampled_logprob = torch.log_softmax(logits / 0.5, dim=-1)[0, 0].item()

    out = sampled_logprobs_to_list(torch.tensor([sampled_logprob]))

    assert out is not None
    assert not math.isclose(raw_logprob, sampled_logprob, abs_tol=1e-4)
    assert math.isclose(out[0], sampled_logprob, abs_tol=1e-4)


def test_sampled_logprobs_to_list_handles_cpu_lists() -> None:
    out = sampled_logprobs_to_list([-0.1, -0.2])
    assert out == [-0.1, -0.2]


def test_none_inputs_return_none() -> None:
    assert sampled_logprobs_to_list(None) is None
