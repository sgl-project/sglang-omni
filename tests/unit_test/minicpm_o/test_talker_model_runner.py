# SPDX-License-Identifier: Apache-2.0
"""Talker repetition penalty: windowed frequency penalty per request."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.minicpm_o.talker_model_runner import MiniCPMOTalkerModelRunner


def scheduler_request(rep_penalty: float, output_ids: list[int]) -> SimpleNamespace:
    request = SimpleNamespace(output_ids=output_ids)
    return SimpleNamespace(
        data=SimpleNamespace(
            talker_model_inputs={"rep_penalty": rep_penalty}, req=request
        )
    )


def test_rep_penalty_counts_window_tokens_per_request() -> None:
    logits = torch.tensor([[8.0, -8.0, 8.0, 8.0]]).repeat(4, 1)
    requests = [
        scheduler_request(2.0, [3] + [9] * 15 + [0, 0, 1, 1]),
        scheduler_request(4.0, [2]),
        scheduler_request(1.0, [0]),
        scheduler_request(2.0, [-1, 99]),
    ]
    runner = object.__new__(MiniCPMOTalkerModelRunner)
    runner.process_sampling_logits(SimpleNamespace(next_token_logits=logits), requests)
    assert logits.tolist() == [
        [2, -32, 8, 8],
        [8, -8, 2, 8],
        [8, -8, 8, 8],
        [8, -8, 8, 8],
    ]
