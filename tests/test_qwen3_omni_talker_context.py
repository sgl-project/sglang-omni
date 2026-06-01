# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3-Omni Talker context validation."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.talker_executor import (
    TalkerStreamingExecutor,
)


def test_talker_context_validation_reserves_generation_room() -> None:
    executor = object.__new__(TalkerStreamingExecutor)
    executor._max_seq_len = 8

    with pytest.raises(ValueError, match="8 total tokens"):
        executor._resolve_effective_max_new_tokens(
            request_id="req",
            input_ids=torch.arange(6),
            max_new_tokens=2,
            explicit=True,
        )


def test_talker_context_bounds_default_generation_room() -> None:
    executor = object.__new__(TalkerStreamingExecutor)
    executor._max_seq_len = 8

    max_new_tokens = executor._resolve_effective_max_new_tokens(
        request_id="req",
        input_ids=torch.arange(6),
        max_new_tokens=2,
        explicit=False,
    )

    assert max_new_tokens == 1
