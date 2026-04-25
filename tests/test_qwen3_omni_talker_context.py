# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Qwen3-Omni Talker context validation."""

from __future__ import annotations

import pytest

from sglang_omni.models.qwen3_omni.components.talker_executor import (
    _validate_talker_context,
)


def test_talker_context_validation_reserves_generation_room() -> None:
    with pytest.raises(ValueError, match="8 total tokens"):
        _validate_talker_context(
            request_id="req",
            prefill_tokens=6,
            max_new_tokens=2,
            max_seq_len=8,
        )
