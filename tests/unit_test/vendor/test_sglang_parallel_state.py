# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pytest

from sglang_omni.vendor.sglang.parallel_state import create_parallel_state


@dataclass(kw_only=True)
class _LegacyParallelState:
    tp_rank: int
    dcp_size: int
    marker: str


@dataclass(kw_only=True)
class _CurrentParallelState:
    tp_rank: int
    attn_dcp_rank: int
    attn_dcp_size: int
    marker: str


@dataclass(kw_only=True)
class _UnsupportedParallelState:
    tp_rank: int


def test_create_parallel_state_supports_legacy_dcp_size() -> None:
    state = create_parallel_state(
        _LegacyParallelState,
        tp_rank=3,
        dcp_size=2,
        marker="legacy",
    )

    assert state == _LegacyParallelState(tp_rank=3, dcp_size=2, marker="legacy")


def test_create_parallel_state_supports_current_attention_dcp_fields() -> None:
    state = create_parallel_state(
        _CurrentParallelState,
        tp_rank=3,
        dcp_size=2,
        marker="current",
    )

    assert state == _CurrentParallelState(
        tp_rank=3,
        attn_dcp_rank=1,
        attn_dcp_size=2,
        marker="current",
    )


def test_create_parallel_state_rejects_unknown_layout() -> None:
    with pytest.raises(TypeError, match="Unsupported SGLang ParallelState"):
        create_parallel_state(
            _UnsupportedParallelState,
            tp_rank=0,
            dcp_size=1,
        )
