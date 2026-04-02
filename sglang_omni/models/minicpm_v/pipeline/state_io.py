# SPDX-License-Identifier: Apache-2.0
"""Helpers to convert between StagePayload.data and PipelineState."""

from __future__ import annotations

from sglang_omni.models.minicpm_v.io import PipelineState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> PipelineState:
    """Load PipelineState from StagePayload.data."""
    return PipelineState.from_dict(payload.data)


def store_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    """Store PipelineState back to StagePayload.data."""
    payload.data = state.to_dict()
    return payload
