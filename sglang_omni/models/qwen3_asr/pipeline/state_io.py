# SPDX-License-Identifier: Apache-2.0
"""Serialization helpers for Qwen3-ASR pipeline state."""

from __future__ import annotations

from sglang_omni.models.qwen3_asr.io import PipelineState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> PipelineState:
    """Extract PipelineState from payload.data."""
    return PipelineState.from_dict(payload.data)


def store_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    """Update payload.data with PipelineState and return updated payload."""
    payload.data = state.to_dict()
    return payload
