# SPDX-License-Identifier: Apache-2.0
"""Helpers to convert between StagePayload.data and PipelineState."""

from __future__ import annotations

from sglang_omni.models.ming_omni.io import PipelineState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> PipelineState:
    return PipelineState.from_dict(payload.data)


def store_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    payload.data = state.to_dict()
    return payload
