# SPDX-License-Identifier: Apache-2.0
"""Helpers to convert between StagePayload.data and S2ProState."""

from __future__ import annotations

from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> S2ProState:
    return S2ProState.from_dict(payload.data)


def store_state(payload: StagePayload, state: S2ProState) -> StagePayload:
    payload.data = state.to_dict()
    return payload
