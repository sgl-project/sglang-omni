# SPDX-License-Identifier: Apache-2.0
"""Helpers to convert between StagePayload.data and FishAudioState."""

from __future__ import annotations

from sglang_omni.models.fishaudio_s1.io import FishAudioState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> FishAudioState:
    return FishAudioState.from_dict(payload.data)


def store_state(payload: StagePayload, state: FishAudioState) -> StagePayload:
    payload.data = state.to_dict()
    return payload
