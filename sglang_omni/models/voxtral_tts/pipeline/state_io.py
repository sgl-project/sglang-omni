"""Serialize/deserialize VoxtralTTSState to/from StagePayload.data."""

from __future__ import annotations

from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> VoxtralTTSState:
    return VoxtralTTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: VoxtralTTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload
