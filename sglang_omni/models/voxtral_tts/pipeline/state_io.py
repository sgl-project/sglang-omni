"""Serialize/deserialize VoxtralTTSState to/from StagePayload.data."""

from __future__ import annotations

from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state


def load_state(payload: StagePayload) -> VoxtralTTSState:
    return _load_pipeline_state(payload, VoxtralTTSState)


def store_state(payload: StagePayload, state: VoxtralTTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)
