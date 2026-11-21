# SPDX-License-Identifier: Apache-2.0
"""GPU-free streaming-vocoder clear_stream_state contract."""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.models.csm_tts.vocoder_scheduler import CsmStreamingVocoderScheduler
from sglang_omni.models.csm_tts.vocoder_stream_state import CsmVocoderStreamState


def test_clear_stream_state_is_idempotent() -> None:
    scheduler = CsmStreamingVocoderScheduler(SimpleNamespace())
    scheduler.stream_states["speech-1"] = CsmVocoderStreamState()
    scheduler.clear_stream_state("speech-1")
    assert "speech-1" not in scheduler.stream_states
    scheduler.clear_stream_state("speech-1")
