# SPDX-License-Identifier: Apache-2.0
"""GPU-free streaming-vocoder abort / knob contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.csm_tts.vocoder_scheduler import (
    DEFAULT_CSM_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_CSM_STREAM_STRIDE,
    CsmStreamingVocoderScheduler,
    CsmStreamState,
)


def _scheduler() -> CsmStreamingVocoderScheduler:
    return CsmStreamingVocoderScheduler(SimpleNamespace())


def test_stream_knob_defaults_match_cookbook() -> None:
    scheduler = _scheduler()
    assert scheduler._stream_stride == DEFAULT_CSM_STREAM_STRIDE == 13
    assert scheduler._stream_followup_stride == DEFAULT_CSM_STREAM_FOLLOWUP_STRIDE == 6


def test_clear_stream_state_is_idempotent() -> None:
    scheduler = _scheduler()
    scheduler._stream_states["req"] = CsmStreamState()
    scheduler.clear_stream_state("req")
    assert "req" not in scheduler._stream_states
    scheduler.clear_stream_state("req")


def test_invalid_stream_stride_is_rejected() -> None:
    with pytest.raises(ValueError, match="stream_stride"):
        CsmStreamingVocoderScheduler(SimpleNamespace(), stream_stride=0)
