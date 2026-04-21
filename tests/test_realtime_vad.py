# SPDX-License-Identifier: Apache-2.0

import numpy as np

from sglang_omni.realtime.vad import EnergyVad, VadConfig


def _dummy_chunk(num_samples: int = 3200) -> np.ndarray:
    return np.zeros(num_samples, dtype=np.float32)


def test_webrtc_vad_detects_start_and_stop_with_scripted_votes(monkeypatch):
    vad = EnergyVad(
        VadConfig(
            min_speech_s=0.06,
            min_silence_s=0.08,
            frame_duration_ms=20,
        )
    )
    votes = iter(
        [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
    )
    monkeypatch.setattr(vad, "_detect_frame", lambda _frame: next(votes))

    start_event = vad.process(_dummy_chunk(960))
    assert start_event.speech_started is True
    assert vad.speaking is True

    stop_event = vad.process(_dummy_chunk(1280))
    assert stop_event.speech_stopped is True
    assert vad.speaking is False


def test_webrtc_vad_ignores_short_spike(monkeypatch):
    vad = EnergyVad(
        VadConfig(
            min_speech_s=0.10,
            min_silence_s=0.10,
            frame_duration_ms=20,
        )
    )
    votes = iter([True, False, False, False, False])
    monkeypatch.setattr(vad, "_detect_frame", lambda _frame: next(votes))

    event = vad.process(_dummy_chunk(1600))

    assert event.speech_started is False
    assert vad.speaking is False


def test_webrtc_vad_tracks_frame_statistics(monkeypatch):
    vad = EnergyVad(VadConfig(frame_duration_ms=20))
    votes = iter([True, False, True, True])
    monkeypatch.setattr(vad, "_detect_frame", lambda _frame: next(votes))

    vad.process(_dummy_chunk(1280))

    assert vad.last_frame_count == 4
    assert vad.last_voiced_frame_count == 3
    assert vad.last_speech_ratio == 0.75
