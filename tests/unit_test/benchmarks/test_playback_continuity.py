# SPDX-License-Identifier: Apache-2.0
"""Unit tests for playback continuity / underrun helpers."""

from __future__ import annotations

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.playback_continuity import (
    compute_max_playback_underrun_s,
    continuity_pass_rate,
    summarize_playback_continuity,
)


def test_compute_max_playback_underrun_single_chunk_is_na() -> None:
    assert compute_max_playback_underrun_s([1.0], [0.5]) is None


def test_compute_max_playback_underrun_gapless_stream() -> None:
    arrivals = [0.0, 0.4, 0.8]
    durations = [0.4, 0.4, 0.4]
    assert compute_max_playback_underrun_s(arrivals, durations) == pytest.approx(0.0)


def test_compute_max_playback_underrun_reports_largest_gap() -> None:
    arrivals = [0.0, 0.45, 1.05]
    durations = [0.4, 0.4, 0.4]
    assert compute_max_playback_underrun_s(arrivals, durations) == pytest.approx(0.2)


def test_compute_max_playback_underrun_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_max_playback_underrun_s([0.0, 1.0], [0.5])


def test_continuity_pass_rate_excludes_na() -> None:
    underruns = [0.01, None, 0.2, 0.04]
    assert continuity_pass_rate(underruns, threshold_s=0.05) == pytest.approx(2 / 3)


def test_summarize_playback_continuity_reports_c_thresholds() -> None:
    summary = summarize_playback_continuity([0.01, 0.08, 0.15, None])
    assert summary["playback_continuity_requests"] == 3
    assert summary["playback_continuity_na_requests"] == 1
    assert summary["c50"] == pytest.approx(33.33)
    assert summary["c100"] == pytest.approx(66.67)
    assert summary["c200"] == pytest.approx(100.0)


def test_compute_speed_metrics_includes_continuity_gates() -> None:
    outputs = [
        RequestResult(
            request_id="ok-multi",
            is_success=True,
            latency_s=1.0,
            audio_duration_s=1.0,
            rtf=1.0,
            audio_ttfp_s=0.1,
            inter_chunk_s=[0.2],
            chunk_audio_duration_s=[0.4, 0.6],
            max_playback_underrun_s=0.02,
            audio_chunk_count=2,
        ),
        RequestResult(
            request_id="ok-single",
            is_success=True,
            latency_s=1.0,
            audio_duration_s=1.0,
            rtf=1.0,
            audio_ttfp_s=0.1,
            chunk_audio_duration_s=[1.0],
            max_playback_underrun_s=None,
            audio_chunk_count=1,
        ),
        RequestResult(
            request_id="late",
            is_success=True,
            latency_s=1.0,
            audio_duration_s=1.0,
            rtf=1.0,
            audio_ttfp_s=0.1,
            inter_chunk_s=[0.5],
            chunk_audio_duration_s=[0.2, 0.8],
            max_playback_underrun_s=0.25,
            audio_chunk_count=2,
        ),
    ]

    metrics = compute_speed_metrics(outputs, wall_clock_s=2.0)

    assert metrics["playback_continuity_requests"] == 2
    assert metrics["playback_continuity_na_requests"] == 1
    assert metrics["c50"] == pytest.approx(50.0)
    assert metrics["c100"] == pytest.approx(50.0)
    assert metrics["c200"] == pytest.approx(50.0)
    assert metrics["max_playback_underrun_mean_s"] == pytest.approx(0.135)


_CONTINUITY_KEYS = (
    "playback_continuity_requests",
    "playback_continuity_na_requests",
    "max_playback_underrun_mean_s",
    "max_playback_underrun_p95_s",
    "max_playback_underrun_p99_s",
    "c50",
    "c100",
    "c200",
)


def test_compute_speed_metrics_skips_continuity_for_non_streaming_audio() -> None:
    # note (akazaakane): compute_speed_metrics is shared with the ASR/Omni
    # benchmarks, which never emit audio chunks, so the continuity fields must
    # stay absent rather than reporting None for every unrelated request.
    outputs = [
        RequestResult(
            request_id="asr-1",
            is_success=True,
            latency_s=1.0,
            audio_duration_s=1.0,
            rtf=1.0,
        ),
        RequestResult(
            request_id="asr-2",
            is_success=True,
            latency_s=2.0,
            audio_duration_s=2.0,
            rtf=1.0,
        ),
    ]

    metrics = compute_speed_metrics(outputs, wall_clock_s=2.0)

    assert metrics["completed_requests"] == 2
    for key in _CONTINUITY_KEYS:
        assert key not in metrics


def test_compute_speed_metrics_reports_all_single_chunk_streams() -> None:
    outputs = [
        RequestResult(
            request_id=f"short-{index}",
            is_success=True,
            latency_s=1.0,
            audio_duration_s=0.5,
            rtf=2.0,
            audio_ttfp_s=0.4,
            chunk_audio_duration_s=[0.5],
            max_playback_underrun_s=None,
            audio_chunk_count=1,
        )
        for index in range(3)
    ]

    metrics = compute_speed_metrics(outputs, wall_clock_s=1.0)

    assert metrics["playback_continuity_requests"] == 0
    assert metrics["playback_continuity_na_requests"] == 3
    assert metrics["c50"] is None
    assert metrics["c100"] is None
    assert metrics["c200"] is None
