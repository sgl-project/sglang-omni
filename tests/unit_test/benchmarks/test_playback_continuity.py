# SPDX-License-Identifier: Apache-2.0
"""Unit tests for playback continuity / underrun helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.benchmarker.data import FinishReason, RequestResult
from benchmarks.metrics.performance import (
    compute_speed_metrics,
    print_saved_tts_speed_summary,
)
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


CONTINUITY_KEYS = (
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
    for key in CONTINUITY_KEYS:
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


def test_compute_speed_metrics_counts_max_token_hits() -> None:
    def request_result(finish_reason: FinishReason) -> RequestResult:
        return RequestResult(
            request_id=finish_reason.value,
            is_success=True,
            latency_s=1.0,
            audio_duration_s=1.0,
            rtf=1.0,
            finish_reason=finish_reason,
        )

    metrics = compute_speed_metrics([request_result(reason) for reason in FinishReason])
    assert metrics["max_token_hits"] == 1
    assert metrics["finish_reason_observed"] == 2

    # A run whose responses carried no finish reason observed nothing.
    unknown = compute_speed_metrics([request_result(FinishReason.UNKNOWN)])
    assert unknown["max_token_hits"] == 0
    assert unknown["finish_reason_observed"] == 0


@pytest.mark.parametrize(
    "cap_metrics",
    [
        None,
        {"max_token_hits": 0, "finish_reason_observed": 0},
        {"max_token_hits": 1, "finish_reason_observed": 2},
    ],
)
def test_saved_speed_summary_preserves_missing_and_observed_cap_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cap_metrics: dict[str, int] | None,
) -> None:
    summary = {
        "completed_requests": 3,
        "failed_requests": 1,
        "latency_mean_s": 0.5,
    }
    if cap_metrics is not None:
        summary.update(cap_metrics)
    else:
        pass
    results_path = tmp_path / "speed_results.json"
    results_path.write_text(
        json.dumps({"summary": summary, "config": {"concurrency": 2}})
    )

    assert print_saved_tts_speed_summary(
        str(tmp_path), "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )

    printed = capsys.readouterr().out
    assert "Latency mean (s):" in printed
    assert "0.5" in printed
    if cap_metrics is None:
        assert "Max token hits:" not in printed
    else:
        assert (
            f"{cap_metrics['max_token_hits']} / "
            f"{cap_metrics['finish_reason_observed']} observed"
        ) in printed
    assert json.loads(results_path.read_text())["summary"] == summary
