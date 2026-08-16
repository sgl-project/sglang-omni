# SPDX-License-Identifier: Apache-2.0
"""Client-side playback underrun and continuity pass-rate helpers."""

from __future__ import annotations

from collections.abc import Sequence

CONTINUITY_THRESHOLDS_MS: tuple[int, ...] = (50, 100, 200)


def compute_max_playback_underrun_s(
    arrival_times_s: Sequence[float],
    chunk_durations_s: Sequence[float],
) -> float | None:
    """Max playback underrun in seconds; None for fewer than two chunks."""
    if len(arrival_times_s) != len(chunk_durations_s):
        raise ValueError(
            "arrival_times_s and chunk_durations_s must have the same length "
            f"(got {len(arrival_times_s)} and {len(chunk_durations_s)})"
        )
    if len(arrival_times_s) < 2:
        return None
    if any(duration < 0 for duration in chunk_durations_s):
        raise ValueError("chunk_durations_s must be non-negative")

    deadline = float(arrival_times_s[0]) + float(chunk_durations_s[0])
    max_underrun = 0.0
    for arrival, duration in zip(arrival_times_s[1:], chunk_durations_s[1:]):
        arrival_f = float(arrival)
        duration_f = float(duration)
        underrun = max(0.0, arrival_f - deadline)
        if underrun > max_underrun:
            max_underrun = underrun
        deadline = max(deadline, arrival_f) + duration_f
    return max_underrun


def continuity_pass_rate(
    max_underruns_s: Sequence[float | None],
    *,
    threshold_s: float,
) -> float | None:
    """Share of multi-chunk requests with max underrun <= threshold_s."""
    if threshold_s < 0:
        raise ValueError("threshold_s must be non-negative")
    scored = [value for value in max_underruns_s if value is not None]
    if not scored:
        return None
    passed = sum(1 for value in scored if value <= threshold_s)
    return passed / len(scored)


def summarize_playback_continuity(
    max_underruns_s: Sequence[float | None],
    *,
    thresholds_ms: Sequence[int] = CONTINUITY_THRESHOLDS_MS,
) -> dict[str, float | int | None]:
    """Aggregate underrun percentiles and c50/c100/c200 pass rates."""
    scored = [float(value) for value in max_underruns_s if value is not None]
    summary: dict[str, float | int | None] = {
        "playback_continuity_requests": len(scored),
        "playback_continuity_na_requests": len(max_underruns_s) - len(scored),
    }
    if not scored:
        summary["max_playback_underrun_mean_s"] = None
        summary["max_playback_underrun_p95_s"] = None
        summary["max_playback_underrun_p99_s"] = None
        for threshold_ms in thresholds_ms:
            summary[f"c{int(threshold_ms)}"] = None
        return summary

    import numpy as np

    summary["max_playback_underrun_mean_s"] = round(float(np.mean(scored)), 4)
    summary["max_playback_underrun_p95_s"] = round(float(np.percentile(scored, 95)), 4)
    summary["max_playback_underrun_p99_s"] = round(float(np.percentile(scored, 99)), 4)
    for threshold_ms in thresholds_ms:
        rate = continuity_pass_rate(scored, threshold_s=threshold_ms / 1000.0)
        summary[f"c{int(threshold_ms)}"] = (
            None if rate is None else round(100.0 * rate, 2)
        )
    return summary
