# SPDX-License-Identifier: Apache-2.0
"""Shared types and constants for generated audio metrics."""

from __future__ import annotations

from typing import Literal

MetricValue = str | int | float | bool | None
MetricDetails = dict[str, MetricValue]
RuntimeMetricEvent = Literal[
    "audio_chunk",
    "audio_response_aborted",
    "audio_response_done",
    "audio_response_start",
]

CONTINUITY_THRESHOLDS_MS = (50, 100, 200)
LATENCY_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
FAST_BUCKETS_S = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
    60,
)
RTF_BUCKETS = (0.1, 0.2, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 10)
ORPHANED_STATE_TTL_NS = 3_600_000_000_000
