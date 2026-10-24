# SPDX-License-Identifier: Apache-2.0
"""Audio timing, continuity, and lifecycle in the shared serving registry."""

import unittest

from prometheus_client import generate_latest

from sglang_omni.metrics.types import ORPHANED_STATE_TTL_NS
from sglang_omni.serve.request_metrics import RequestMetrics


class AudioMetricsTest(unittest.TestCase):
    def test_audio_timing_and_continuity(self) -> None:
        metrics = RequestMetrics("test")
        metrics.audio_metrics.record("audio_response_start", "request-1", 1_000_000_000)
        metrics.audio_metrics.record(
            "audio_chunk",
            "request-1",
            1_200_000_000,
            {"duration_s": 0.1},
        )
        metrics.audio_metrics.record(
            "audio_chunk",
            "request-1",
            1_350_000_000,
            {"duration_s": 0.1},
        )
        metrics.audio_metrics.record(
            "audio_response_done",
            "request-1",
            1_400_000_000,
            {"audio_generation_s": 0.25},
        )
        snapshot = generate_latest(metrics.registry).decode()

        self.assertIn("sglang_omni:audio_ttfp_s_sum 0.2", snapshot)
        self.assertIn("sglang_omni:audio_duration_s_sum 0.2", snapshot)
        self.assertIn("sglang_omni:audio_chunk_interval_s_count 1.0", snapshot)
        self.assertIn("sglang_omni:audio_underrun_s_count 1.0", snapshot)
        self.assertIn("sglang_omni:audio_rtf_count 1.0", snapshot)
        self.assertIn(
            'sglang_omni:audio_continuity_ok_total{threshold_ms="100"} 1.0',
            snapshot,
        )

    def test_aborted_audio_does_not_count_as_completed(self) -> None:
        metrics = RequestMetrics("test")
        metrics.audio_metrics.record("audio_response_start", "request-1", 1_000_000_000)
        metrics.audio_metrics.record(
            "audio_chunk",
            "request-1",
            1_100_000_000,
            {"duration_s": 0.1},
        )
        metrics.audio_metrics.record(
            "audio_response_aborted", "request-1", 1_200_000_000
        )
        metrics.audio_metrics.record("audio_response_done", "request-1", 1_300_000_000)
        snapshot = generate_latest(metrics.registry).decode()

        self.assertIn("sglang_omni:audio_duration_s_count 0.0", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_s_count 1.0", snapshot)

    def test_render_prunes_orphaned_audio_state(self) -> None:
        metrics = RequestMetrics("test")
        metrics.audio_metrics.record("audio_response_start", "request-1", 1)
        metrics.audio_metrics.record("audio_chunk", "request-1", 2, {"duration_s": 0.1})

        metrics.audio_metrics.prune(ORPHANED_STATE_TTL_NS + 3)

        metrics.audio_metrics.record(
            "audio_response_done", "request-1", ORPHANED_STATE_TTL_NS + 4
        )
        self.assertIn(
            "sglang_omni:audio_duration_s_count 0.0",
            generate_latest(metrics.registry).decode(),
        )
