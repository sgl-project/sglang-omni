# SPDX-License-Identifier: Apache-2.0
"""Contract tests for generated audio Prometheus metrics."""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sglang_omni.metrics.runtime import RuntimeMetrics
from sglang_omni.metrics.types import ORPHANED_STATE_TTL_NS


class RuntimeMetricsTest(unittest.TestCase):
    def test_runtime_owns_only_audio_metric_names(self) -> None:
        metrics = RuntimeMetrics()
        metric_names = {
            metric.name
            for collector in metrics.registry.collect()
            for metric in [collector]
        }

        omni_metric_names = {
            name for name in metric_names if name.startswith("sglang_omni:")
        }
        self.assertTrue(omni_metric_names)
        self.assertTrue(
            all(name.startswith("sglang_omni:audio_") for name in omni_metric_names)
        )
        self.assertFalse(any(name.startswith("sglang:") for name in metric_names))

    def test_render_combines_upstream_and_audio_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as metrics_dir:
            env = dict(os.environ)
            env["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from prometheus_client import Counter; "
                        "Counter('sglang:test_upstream_total', 'test').inc(2)"
                    ),
                ],
                check=True,
                env=env,
            )
            with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": metrics_dir}):
                snapshot = RuntimeMetrics().render().decode()

        self.assertIn("sglang:test_upstream_total 2.0", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_s", snapshot)

    def test_audio_timing_and_continuity(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("audio_response_start", "request-1", "api", 1_000_000_000)
        metrics.record(
            "audio_chunk",
            "request-1",
            "api",
            1_200_000_000,
            {"duration_s": 0.1},
        )
        metrics.record(
            "audio_chunk",
            "request-1",
            "api",
            1_350_000_000,
            {"duration_s": 0.1},
        )
        metrics.record(
            "audio_response_done",
            "request-1",
            "api",
            1_400_000_000,
            {"audio_generation_s": 0.25},
        )
        snapshot = metrics.render().decode()

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
        metrics = RuntimeMetrics()
        metrics.record("audio_response_start", "request-1", "api", 1_000_000_000)
        metrics.record(
            "audio_chunk",
            "request-1",
            "api",
            1_100_000_000,
            {"duration_s": 0.1},
        )
        metrics.record("audio_response_aborted", "request-1", "api", 1_200_000_000)
        metrics.record("audio_response_done", "request-1", "api", 1_300_000_000)
        snapshot = metrics.render().decode()

        self.assertIn("sglang_omni:audio_duration_s_count 0.0", snapshot)
        self.assertIn("sglang_omni:audio_ttfp_s_count 1.0", snapshot)

    def test_render_prunes_orphaned_audio_state(self) -> None:
        metrics = RuntimeMetrics()
        metrics.record("audio_response_start", "request-1", "api", 1)
        metrics.record("audio_chunk", "request-1", "api", 2, {"duration_s": 0.1})

        with patch("sglang_omni.metrics.runtime.time.perf_counter_ns") as now_ns:
            now_ns.return_value = ORPHANED_STATE_TTL_NS + 3
            metrics.render()

        metrics.record(
            "audio_response_done", "request-1", "api", ORPHANED_STATE_TTL_NS + 4
        )
        self.assertIn(
            "sglang_omni:audio_duration_s_count 0.0",
            metrics.render().decode(),
        )


if __name__ == "__main__":
    unittest.main()
