# SPDX-License-Identifier: Apache-2.0
"""Video-AMME accuracy and speed CI for Qwen3-Omni (Video+Audio -> Text).

Usage:
    pytest tests/test_model/test_qwen3_omni_videoamme_ci.py -s -x

Author:
    Ratish P https://github.com/Ratish21
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, apply_slack, assert_speed_thresholds

CONCURRENCY = 16
MAX_SAMPLES = 50

VIDEOAMME_MIN_ACCURACY = 0.66

_VIDEOAMME_P95 = {
    16: {
        "throughput_qps": 1.689,
        "output_tok_per_req_s": 6.2,
        "latency_mean_s": 8.234,
    },
}
VIDEOAMME_THRESHOLDS = apply_slack(_VIDEOAMME_P95)


@pytest.mark.benchmark
def test_videoamme_accuracy_and_speed(
    qwen3_omni_fp8_colocated_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run videoamme-ci-50 at concurrency=16 and report accuracy + speed."""
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_fp8_colocated_server.port,
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videoamme"),
        repo_id=DATASETS["videoamme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        disable_tqdm=False,
        timeout_s=500,
    )
    with router_worker_traffic_guard(
        qwen3_omni_fp8_colocated_server,
        label="Qwen3-Omni Video-AMME",
    ) as router_guard:
        results = asyncio.run(run_videoamme_eval(config))

    summary = results["summary"]
    dataset_label = format_benchmark_dataset_label(
        dataset="videoamme-ci-50",
        repo_id=config.repo_id,
    )
    print_videomme_accuracy_summary(
        summary,
        config.model,
        title="Video-AMME Accuracy",
        dataset=dataset_label,
    )
    print_speed_summary(
        results["speed"],
        config.model,
        CONCURRENCY,
        title="Video-AMME Speed",
        dataset=dataset_label,
    )
    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    checks = MetricCheckCollector("Video-AMME accuracy and speed")
    checks.check_assertion(
        "router traffic",
        router_guard.assert_served,
        min_total_requests=total,
    )
    checks.check(
        failed == 0,
        f"Video-AMME had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test",
    )
    accuracy = summary.get("accuracy")
    if accuracy is None:
        checks.fail("Video-AMME accuracy missing from summary")
    else:
        checks.check(
            accuracy >= VIDEOAMME_MIN_ACCURACY,
            f"Video-AMME accuracy {accuracy:.4f} "
            f"({accuracy * 100:.1f}%) < "
            f"threshold {VIDEOAMME_MIN_ACCURACY} ({VIDEOAMME_MIN_ACCURACY * 100:.0f}%)",
        )
    assert_speed_thresholds(
        results["speed"], VIDEOAMME_THRESHOLDS, CONCURRENCY, collector=checks
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
