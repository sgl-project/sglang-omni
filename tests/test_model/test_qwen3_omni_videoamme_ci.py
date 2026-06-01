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
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from tests.utils import ServerHandle, apply_slack, assert_speed_thresholds

CONCURRENCY = 16
MAX_SAMPLES = 30

# Threshold reference: https://github.com/sgl-project/sglang-omni/pull/382#issuecomment-4366925373
VIDEOAMME_MIN_ACCURACY = 0.65

_VIDEOAMME_P95 = {
    16: {
        "throughput_qps": 0.222,
        "tok_per_s_agg": 0.8,
        "latency_mean_s": 55.085,
    },
}
VIDEOAMME_THRESHOLDS = apply_slack(_VIDEOAMME_P95)


@pytest.mark.benchmark
def test_videoamme_accuracy_and_speed(
    qwen3_omni_thinker_server: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run first 30 of videoamme-ci-50 at concurrency=16 and report accuracy + speed."""
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_thinker_server.port,
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
    results = asyncio.run(run_videoamme_eval(config))

    summary = results["summary"]
    print_videomme_accuracy_summary(
        summary,
        config.model,
        title="Video-AMME Accuracy",
    )
    print_speed_summary(
        results["speed"],
        config.model,
        CONCURRENCY,
        title="Video-AMME Speed",
    )
    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    assert failed == 0, (
        f"Video-AMME had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test"
    )
    assert summary["accuracy"] >= VIDEOAMME_MIN_ACCURACY, (
        f"Video-AMME accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {VIDEOAMME_MIN_ACCURACY} ({VIDEOAMME_MIN_ACCURACY * 100:.0f}%)"
    )
    assert_speed_thresholds(results["speed"], VIDEOAMME_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
