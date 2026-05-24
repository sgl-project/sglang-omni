# SPDX-License-Identifier: Apache-2.0
"""Video-MME accuracy and speed CI for Qwen3-Omni (Text+Video -> Text, Talker OFF).

Usage:
    pytest tests/test_model/test_qwen3_omni_videomme_ci.py -s -x

Author:
    Qiujiang Chen https://github.com/Jayon02
    Chenyang Zhao https://github.com/zhaochenyang20
    Yifei Gao https://github.com/PasserBy4
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, apply_slack, assert_speed_thresholds

CONCURRENCY = 16
MAX_SAMPLES = 50

VIDEOMME_MIN_ACCURACY = 0.52

_VIDEOMME_P95 = {
    16: {
        "throughput_qps": 0.996,
        "tok_per_s_agg": 7.7,
        "latency_mean_s": 15.296,
    },
}
VIDEOMME_THRESHOLDS = apply_slack(_VIDEOMME_P95)


@pytest.mark.benchmark
def test_videomme_accuracy_and_speed(
    qwen3_omni_thinker_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run videomme-ci-50 at concurrency=16 and report accuracy + speed."""
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_thinker_server.port,
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videomme"),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        disable_tqdm=False,
        timeout_s=500,
    )
    with router_worker_traffic_guard(
        qwen3_omni_thinker_server,
        label="Qwen3-Omni Video-MME",
    ) as router_guard:
        results = asyncio.run(
            run_video_eval(
                config,
                task_label="Video-MME",
                output_filename="videomme_results.json",
                audio_output_dir_default="results/videomme_audio",
            )
        )

    summary = results["summary"]
    print_videomme_accuracy_summary(summary, config.model)
    print_speed_summary(
        results["speed"],
        config.model,
        CONCURRENCY,
        title="Video-MME Speed",
    )
    total = summary.get("total_samples", 0)
    checks = MetricCheckCollector("Video-MME accuracy and speed")
    checks.check_assertion(
        "router traffic",
        router_guard.assert_served,
        min_total_requests=total,
    )
    accuracy = summary.get("accuracy")
    if accuracy is None:
        checks.fail("Video-MME accuracy missing from summary")
    else:
        checks.check(
            accuracy >= VIDEOMME_MIN_ACCURACY,
            f"Video-MME accuracy {accuracy:.4f} "
            f"({accuracy * 100:.1f}%) < "
            f"threshold {VIDEOMME_MIN_ACCURACY} ({VIDEOMME_MIN_ACCURACY * 100:.0f}%)",
        )
    assert_speed_thresholds(
        results["speed"], VIDEOMME_THRESHOLDS, CONCURRENCY, collector=checks
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
