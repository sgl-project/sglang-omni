# SPDX-License-Identifier: Apache-2.0
"""MMMU accuracy and speed CI for Qwen3-Omni (Text+Image → Text, Talker OFF).

Usage:
    pytest tests/test_model/test_qwen3_omni_mmmu_ci.py -s -x

Author:
    Yifei Gao https://github.com/PasserBy4
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from benchmarks.metrics.mmmu import print_mmmu_accuracy_summary
from benchmarks.metrics.performance import print_speed_summary
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import apply_slack, assert_speed_thresholds

CONCURRENCY = 16

MMMU_MIN_ACCURACY = 0.54

_MMMU_P95 = {
    16: {
        "throughput_qps": 1.245,
        "tok_per_s_agg": 57.8,
        "latency_mean_s": 10.881,
    },
}
MMMU_THRESHOLDS = apply_slack(_MMMU_P95)


@pytest.mark.benchmark
def test_mmmu_accuracy_and_speed(
    qwen3_omni_router_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run MMMU eval and assert accuracy and speed meet thresholds."""
    config = MMMUEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_router_server.port,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "mmmu"),
        repo_id=DATASETS["mmmu-ci-50"],
        # Note (Yifei):
        # Regression guard for issue #299: warmup pre-populates the image
        # encoder cache so the first real batch mixes cached and uncached
        # requests. warmup > 1 keeps the lone hit from landing alone.
        warmup=2,
    )
    with router_worker_traffic_guard(
        qwen3_omni_router_server,
        label="Qwen3-Omni MMMU",
    ) as router_guard:
        results = asyncio.run(run_mmmu_eval(config))

    summary = results["summary"]
    speed = results["speed"]
    print_mmmu_accuracy_summary(summary, config.model)
    print_speed_summary(speed, config.model, CONCURRENCY, title="MMMU Speed")

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    router_guard.assert_served(min_total_requests=total)
    assert failed == 0, (
        f"MMMU had {failed}/{total} failed requests (timeouts or empty responses); "
        f"any failure fails the test"
    )

    assert summary["accuracy"] >= MMMU_MIN_ACCURACY, (
        f"MMMU accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {MMMU_MIN_ACCURACY} ({MMMU_MIN_ACCURACY * 100:.0f}%)"
    )

    assert_speed_thresholds(speed, MMMU_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
