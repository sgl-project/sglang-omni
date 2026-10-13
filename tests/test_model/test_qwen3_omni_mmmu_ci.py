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
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.mmmu import print_mmmu_accuracy_summary
from benchmarks.metrics.performance import print_speed_summary
from tests.test_model.omni_ci_config import OmniCiModelPreset
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, assert_speed_thresholds

CONCURRENCY = 16


@pytest.mark.benchmark
def test_mmmu_accuracy_and_speed(
    omni_ci_model: OmniCiModelPreset,
    omni_ci_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run MMMU eval and assert accuracy and speed meet thresholds."""
    config = MMMUEvalConfig(
        model=omni_ci_model.name,
        port=omni_ci_server.port,
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
        omni_ci_server,
        label=f"{omni_ci_model.name} MMMU",
    ) as router_guard:
        results = asyncio.run(run_mmmu_eval(config))

    summary = results["summary"]
    speed = results["speed"]
    dataset_label = format_benchmark_dataset_label(
        dataset="mmmu-ci-50",
        repo_id=config.repo_id,
    )
    print_mmmu_accuracy_summary(summary, config.model, dataset=dataset_label)
    print_speed_summary(
        speed, config.model, CONCURRENCY, title="MMMU Speed", dataset=dataset_label
    )

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    thresholds = omni_ci_model.thresholds["mmmu"]
    checks = MetricCheckCollector("MMMU accuracy and speed")
    checks.check_assertion(
        "router traffic",
        router_guard.assert_served,
        min_total_requests=total,
    )
    checks.check(
        failed == 0,
        f"MMMU had {failed}/{total} failed requests (timeouts or empty responses); "
        f"any failure fails the test",
    )

    accuracy = summary.get("accuracy")
    if accuracy is None:
        checks.fail("MMMU accuracy missing from summary")
    elif thresholds.calibrated:
        checks.check(
            accuracy >= thresholds.accuracy,
            f"MMMU accuracy {accuracy:.4f} "
            f"({accuracy * 100:.1f}%) < "
            f"threshold {thresholds.accuracy} ({thresholds.accuracy * 100:.0f}%)",
        )

    if thresholds.calibrated:
        assert_speed_thresholds(speed, thresholds.speed, CONCURRENCY, collector=checks)
    thresholds.require_calibrated(omni_ci_model.name, "mmmu", checks)
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
