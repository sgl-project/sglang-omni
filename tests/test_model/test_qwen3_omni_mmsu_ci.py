# SPDX-License-Identifier: Apache-2.0
"""MMSU accuracy and speed CI for Qwen3-Omni (Text + Audio → Text, Talker OFF).

Usage:
    pytest tests/test_model/test_qwen3_omni_mmsu_ci.py -s -x

Author:
    Yifei Gao https://github.com/PasserBy4
    Huapeng Zhou https://github.com/PopSoda2002
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from benchmarks.metrics.mmsu import print_mmsu_summary
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, apply_slack, assert_speed_thresholds

CONCURRENCY = 16

MMSU_MIN_ACCURACY = 0.7

_MMSU_P95 = {
    16: {
        "throughput_qps": 50.399,
        "tok_per_s_agg": 6.5,
        "latency_mean_s": 0.317,
    },
}
MMSU_THRESHOLDS = apply_slack(_MMSU_P95)


def _build_args(port: int, output_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=None,
        host="localhost",
        port=port,
        model="qwen3-omni",
        modalities="text",
        output_dir=output_dir,
        max_samples=None,
        task_names=None,
        categories=None,
        prompt=None,
        max_tokens=32,
        temperature=0.0,
        warmup=0,
        max_concurrency=CONCURRENCY,
        request_rate=float("inf"),
        timeout_s=300,
        save_audio=False,
        disable_tqdm=False,
        seed=None,
        repo_id=DATASETS["mmsu-ci-2000"],
        # Unused in text-only mode but kept for API consistency with run().
        lang="en",
        asr_device="cuda:0",
    )


@pytest.mark.benchmark
def test_mmsu_accuracy_and_speed(
    qwen3_omni_mmsu_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run MMSU eval and assert accuracy and speed meet thresholds."""
    args = _build_args(qwen3_omni_mmsu_server.port, str(tmp_path / "mmsu"))
    with router_worker_traffic_guard(
        qwen3_omni_mmsu_server,
        label="Qwen3-Omni MMSU",
    ) as router_guard:
        results = asyncio.run(run_mmsu(args))

    print_mmsu_summary(results["accuracy"], args.model, speed_metrics=results["speed"])

    failed = results["accuracy"].get("failed_samples", 0)
    total = results["accuracy"].get("total_samples", 0)
    checks = MetricCheckCollector("MMSU accuracy and speed")
    checks.check_assertion(
        "router traffic",
        router_guard.assert_served,
        min_total_requests=total,
    )
    checks.check(
        failed == 0,
        f"MMSU had {failed}/{total} failed requests (timeouts or empty responses); "
        f"any failure fails the test",
    )

    accuracy = results["accuracy"].get("overall_accuracy")
    if accuracy is None:
        checks.fail("MMSU overall_accuracy missing from accuracy results")
    else:
        checks.check(
            accuracy >= MMSU_MIN_ACCURACY,
            f"MMSU accuracy {accuracy:.4f} ({accuracy * 100:.1f}%) < "
            f"threshold {MMSU_MIN_ACCURACY} ({MMSU_MIN_ACCURACY * 100:.0f}%)",
        )

    assert_speed_thresholds(
        results["speed"], MMSU_THRESHOLDS, CONCURRENCY, collector=checks
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
