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
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.mmsu import print_mmsu_summary
from tests.test_model.omni_ci_config import OmniCiModelPreset
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, assert_speed_thresholds

CONCURRENCY_BY_MODEL = {"qwen3-omni": 96, "minicpmo": 16}


def _build_args(
    omni_ci_model: OmniCiModelPreset, port: int, output_dir: str
) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=None,
        host="localhost",
        port=port,
        model=omni_ci_model.name,
        modalities="text",
        output_dir=output_dir,
        max_samples=None,
        task_names=None,
        categories=None,
        prompt=None,
        max_tokens=32,
        temperature=0.0,
        warmup=0,
        max_concurrency=CONCURRENCY_BY_MODEL[omni_ci_model.name],
        request_rate=float("inf"),
        timeout_s=300,
        save_audio=False,
        disable_tqdm=False,
        seed=None,
        repo_id=DATASETS["mmsu-ci-2000"],
        # Unused by this text-output benchmark (modalities="text"); kept for API consistency with run().
        lang="en",
        asr_device="cuda:0",
    )


@pytest.mark.benchmark
def test_mmsu_accuracy_and_speed(
    omni_ci_model: OmniCiModelPreset,
    omni_ci_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run MMSU eval and assert accuracy and speed meet thresholds."""
    args = _build_args(omni_ci_model, omni_ci_server.port, str(tmp_path / "mmsu"))
    with router_worker_traffic_guard(
        omni_ci_server,
        label=f"{omni_ci_model.name} MMSU",
    ) as router_guard:
        results = asyncio.run(run_mmsu(args))

    print_mmsu_summary(
        results["accuracy"],
        args.model,
        speed_metrics=results["speed"],
        dataset=format_benchmark_dataset_label(
            dataset="mmsu-ci-2000",
            repo_id=args.repo_id,
        ),
    )

    failed = results["accuracy"].get("failed_samples", 0)
    total = results["accuracy"].get("total_samples", 0)
    thresholds = omni_ci_model.thresholds["mmsu"]
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
    elif thresholds.calibrated:
        checks.check(
            accuracy >= thresholds.accuracy,
            f"MMSU accuracy {accuracy:.4f} ({accuracy * 100:.1f}%) < "
            f"threshold {thresholds.accuracy} ({thresholds.accuracy * 100:.0f}%)",
        )

    if thresholds.calibrated:
        assert_speed_thresholds(
            results["speed"],
            thresholds.speed,
            args.max_concurrency,
            collector=checks,
        )
    thresholds.require_calibrated(omni_ci_model.name, "mmsu", checks)
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
