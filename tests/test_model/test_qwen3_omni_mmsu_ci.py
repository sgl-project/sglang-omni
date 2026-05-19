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
from tests.test_model.omni_router_utils import ManagedRouterHandle
from tests.utils import apply_slack, assert_speed_thresholds

CONCURRENCY = 8

MMSU_MIN_ACCURACY = 0.7

_MMSU_P95 = {
    8: {
        "throughput_qps": 8.478,
        "tok_per_s_agg": 2.2,
        "latency_mean_s": 0.936,
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
    qwen3_omni_router_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run MMSU eval and assert accuracy and speed meet thresholds."""
    args = _build_args(qwen3_omni_router_server.port, str(tmp_path / "mmsu"))
    results = asyncio.run(run_mmsu(args))

    print_mmsu_summary(results["accuracy"], args.model, speed_metrics=results["speed"])

    failed = results["accuracy"].get("failed_samples", 0)
    total = results["accuracy"].get("total_samples", 0)
    assert failed == 0, (
        f"MMSU had {failed}/{total} failed requests (timeouts or empty responses); "
        f"any failure fails the test"
    )

    accuracy = results["accuracy"]["overall_accuracy"]
    assert accuracy >= MMSU_MIN_ACCURACY, (
        f"MMSU accuracy {accuracy:.4f} ({accuracy * 100:.1f}%) < "
        f"threshold {MMSU_MIN_ACCURACY} ({MMSU_MIN_ACCURACY * 100:.0f}%)"
    )

    assert_speed_thresholds(results["speed"], MMSU_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
