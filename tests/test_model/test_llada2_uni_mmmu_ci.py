# SPDX-License-Identifier: Apache-2.0
"""MMMU accuracy and speed CI for LLaDA2-Uni (Text+Image → Text, DLLM pipeline).

Usage:
    pytest tests/test_model/test_llada2_uni_mmmu_ci.py -s -x
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from benchmarks.metrics.mmmu import print_mmmu_accuracy_summary
from benchmarks.metrics.performance import print_speed_summary
from sglang_omni.utils import find_available_port
from tests.utils import (
    apply_slack,
    assert_speed_thresholds,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "inclusionAI/LLaDA2.0-Uni"

CONCURRENCY = 8
STARTUP_TIMEOUT = 300

MMMU_MIN_ACCURACY = 0.35

_MMMU_P95 = {
    8: {
        "throughput_qps": 0.50,
        "tok_per_s_agg": 30.0,
        "latency_mean_s": 15.0,
    },
}
MMMU_THRESHOLDS = apply_slack(_MMMU_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the LLaDA2-Uni server and wait until healthy."""
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory)
    cmd = [
        sys.executable,
        "examples/run_llada2_uni_server.py",
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--model-name",
        "llada2-uni",
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port
    yield proc
    stop_server(proc)


@pytest.mark.benchmark
def test_mmmu_accuracy_and_speed(
    server_process: subprocess.Popen,
    tmp_path: Path,
) -> None:
    """Run MMMU eval and assert accuracy and speed meet thresholds."""
    config = MMMUEvalConfig(
        model="llada2-uni",
        port=server_process.port,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "mmmu"),
        repo_id=DATASETS["mmmu-ci-50"],
        warmup=2,
    )
    results = asyncio.run(run_mmmu_eval(config))

    summary = results["summary"]
    speed = results["speed"]
    print_mmmu_accuracy_summary(summary, config.model)
    print_speed_summary(speed, config.model, CONCURRENCY, title="MMMU Speed")

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
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
