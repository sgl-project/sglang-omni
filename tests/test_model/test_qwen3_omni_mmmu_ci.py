# SPDX-License-Identifier: Apache-2.0
"""MMMU accuracy and speed CI for Qwen3-Omni (Text+Image → Text, Talker OFF).

Usage:
    pytest tests/test_model/test_qwen3_omni_mmmu_ci.py -s -x

Author:
    Yifei Gao https://github.com/PasserBy4
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.eval.mmmu import MMMUEvalConfig, run_mmmu_eval
from tests.utils import (
    apply_slack,
    assert_speed_thresholds,
    find_free_port,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

MAX_SAMPLES = 50
CONCURRENCY = 16
STARTUP_TIMEOUT = 900

MMMU_MIN_ACCURACY = 0.55

# TODO(Yifei): Tighten after gathering P95 baselines on CI hardware.
_MMMU_P95 = {
    16: {
        "throughput_qps": 0.05,
        "tok_per_s_agg": 10.0,
        "latency_mean_s": 200.0,
    },
}
MMMU_THRESHOLDS = apply_slack(_MMMU_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the text-only Qwen3-Omni server and wait until healthy."""
    port = find_free_port()
    log_file = tmp_path_factory.mktemp("server_logs") / "server.log"
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_server.py",
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--model-name",
        "qwen3-omni",
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
        model="qwen3-omni",
        port=server_process.port,
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "mmmu"),
    )
    results = asyncio.run(run_mmmu_eval(config))

    summary = results["summary"]
    assert summary["accuracy"] >= MMMU_MIN_ACCURACY, (
        f"MMMU accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {MMMU_MIN_ACCURACY} ({MMMU_MIN_ACCURACY * 100:.0f}%)"
    )

    speed = results["speed"]
    assert_speed_thresholds(speed, MMMU_THRESHOLDS, CONCURRENCY, check_rtf=False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
