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
from sglang_omni.utils import find_available_port
from tests.utils import (
    ServerHandle,
    apply_slack,
    assert_speed_thresholds,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

CONCURRENCY = 8
STARTUP_TIMEOUT = 300

MMSU_MIN_ACCURACY = 0.69

# Threshold reference: https://github.com/sgl-project/sglang-omni/pull/382#issuecomment-4366925373
_MMSU_P95 = {
    8: {
        "throughput_qps": 30.036,
        "tok_per_s_agg": 7.8,
        "latency_mean_s": 0.266,
    },
}
MMSU_THRESHOLDS = apply_slack(_MMSU_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory)
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
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


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
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run MMSU eval and assert accuracy and speed meet thresholds."""
    args = _build_args(server_process.port, str(tmp_path / "mmsu"))
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
