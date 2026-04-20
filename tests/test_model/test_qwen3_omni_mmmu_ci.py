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
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.dataset.mmmu import MMMUSample, load_mmmu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from sglang_omni.utils import find_available_port
from tests.utils import (
    apply_slack,
    assert_speed_thresholds,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

CONCURRENCY = 8
STARTUP_TIMEOUT = 900

MMMU_MIN_ACCURACY = 0.52

# Note (Yifei, Chenyang): Thresholds reference
# https://github.com/sgl-project/sglang-omni/pull/265#issuecomment-4228251028

_MMMU_P95 = {
    8: {
        "throughput_qps": 0.128,
        "tok_per_s_agg": 13.1,
        "latency_mean_s": 59.30,
    },
}
MMMU_THRESHOLDS = apply_slack(_MMMU_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the text-only Qwen3-Omni server and wait until healthy."""
    port = find_available_port()
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


def _build_samples_with_dups(repo_id: str, concurrency: int) -> list[MMMUSample]:
    """Load the MMMU CI subset and interleave a duplicate right after each
    of the first ``concurrency // 2`` samples.

    Note (Yifei):
    Regression guard for issue #299 (cached + non-cached in the same encoder
    batch). One request almost always finishes the encoder first, and by the
    time the encoder picks up its next batch, ``concurrency - 1`` other
    requests have queued behind it. Placing dups immediately after their
    originals ensures that batch contains at least one sample whose encoder
    result is already cached (the dup) alongside fresh uncached ones —
    exactly the mixed-batch shape that exercises the #299 code path. Only
    surfaces once concurrency > 1.

    For concurrency=8 on the 50-sample ``mmmu-ci-50`` subset this produces
    54 samples ordered as
    ``[s1, s1_dup, s2, s2_dup, s3, s3_dup, s4, s4_dup, s5, s6, ..., s50]``.
    """
    base = load_mmmu_samples(repo_id=repo_id)
    dup_count = concurrency // 2
    samples: list[MMMUSample] = []
    for i, sample in enumerate(base):
        samples.append(sample)
        if i < dup_count:
            dup = deepcopy(sample)
            dup.sample_id = f"{sample.sample_id}__dup"
            samples.append(dup)
    return samples


@pytest.mark.benchmark
def test_mmmu_accuracy_and_speed(
    server_process: subprocess.Popen,
    tmp_path: Path,
) -> None:
    """Run MMMU eval and assert accuracy and speed meet thresholds."""
    config = MMMUEvalConfig(
        model="qwen3-omni",
        port=server_process.port,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "mmmu"),
        repo_id=DATASETS["mmmu-ci-50"],
    )
    samples = _build_samples_with_dups(DATASETS["mmmu-ci-50"], CONCURRENCY)
    results = asyncio.run(run_mmmu_eval(config, samples=samples))

    summary = results["summary"]
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

    speed = results["speed"]
    assert_speed_thresholds(speed, MMMU_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
