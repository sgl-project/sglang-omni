# SPDX-License-Identifier: Apache-2.0
"""Video-AMME accuracy and speed CI for Qwen3-Omni (Video+Audio -> Text).

Usage:
    pytest tests/test_model/test_qwen3_omni_videoamme_ci.py -s -x

Author:
    Ratish P https://github.com/Ratish21
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_video_amme import (
    VideoAMMEEvalConfig,
    run_video_amme_eval,
)
from sglang_omni.utils import find_available_port
from tests.utils import (
    ServerHandle,
    apply_slack,
    assert_speed_thresholds,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

CONCURRENCY = 8
STARTUP_TIMEOUT = 900

# TODO: Recalibrate thresholds on H20.
VIDEOAMME_MIN_ACCURACY = 0.40

_VIDEOAMME_P95 = {
    8: {
        "throughput_qps": 0.05,
        "tok_per_s_agg": 0.30,
        "latency_mean_s": 140.0,
    },
}
VIDEOAMME_THRESHOLDS = apply_slack(_VIDEOAMME_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the text-only Qwen3-Omni server and wait until healthy."""
    port = find_available_port()
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    log_file: Path | None = (
        tmp_path_factory.mktemp("server_logs") / "server.log" if is_ci else None
    )
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_server.py",
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--model-name",
        "qwen3-omni",
        "--thinker-max-seq-len",
        "32768",
        "--mem-fraction-static",
        "0.78",
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


@pytest.mark.benchmark
def test_videoamme_accuracy_and_speed(
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run video-amme-ci-50 at concurrency=8 and assert accuracy + speed."""
    config = VideoAMMEEvalConfig(
        model="qwen3-omni",
        port=server_process.port,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videoamme"),
        repo_id=DATASETS["video-amme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        disable_tqdm=False,
        timeout_s=500,
    )
    results = asyncio.run(run_video_amme_eval(config))

    summary = results["summary"]
    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    assert failed == 0, (
        f"Video-AMME had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test"
    )
    assert summary["accuracy"] >= VIDEOAMME_MIN_ACCURACY, (
        f"Video-AMME accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {VIDEOAMME_MIN_ACCURACY} ({VIDEOAMME_MIN_ACCURACY * 100:.0f}%)"
    )

    assert_speed_thresholds(results["speed"], VIDEOAMME_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
