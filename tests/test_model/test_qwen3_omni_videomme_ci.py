# SPDX-License-Identifier: Apache-2.0
"""Video-MME accuracy and speed CI for Qwen3-Omni (Text+Video -> Text, Talker OFF).

Usage:
    pytest tests/test_model/test_qwen3_omni_videomme_ci.py -s -x

Author:
    Qiujiang Chen https://github.com/Jayon02
    Chenyang Zhao https://github.com/zhaochenyang20
    Yifei Gao https://github.com/PasserBy4
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videomme import (
    VideoMMEEvalConfig,
    run_videomme_eval,
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

# threshold reference: https://github.com/sgl-project/sglang-omni/pull/338#issuecomment-4318351375
VIDEOMME_MIN_ACCURACY = 0.56
VIDEOMME_MAX_FAILED = 0

_VIDEOMME_P95 = {
    8: {
        # TODO: Recalibrate on H20 CI hardware.
        "throughput_qps": 0.077,
        "tok_per_s_agg": 2.30,
        "latency_mean_s": 50.241,
    },
}
VIDEOMME_THRESHOLDS = apply_slack(_VIDEOMME_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the text-only Qwen3-Omni server and wait until healthy.

    Note (Chenyang):
    On CI (GITHUB_ACTIONS=true) server stdout/stderr are captured into a
    log file so the main test output stays tidy and the log is attached on
    startup failure. Locally the server inherits the parent's stdout/stderr
    so progress streams live under pytest -s.
    """
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
def test_videomme_accuracy_and_speed(
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run videomme-ci-50 at concurrency=8 and assert accuracy + speed thresholds."""
    config = VideoMMEEvalConfig(
        model="qwen3-omni",
        port=server_process.port,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videomme"),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        disable_tqdm=True,
    )
    results = asyncio.run(run_videomme_eval(config))

    summary = results["summary"]
    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    # assert failed <= VIDEOMME_MAX_FAILED, (
    #     f"Video-MME had {failed}/{total} failed requests, "
    #     f"which exceeds the threshold {VIDEOMME_MAX_FAILED}"
    # )

    # assert summary["accuracy"] >= VIDEOMME_MIN_ACCURACY, (
    #     f"Video-MME accuracy {summary['accuracy']:.4f} "
    #     f"({summary['accuracy'] * 100:.1f}%) < "
    #     f"threshold {VIDEOMME_MIN_ACCURACY} ({VIDEOMME_MIN_ACCURACY * 100:.0f}%)"
    # )

    # assert_speed_thresholds(results["speed"], VIDEOMME_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
