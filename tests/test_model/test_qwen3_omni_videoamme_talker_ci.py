# SPDX-License-Identifier: Apache-2.0
"""Video-AMME Talker CI for Qwen3-Omni (Video+Audio -> Text+Audio).

Runs a small Video-AMME subset through Video+Audio -> Text+Audio, then checks
text answer accuracy, text-audio WER, and basic speed metrics.

Usage:
    pytest tests/test_model/test_qwen3_omni_videoamme_talker_ci.py -v -s -x

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
from benchmarks.tasks.tts import print_speed_summary, print_wer_summary
from benchmarks.tasks.video_understanding import print_videomme_accuracy_summary
from sglang_omni.utils import find_available_port
from tests.utils import (
    ServerHandle,
    apply_slack,
    assert_speed_thresholds,
    assert_wer_partitioned,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

CONCURRENCY = 8
MAX_SAMPLES = 10
MAX_TOKENS = 256
STARTUP_TIMEOUT = 900

# TODO: Recalibrate thresholds on H20.
VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY = 0.40
VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.10
VIDEOAMME_TALKER_N_ABOVE_50_MAX = 0

_VIDEOAMME_TALKER_AUDIO_P95 = {
    8: {
        "throughput_qps": 0.05,
        "tok_per_s_agg": 0.40,
        "latency_mean_s": 150.0,
        "rtf_mean": 10.0,
    },
}
VIDEOAMME_TALKER_THRESHOLDS = apply_slack(_VIDEOAMME_TALKER_AUDIO_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Qwen3-Omni speech server and wait until healthy."""
    port = find_available_port()
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    log_file: Path | None = (
        tmp_path_factory.mktemp("server_logs") / "server.log" if is_ci else None
    )
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_speech_server.py",
        "--model-path",
        MODEL_PATH,
        "--gpu-thinker",
        "0",
        "--gpu-talker",
        "1",
        "--gpu-code-predictor",
        "1",
        "--gpu-code2wav",
        "1",
        "--port",
        str(port),
        "--model-name",
        "qwen3-omni",
        "--thinker-max-seq-len",
        "32768",
        "--thinker-mem-fraction-static",
        "0.78",
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


@pytest.mark.benchmark
def test_videoamme_talker_accuracy_wer_and_speed(
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run Video-AMME with Talker enabled and assert text/audio metrics."""
    config = VideoAMMEEvalConfig(
        model="qwen3-omni",
        port=server_process.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videoamme_audio"),
        repo_id=DATASETS["video-amme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        asr_device="cuda:0",
        disable_tqdm=False,
        timeout_s=500,
    )
    results = asyncio.run(run_video_amme_eval(config))

    summary = results["summary"]
    print_videomme_accuracy_summary(
        summary,
        config.model,
        title="Video-AMME Talker Accuracy",
    )
    print_speed_summary(
        results["speed"],
        config.model,
        CONCURRENCY,
        title="Video-AMME Talker Speed",
    )
    print_wer_summary(results["wer"]["summary"], config.model)

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    assert failed == 0, (
        f"Video-AMME Talker had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test"
    )
    assert summary["accuracy"] >= VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY, (
        f"Video-AMME Talker thinker-text accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY} "
        f"({VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY * 100:.0f}%)"
    )

    assert "wer" in results, "Audio WER results missing from Video-AMME Talker output"
    assert_wer_partitioned(
        results["wer"],
        max_wer_below_50_corpus=VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX,
        max_n_above_50=VIDEOAMME_TALKER_N_ABOVE_50_MAX,
    )
    assert_speed_thresholds(results["speed"], VIDEOAMME_TALKER_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
