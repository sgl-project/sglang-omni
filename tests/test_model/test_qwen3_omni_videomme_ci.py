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
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from tests.utils import ServerHandle, apply_slack, assert_speed_thresholds

CONCURRENCY = 16
MAX_SAMPLES = 30

# threshold reference: https://github.com/sgl-project/sglang-omni/pull/367#issue-4333687689
VIDEOMME_MIN_ACCURACY = 0.53

_VIDEOMME_P95 = {
    16: {
        "throughput_qps": 0.127,
        "tok_per_s_agg": 0.90,
        "latency_mean_s": 121.264,
    },
}
VIDEOMME_THRESHOLDS = apply_slack(_VIDEOMME_P95)


@pytest.mark.benchmark
def test_videomme_accuracy_and_speed(
    qwen3_omni_thinker_server: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run first 30 of videomme-ci-50 at concurrency=16 and report accuracy + speed."""
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_thinker_server.port,
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videomme"),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        disable_tqdm=False,
    )
    results = asyncio.run(
        run_video_eval(
            config,
            task_label="Video-MME",
            output_filename="videomme_results.json",
            audio_output_dir_default="results/videomme_audio",
        )
    )

    summary = results["summary"]
    print_videomme_accuracy_summary(summary, config.model)
    print_speed_summary(
        results["speed"],
        config.model,
        CONCURRENCY,
        title="Video-MME Speed",
    )
    assert summary["accuracy"] >= VIDEOMME_MIN_ACCURACY, (
        f"Video-MME accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {VIDEOMME_MIN_ACCURACY} ({VIDEOMME_MIN_ACCURACY * 100:.0f}%)"
    )
    assert_speed_thresholds(results["speed"], VIDEOMME_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
