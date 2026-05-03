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
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from benchmarks.metrics.wer import print_wer_summary
from tests.utils import (
    ServerHandle,
    apply_slack,
    assert_speed_thresholds,
    assert_wer_partitioned,
)

CONCURRENCY = 8
MAX_SAMPLES = 10
MAX_TOKENS = 256

# Threshold reference: https://github.com/sgl-project/sglang-omni/pull/382#issuecomment-4366925373
# Relaxed in V1 refactor: v0=0.5 → v1=0.4.
VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY = 0.4
# Relaxed in V1 refactor: v0=0.015 → v1=0.02.
VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.02
# Relaxed in V1 refactor: v0=1 → v1=2.
VIDEOAMME_TALKER_N_ABOVE_50_MAX = 2

_VIDEOAMME_TALKER_AUDIO_P95 = {
    8: {
        "throughput_qps": 0.149,
        "tok_per_s_agg": 1.0,
        "latency_mean_s": 45.381,
        "rtf_mean": 5.6402,
    },
}
VIDEOAMME_TALKER_THRESHOLDS = apply_slack(_VIDEOAMME_TALKER_AUDIO_P95)


@pytest.mark.benchmark
def test_videoamme_talker_accuracy_wer_and_speed(
    qwen3_omni_talker_server: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run Video-AMME with Talker enabled and report text/audio metrics."""
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_talker_server.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videoamme_audio"),
        repo_id=DATASETS["videoamme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        asr_device="cuda:0",
        disable_tqdm=False,
        timeout_s=500,
    )
    results = asyncio.run(run_videoamme_eval(config))

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
