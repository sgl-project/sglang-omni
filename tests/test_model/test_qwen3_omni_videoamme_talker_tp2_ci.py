# SPDX-License-Identifier: Apache-2.0
"""Video-AMME Talker TP=2 CI for Qwen3-Omni (Video+Audio -> Text+Audio).

Runs a small Video-AMME subset through Video+Audio -> Text+Audio with the
thinker stage sharded across two GPUs (tp_size=2), then checks text answer
accuracy, text-audio WER, and basic speed metrics.

Usage:
    pytest tests/test_model/test_qwen3_omni_videoamme_talker_tp2_ci.py -v -s -x

Author:
    Yichi Zhang https://github.com/Ccyest
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
    apply_wer_slack,
    assert_speed_thresholds,
    assert_wer_partitioned,
)

CONCURRENCY = 8
MAX_SAMPLES = 10
MAX_TOKENS = 256

VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY = 0.4
VIDEOAMME_TALKER_TP2_WER_BELOW_50_CORPUS_MAX = 0.01
VIDEOAMME_TALKER_TP2_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(
    VIDEOAMME_TALKER_TP2_WER_BELOW_50_CORPUS_MAX
)
VIDEOAMME_TALKER_TP2_N_ABOVE_50_MAX = 1

_VIDEOAMME_TALKER_TP2_AUDIO_P95 = {
    8: {
        "throughput_qps": 0.064,
        "tok_per_s_agg": 0.4,
        "latency_mean_s": 107.537,
        "rtf_mean": 18.622,
    },
}
VIDEOAMME_TALKER_TP2_THRESHOLDS = apply_slack(_VIDEOAMME_TALKER_TP2_AUDIO_P95)


@pytest.mark.benchmark
def test_thinker_tp2_actually_applied(
    qwen3_omni_talker_server_tp2: ServerHandle,
) -> None:
    """Confirm the thinker stage actually came up at tp_size=2.
    Prevents silent fallback to TP=1
    """
    log_file = qwen3_omni_talker_server_tp2.log_file
    assert log_file is not None and log_file.exists(), (
        "TP=2 fixture did not capture a server log — check that the fixture "
        "passes log_file=... to ServerHandle"
    )
    text = log_file.read_text()
    assert "tp_rank=0/2" in text, (
        f"Thinker leader (rank 0) is not running at tp_size=2; "
        f"'tp_rank=0/2' missing from server log:\n{text[-2000:]}"
    )
    assert "tp_rank=1/2" in text, (
        f"Thinker follower (rank 1) did not come up; 'tp_rank=1/2' "
        f"missing from server log:\n{text[-2000:]}"
    )


@pytest.mark.benchmark
def test_videoamme_talker_tp2_accuracy_wer_and_speed(
    qwen3_omni_talker_server_tp2: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run Video-AMME with TP=2 thinker + Talker enabled."""
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_talker_server_tp2.port,
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
        title="Video-AMME Talker TP=2 Accuracy",
    )
    print_speed_summary(
        results["speed"],
        config.model,
        CONCURRENCY,
        title="Video-AMME Talker TP=2 Speed",
    )
    print_wer_summary(results["wer"]["summary"], config.model)
    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    assert failed == 0, (
        f"Video-AMME Talker TP=2 had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test"
    )
    assert summary["accuracy"] >= VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY, (
        f"Video-AMME Talker TP=2 thinker-text accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY} "
        f"({VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY * 100:.0f}%)"
    )

    assert (
        "wer" in results
    ), "Audio WER results missing from Video-AMME Talker TP=2 output"
    assert_wer_partitioned(
        results["wer"],
        max_wer_below_50_corpus=VIDEOAMME_TALKER_TP2_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=VIDEOAMME_TALKER_TP2_N_ABOVE_50_MAX,
    )
    assert_speed_thresholds(
        results["speed"], VIDEOAMME_TALKER_TP2_THRESHOLDS, CONCURRENCY
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
