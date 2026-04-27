# SPDX-License-Identifier: Apache-2.0
"""Video-MME Talker CI for Qwen3-Omni. Thinker-Talker ON.

Runs a small Video-MME subset through Text+Video -> Text+Audio, then checks
text answer accuracy, text-audio WER, and basic speed metrics.

Note (Chenyang, Ratish, Yifei):

Two notions of correctness are measured here:
    1. THINKER_TEXT_MIN_ACCURACY: accuracy of the Thinker's text answer
      (parsed multiple-choice letter vs. ground truth). Independent of audio.
   2. TEXT_AUDIO_WER_MAX_*: word error rate between the Thinker text and the
      ASR transcription of the Talker's synthesized audio. Measures
      text<->audio consistency, not answer correctness.

Author:
    Qiujiang Chen https://github.com/Jayon02
    Raitsh P https://github.com/Ratish1
    Chenyang Zhao https://github.com/zhaochenyang20
    Yifei Gao https://github.com/PasserBy4
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.videomme import VideoMMESample, load_videomme_samples
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
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
SHORT_ANSWER_PROMPT = (
    "For the audio response, answer briefly in one sentence and end with "
    "'Answer: $LETTER'. Do not include step-by-step reasoning."
)

# Threshold reference: https://github.com/sgl-project/sglang-omni/pull/337#issuecomment-4321051462
VIDEOMME_TALKER_THINKER_TEXT_MIN_ACCURACY = 0.5
VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.015
VIDEOMME_TALKER_N_ABOVE_50_MAX = 0

_VIDEOMME_TALKER_AUDIO_P95 = {
    8: {
        "throughput_qps": 0.077,
        "tok_per_s_agg": 0.60,
        "latency_mean_s": 79.913,
        "rtf_mean": 8.3397,
    },
}
VIDEOMME_TALKER_THRESHOLDS = apply_slack(_VIDEOMME_TALKER_AUDIO_P95)


def _load_short_answer_samples() -> list[VideoMMESample]:
    samples = load_videomme_samples(
        max_samples=MAX_SAMPLES,
        repo_id=DATASETS["videomme-ci-50"],
    )
    for sample in samples:
        sample.prompt = f"{sample.prompt}\n{SHORT_ANSWER_PROMPT}"
    return samples


@pytest.mark.benchmark
def test_videomme_tts_accuracy_wer_and_speed(
    qwen3_omni_talker_server: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run Video-MME with Talker enabled and report text/audio metrics."""
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_talker_server.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videomme_audio"),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        asr_device="cuda:0",
        disable_tqdm=False,
        timeout_s=500,
    )
    results = asyncio.run(
        run_video_eval(
            config,
            samples=_load_short_answer_samples(),
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
        title="Video-MME Talker Speed",
    )
    print_wer_summary(results["wer"]["summary"], config.model)
    assert summary["accuracy"] >= VIDEOMME_TALKER_THINKER_TEXT_MIN_ACCURACY, (
        f"Video-MME Talker thinker-text accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {VIDEOMME_TALKER_THINKER_TEXT_MIN_ACCURACY} "
        f"({VIDEOMME_TALKER_THINKER_TEXT_MIN_ACCURACY * 100:.0f}%)"
    )

    assert "wer" in results, "Audio WER results missing from Video-MME Talker output"
    assert_wer_partitioned(
        results["wer"],
        max_wer_below_50_corpus=VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX,
        max_n_above_50=VIDEOMME_TALKER_N_ABOVE_50_MAX,
    )
    assert_speed_thresholds(results["speed"], VIDEOMME_TALKER_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
