# SPDX-License-Identifier: Apache-2.0
"""MMMU Talker CI for Qwen3-Omni (Text+Image → Text+Audio, Talker ON).

Evaluates text-audio consistency by comparing the model's text output
with ASR transcription of its audio output on MMMU image-QA tasks.

Usage:
    pytest tests/test_model/test_qwen3_omni_mmmu_talker_ci.py -v -s -x

Note (Chenyang):
    Currently due to the performance limitation of the Talker, we run limited
    samples for the MMMU tts CI.
    reference: https://github.com/sgl-project/sglang-omni/issues/276

Author:
    Yifei Gao https://github.com/PasserBy4
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from benchmarks.metrics.mmmu import print_mmmu_accuracy_summary
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.wer import print_wer_summary
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import (
    MetricCheckCollector,
    apply_slack,
    apply_wer_slack,
    assert_speed_thresholds,
    assert_wer_partitioned,
)

MAX_SAMPLES = 20
MAX_TOKENS = 256

CONCURRENCY = 16

# Note (Yifei): "2-3 sentences" floor prevents terse "Answer: X" replies that
# would starve the WER signal; the 120-word cap keeps p95 output well under
# MAX_TOKENS so the final 'Answer: $LETTER' line is never truncated.
MMMU_TTS_PROMPT = (
    "Look at the image and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)

# Accuracy floor — audio-mode MMMU.
MMMU_AUDIO_MIN_ACCURACY = 0.75

# WER thresholds use a partitioned view of the per-sample distribution:
#  - corpus WER over the "sane" subset (per-sample WER <= 50%)
#  - count of catastrophic failures (per-sample WER > 50%)
MMMU_AUDIO_WER_BELOW_50_CORPUS_MAX = 0.15390334572490708
MMMU_AUDIO_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(
    MMMU_AUDIO_WER_BELOW_50_CORPUS_MAX
)
MMMU_AUDIO_N_ABOVE_50_MAX = 3

_MMMU_AUDIO_P95 = {
    16: {
        "throughput_qps": 0.758,
        "output_tok_per_req_s": 8.9,
        "latency_mean_s": 14.645,
        "rtf_mean": 0.4056,
    },
}
MMMU_AUDIO_THRESHOLDS = apply_slack(_MMMU_AUDIO_P95)


@pytest.mark.benchmark
def test_mmmu_audio_wer_and_speed(
    qwen3_omni_router_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    """Run MMMU eval with audio and assert WER and speed meet thresholds."""
    config = MMMUEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_router_server.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "mmmu_audio"),
        enable_audio=True,
        repo_id=DATASETS["mmmu-ci-50"],
        prompt_override=MMMU_TTS_PROMPT,
        timeout_s=500,
    )
    with router_worker_traffic_guard(
        qwen3_omni_router_server,
        label="Qwen3-Omni MMMU Talker",
    ) as router_guard:
        results = asyncio.run(run_mmmu_eval(config))

    summary = results["summary"]
    speed = results["speed"]
    print_mmmu_accuracy_summary(summary, config.model)
    print_speed_summary(speed, config.model, CONCURRENCY, title="MMMU Talker Speed")
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], config.model)

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    checks = MetricCheckCollector("MMMU Talker accuracy, WER, and speed")
    checks.check_assertion(
        "router traffic",
        router_guard.assert_served,
        min_total_requests=total,
    )
    checks.check(
        failed == 0,
        f"MMMU Talker had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test",
    )

    accuracy = summary.get("accuracy")
    if accuracy is None:
        checks.fail("MMMU audio accuracy missing from summary")
    else:
        checks.check(
            accuracy >= MMMU_AUDIO_MIN_ACCURACY,
            f"MMMU audio accuracy {accuracy:.4f} ({accuracy * 100:.1f}%) < "
            f"threshold {MMMU_AUDIO_MIN_ACCURACY} "
            f"({MMMU_AUDIO_MIN_ACCURACY * 100:.0f}%)",
        )

    if "wer" not in results:
        checks.fail("Audio WER results missing from eval output")
    else:
        assert_wer_partitioned(
            results["wer"],
            max_wer_below_50_corpus=MMMU_AUDIO_WER_BELOW_50_CORPUS_THRESHOLD,
            max_n_above_50=MMMU_AUDIO_N_ABOVE_50_MAX,
            collector=checks,
        )

    assert_speed_thresholds(speed, MMMU_AUDIO_THRESHOLDS, CONCURRENCY, collector=checks)
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
