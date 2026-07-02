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
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from benchmarks.metrics.mmmu import print_mmmu_accuracy_summary
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.asr import compute_text_audio_consistency_from_records
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import (
    QWEN3_ASR_WER_CONCURRENCY,
    MetricCheckCollector,
    apply_slack,
    apply_wer_slack,
    assert_speed_thresholds,
    assert_wer_partitioned,
    persist_wer_in_benchmark_results,
    wait_for_gpu_memory_release,
)

MAX_SAMPLES = 20
MAX_TOKENS = 256
CONCURRENCY = 16
ASR_DEVICE = "cuda:0"

MMMU_TTS_PROMPT = (
    "Look at the image and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)

MMMU_AUDIO_MIN_ACCURACY = 0.7
MMMU_AUDIO_WER_BELOW_50_CORPUS_MAX = 0.2387
MMMU_AUDIO_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(
    MMMU_AUDIO_WER_BELOW_50_CORPUS_MAX
)
MMMU_AUDIO_N_ABOVE_50_MAX = 8.0

_MMMU_AUDIO_P95 = {
    16: {
        "throughput_qps": 0.609,
        "output_tok_per_req_s": 6.4,
        "latency_mean_s": 21.573,
        "rtf_mean": 0.5456,
    },
}
MMMU_AUDIO_THRESHOLDS = apply_slack(_MMMU_AUDIO_P95)


@dataclass
class _TalkerEvalArtifacts:
    summary: dict
    speed: dict
    per_sample: list
    audio_dir: str
    lang: str


@pytest.fixture(scope="module")
def talker_eval_artifacts(
    qwen3_omni_bf16_disagg_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> _TalkerEvalArtifacts:
    output_dir = str(tmp_path_factory.mktemp("mmmu_audio"))
    config = MMMUEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_bf16_disagg_server.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=output_dir,
        enable_audio=True,
        asr_device=ASR_DEVICE,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
        repo_id=DATASETS["mmmu-ci-50"],
        prompt_override=MMMU_TTS_PROMPT,
        timeout_s=500,
    )
    with router_worker_traffic_guard(
        qwen3_omni_bf16_disagg_server,
        label="Qwen3-Omni MMMU Talker",
    ) as router_guard:
        results = asyncio.run(run_mmmu_eval(config, compute_wer=False))
        router_guard.assert_served(
            min_total_requests=results["summary"].get("total_samples", 0)
        )
    return _TalkerEvalArtifacts(
        summary=results["summary"],
        speed=results["speed"],
        per_sample=results["per_sample"],
        audio_dir=str(Path(output_dir) / "audio"),
        lang=config.lang,
    )


@pytest.fixture(scope="module")
def wer_eval_artifacts(
    qwen3_omni_bf16_disagg_server: ManagedRouterHandle,
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> _TalkerEvalArtifacts:
    """Reuse saved benchmark audio for WER after freeing the talker server GPU."""
    qwen3_omni_bf16_disagg_server.stop()
    wait_for_gpu_memory_release()
    return talker_eval_artifacts


@pytest.mark.benchmark
def test_mmmu_talker_accuracy_and_speed(
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> None:
    """Run MMMU eval with audio and assert accuracy and speed meet thresholds."""
    summary = talker_eval_artifacts.summary
    print_mmmu_accuracy_summary(summary, "qwen3-omni")
    print_speed_summary(
        talker_eval_artifacts.speed,
        "qwen3-omni",
        CONCURRENCY,
        title="MMMU Talker Speed",
    )

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    checks = MetricCheckCollector("MMMU Talker accuracy and speed")
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
    assert_speed_thresholds(
        talker_eval_artifacts.speed,
        MMMU_AUDIO_THRESHOLDS,
        CONCURRENCY,
        collector=checks,
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_mmmu_talker_wer(
    wer_eval_artifacts: _TalkerEvalArtifacts,
    qwen3_asr_wer_router: ManagedRouterHandle,
) -> None:
    """Transcribe saved talker audio after the inference server is stopped."""
    wer = compute_text_audio_consistency_from_records(
        wer_eval_artifacts.per_sample,
        wer_eval_artifacts.lang,
        ASR_DEVICE,
        audio_dir=wer_eval_artifacts.audio_dir,
        asr_router_port=qwen3_asr_wer_router.port,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
    )
    print_wer_summary(wer["summary"], "qwen3-omni")
    persist_wer_in_benchmark_results(
        wer_eval_artifacts.audio_dir, wer, "mmmu_results.json"
    )
    checks = MetricCheckCollector("MMMU Talker WER")
    assert_wer_partitioned(
        wer,
        max_wer_below_50_corpus=MMMU_AUDIO_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=MMMU_AUDIO_N_ABOVE_50_MAX,
        collector=checks,
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
