# SPDX-License-Identifier: Apache-2.0
"""MMSU Talker CI for Qwen3-Omni (Text + Audio → Text+Audio, Talker ON).

Evaluates text-audio consistency by comparing the model's text output with
ASR transcription of its audio output on MMSU audio-QA tasks. Uses a
chain-of-thought prompt (mirroring MMMU style) so the model reasons step
by step before giving the final answer letter, producing longer responses
more suitable for WER evaluation.

Usage:
    pytest tests/test_model/test_qwen3_omni_mmsu_talker_ci.py -v -s -x

Author:
    Yifei Gao https://github.com/PasserBy4
    Huapeng Zhou https://github.com/PopSoda2002
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.mmsu import print_mmsu_summary
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

MAX_SAMPLES = 40
MAX_TOKENS = 256
CONCURRENCY = 16
ASR_DEVICE = "cuda:0"

MMSU_TTS_PROMPT = (
    "Listen to the audio and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)

MMSU_AUDIO_MIN_ACCURACY = 0.625
MMSU_AUDIO_WER_BELOW_50_CORPUS_MAX = 0.0258
MMSU_AUDIO_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(
    MMSU_AUDIO_WER_BELOW_50_CORPUS_MAX
)
MMSU_AUDIO_N_ABOVE_50_MAX = 1.0

_MMSU_AUDIO_P95 = {
    16: {
        "throughput_qps": 1.252,
        "output_tok_per_req_s": 5.2,
        "latency_mean_s": 12.072,
        "rtf_mean": 0.6445,
    },
}
MMSU_AUDIO_THRESHOLDS = apply_slack(_MMSU_AUDIO_P95)

MMSU_TALKER_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="mmsu-ci-2000",
    repo_id=DATASETS["mmsu-ci-2000"],
)
MMSU_TALKER_WER_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="mmsu-ci-2000 (talker output WER)",
    repo_id=DATASETS["mmsu-ci-2000"],
)


def _build_args(port: int, output_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=None,
        host="localhost",
        port=port,
        model="qwen3-omni",
        modalities="text+audio",
        output_dir=output_dir,
        max_samples=MAX_SAMPLES,
        task_names=None,
        categories=None,
        prompt=MMSU_TTS_PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        warmup=0,
        max_concurrency=CONCURRENCY,
        request_rate=float("inf"),
        save_audio=True,
        disable_tqdm=False,
        seed=None,
        lang="en",
        asr_device=ASR_DEVICE,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
        timeout_s=500,
    )


@dataclass
class _TalkerEvalArtifacts:
    accuracy: dict
    speed: dict
    per_sample: list
    audio_dir: str
    lang: str


@pytest.fixture(scope="module")
def talker_eval_artifacts(
    qwen3_omni_fp8_tp2_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> _TalkerEvalArtifacts:
    output_dir = str(tmp_path_factory.mktemp("mmsu_audio"))
    args = _build_args(qwen3_omni_fp8_tp2_server.port, output_dir)
    samples = load_mmsu_samples(
        max_samples=MAX_SAMPLES, repo_id=DATASETS["mmsu-ci-2000"]
    )
    with router_worker_traffic_guard(
        qwen3_omni_fp8_tp2_server,
        label="Qwen3-Omni MMSU Talker",
    ) as router_guard:
        results = asyncio.run(run_mmsu(args, samples=samples, compute_wer=False))
        router_guard.assert_served(
            min_total_requests=results["accuracy"].get("total_samples", 0)
        )
    return _TalkerEvalArtifacts(
        accuracy=results["accuracy"],
        speed=results["speed"],
        per_sample=results["per_sample"],
        audio_dir=str(Path(output_dir) / "audio"),
        lang=args.lang,
    )


@pytest.fixture(scope="module")
def wer_eval_artifacts(
    qwen3_omni_fp8_tp2_server: ManagedRouterHandle,
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> _TalkerEvalArtifacts:
    """Reuse saved benchmark audio for WER after freeing the talker server GPU."""
    qwen3_omni_fp8_tp2_server.stop()
    wait_for_gpu_memory_release()
    return talker_eval_artifacts


@pytest.mark.benchmark
def test_mmsu_talker_accuracy_and_speed(
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> None:
    """Run MMSU eval with audio and assert accuracy and speed meet thresholds."""
    print_mmsu_summary(
        talker_eval_artifacts.accuracy,
        "qwen3-omni",
        speed_metrics=talker_eval_artifacts.speed,
        dataset=MMSU_TALKER_DATASET_LABEL,
    )

    failed = talker_eval_artifacts.accuracy.get("failed_samples", 0)
    total = talker_eval_artifacts.accuracy.get("total_samples", 0)
    checks = MetricCheckCollector("MMSU Talker accuracy and speed")
    checks.check(
        failed == 0,
        f"MMSU Talker had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test",
    )
    accuracy = talker_eval_artifacts.accuracy.get("overall_accuracy")
    if accuracy is None:
        checks.fail("MMSU audio overall_accuracy missing from accuracy results")
    else:
        checks.check(
            accuracy >= MMSU_AUDIO_MIN_ACCURACY,
            f"MMSU audio accuracy {accuracy:.4f} ({accuracy * 100:.1f}%) < "
            f"threshold {MMSU_AUDIO_MIN_ACCURACY} "
            f"({MMSU_AUDIO_MIN_ACCURACY * 100:.0f}%)",
        )
    assert_speed_thresholds(
        talker_eval_artifacts.speed,
        MMSU_AUDIO_THRESHOLDS,
        CONCURRENCY,
        collector=checks,
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_mmsu_talker_wer(
    wer_eval_artifacts: _TalkerEvalArtifacts,
    qwen3_asr_wer_router: ManagedRouterHandle,
) -> None:
    """Transcribe saved talker audio after the inference server is stopped."""
    wer = compute_text_audio_consistency_from_records(
        wer_eval_artifacts.per_sample,
        wer_eval_artifacts.lang,
        ASR_DEVICE,
        audio_dir=wer_eval_artifacts.audio_dir,
        text_key="raw_response",
        asr_router_port=qwen3_asr_wer_router.port,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
    )
    print_wer_summary(
        wer["summary"], "qwen3-omni", dataset=MMSU_TALKER_WER_DATASET_LABEL
    )
    persist_wer_in_benchmark_results(
        wer_eval_artifacts.audio_dir, wer, "mmsu_results.json"
    )
    checks = MetricCheckCollector("MMSU Talker WER")
    assert_wer_partitioned(
        wer,
        max_wer_below_50_corpus=MMSU_AUDIO_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=MMSU_AUDIO_N_ABOVE_50_MAX,
        collector=checks,
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
