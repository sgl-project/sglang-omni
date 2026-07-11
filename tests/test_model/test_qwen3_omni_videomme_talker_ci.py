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
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.videomme import VideoMMESample, load_videomme_samples
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
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

CONCURRENCY = 16
MAX_SAMPLES = 20
MAX_TOKENS = 256
ASR_DEVICE = "cuda:0"
SHORT_ANSWER_PROMPT = (
    "For the audio response, answer briefly in one sentence and end with "
    "'Answer: $LETTER'. Do not include step-by-step reasoning."
)

VIDEOMME_TALKER_THINKER_TEXT_MIN_ACCURACY = 0.5
VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.04
VIDEOMME_TALKER_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(
    VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX
)
VIDEOMME_TALKER_N_ABOVE_50_MAX = 1.0

_VIDEOMME_TALKER_AUDIO_P95 = {
    16: {
        "throughput_qps": 0.887,
        "output_tok_per_req_s": 3.2,
        "latency_mean_s": 13.059,
        "rtf_mean": 1.5125,
    },
}
VIDEOMME_TALKER_THRESHOLDS = apply_slack(_VIDEOMME_TALKER_AUDIO_P95)

VIDEOMME_TALKER_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="videomme-ci-50",
    repo_id=DATASETS["videomme-ci-50"],
)
VIDEOMME_TALKER_WER_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="videomme-ci-50 (talker output WER)",
    repo_id=DATASETS["videomme-ci-50"],
)


def _load_short_answer_samples() -> list[VideoMMESample]:
    samples = load_videomme_samples(
        max_samples=MAX_SAMPLES,
        repo_id=DATASETS["videomme-ci-50"],
    )
    for sample in samples:
        sample.prompt = f"{sample.prompt}\n{SHORT_ANSWER_PROMPT}"
    return samples


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
    output_dir = str(tmp_path_factory.mktemp("videomme_audio"))
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_bf16_disagg_server.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=output_dir,
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        asr_device=ASR_DEVICE,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
        disable_tqdm=False,
        timeout_s=500,
    )
    with router_worker_traffic_guard(
        qwen3_omni_bf16_disagg_server,
        label="Qwen3-Omni Video-MME Talker",
    ) as router_guard:
        results = asyncio.run(
            run_video_eval(
                config,
                samples=_load_short_answer_samples(),
                task_label="Video-MME",
                output_filename="videomme_results.json",
                audio_output_dir_default="results/videomme_audio",
                compute_wer=False,
            )
        )
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
def test_videomme_talker_accuracy_and_speed(
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> None:
    """Run Video-MME with Talker enabled and assert accuracy + speed."""
    summary = talker_eval_artifacts.summary
    print_videomme_accuracy_summary(
        summary, "qwen3-omni", dataset=VIDEOMME_TALKER_DATASET_LABEL
    )
    print_speed_summary(
        talker_eval_artifacts.speed,
        "qwen3-omni",
        CONCURRENCY,
        title="Video-MME Talker Speed",
        dataset=VIDEOMME_TALKER_DATASET_LABEL,
    )

    accuracy = summary.get("accuracy")
    checks = MetricCheckCollector("Video-MME Talker accuracy and speed")
    if accuracy is None:
        checks.fail("Video-MME Talker thinker-text accuracy missing from summary")
    else:
        checks.check(
            accuracy >= VIDEOMME_TALKER_THINKER_TEXT_MIN_ACCURACY,
            f"Video-MME Talker thinker-text accuracy {accuracy:.4f} "
            f"({accuracy * 100:.1f}%) < "
            f"threshold {VIDEOMME_TALKER_THINKER_TEXT_MIN_ACCURACY} "
            f"({VIDEOMME_TALKER_THINKER_TEXT_MIN_ACCURACY * 100:.0f}%)",
        )
    assert_speed_thresholds(
        talker_eval_artifacts.speed,
        VIDEOMME_TALKER_THRESHOLDS,
        CONCURRENCY,
        collector=checks,
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_videomme_talker_wer(
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
    print_wer_summary(
        wer["summary"],
        "qwen3-omni",
        dataset=VIDEOMME_TALKER_WER_DATASET_LABEL,
    )
    persist_wer_in_benchmark_results(
        wer_eval_artifacts.audio_dir, wer, "videomme_results.json"
    )
    checks = MetricCheckCollector("Video-MME Talker WER")
    assert_wer_partitioned(
        wer,
        max_wer_below_50_corpus=VIDEOMME_TALKER_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=VIDEOMME_TALKER_N_ABOVE_50_MAX,
        collector=checks,
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
