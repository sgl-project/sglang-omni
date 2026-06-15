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
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.tts import compute_text_audio_consistency_from_records
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

VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY = 0.5
VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX = 0.027
VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(
    VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_MAX
)
VIDEOAMME_TALKER_N_ABOVE_50_MAX = 1

_VIDEOAMME_TALKER_AUDIO_P95 = {
    16: {
        "throughput_qps": 0.679,
        "output_tok_per_req_s": 2.4,
        "latency_mean_s": 19.349,
        "rtf_mean": 1.613,
    },
}
VIDEOAMME_TALKER_THRESHOLDS = apply_slack(_VIDEOAMME_TALKER_AUDIO_P95)


@dataclass
class _TalkerEvalArtifacts:
    summary: dict
    speed: dict
    per_sample: list
    audio_dir: str
    lang: str


@pytest.fixture(scope="module")
def talker_eval_artifacts(
    qwen3_omni_talker_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> _TalkerEvalArtifacts:
    output_dir = str(tmp_path_factory.mktemp("videoamme_audio"))
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_talker_server.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=output_dir,
        repo_id=DATASETS["videoamme-ci-50"],
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
        qwen3_omni_talker_server,
        label="Qwen3-Omni Video-AMME Talker",
    ) as router_guard:
        results = asyncio.run(run_videoamme_eval(config, compute_wer=False))
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
    qwen3_omni_talker_server: ManagedRouterHandle,
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> _TalkerEvalArtifacts:
    """Reuse saved benchmark audio for WER after freeing the talker server GPU."""
    qwen3_omni_talker_server.stop()
    wait_for_gpu_memory_release()
    return talker_eval_artifacts


@pytest.mark.benchmark
def test_videoamme_talker_accuracy_and_speed(
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> None:
    """Run Video-AMME with Talker enabled and assert accuracy + speed."""
    summary = talker_eval_artifacts.summary
    print_videomme_accuracy_summary(
        summary,
        "qwen3-omni",
        title="Video-AMME Talker Accuracy",
    )
    print_speed_summary(
        talker_eval_artifacts.speed,
        "qwen3-omni",
        CONCURRENCY,
        title="Video-AMME Talker Speed",
    )

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    checks = MetricCheckCollector("Video-AMME Talker accuracy and speed")
    checks.check(
        failed == 0,
        f"Video-AMME Talker had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test",
    )
    accuracy = summary.get("accuracy")
    if accuracy is None:
        checks.fail("Video-AMME Talker thinker-text accuracy missing from summary")
    else:
        checks.check(
            accuracy >= VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY,
            f"Video-AMME Talker thinker-text accuracy {accuracy:.4f} "
            f"({accuracy * 100:.1f}%) < "
            f"threshold {VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY} "
            f"({VIDEOAMME_TALKER_THINKER_TEXT_MIN_ACCURACY * 100:.0f}%)",
        )
    assert_speed_thresholds(
        talker_eval_artifacts.speed,
        VIDEOAMME_TALKER_THRESHOLDS,
        CONCURRENCY,
        collector=checks,
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_videoamme_talker_wer(
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
        wer_eval_artifacts.audio_dir, wer, "videoamme_results.json"
    )
    checks = MetricCheckCollector("Video-AMME Talker WER")
    assert_wer_partitioned(
        wer,
        max_wer_below_50_corpus=VIDEOAMME_TALKER_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=VIDEOAMME_TALKER_N_ABOVE_50_MAX,
        collector=checks,
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
