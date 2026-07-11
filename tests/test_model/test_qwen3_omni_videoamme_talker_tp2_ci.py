# SPDX-License-Identifier: Apache-2.0
"""Video-AMME Talker TP=2 CI for Qwen3-Omni FP8 (Video+Audio -> Text+Audio).

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
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.asr import compute_text_audio_consistency_from_records
from tests.test_model.omni_router_utils import ManagedRouterHandle
from tests.utils import (
    QWEN3_ASR_WER_CONCURRENCY,
    MetricCheckCollector,
    ServerHandle,
    apply_slack,
    apply_wer_slack,
    assert_speed_thresholds,
    assert_wer_partitioned,
    persist_wer_in_benchmark_results,
    stop_server,
    wait_for_gpu_memory_release,
)

CONCURRENCY = 16
MAX_SAMPLES = 10
MAX_TOKENS = 256
ASR_DEVICE = "cuda:0"

VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY = 0.4
VIDEOAMME_TALKER_TP2_WER_BELOW_50_CORPUS_MAX = 0.0548
VIDEOAMME_TALKER_TP2_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(
    VIDEOAMME_TALKER_TP2_WER_BELOW_50_CORPUS_MAX
)
VIDEOAMME_TALKER_TP2_N_ABOVE_50_MAX = 0.0

_VIDEOAMME_TALKER_TP2_AUDIO_P95 = {
    16: {
        "throughput_qps": 0.058,
        "output_tok_per_req_s": 0.3,
        "latency_mean_s": 170.36,
        "rtf_mean": 12.8415,
    },
}
VIDEOAMME_TALKER_TP2_THRESHOLDS = apply_slack(_VIDEOAMME_TALKER_TP2_AUDIO_P95)

# note (Yue Yin, Chenyang): 0.55-calibrated latency baseline (146.7 gate) is
# unreachable on the #765 0.40 OOM-fix config (~148s); pin the gate to 155

VIDEOAMME_TALKER_TP2_THRESHOLDS[16]["latency_mean_s_max"] = 155

VIDEOAMME_TALKER_TP2_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="videoamme-ci-50",
    repo_id=DATASETS["videoamme-ci-50"],
)
VIDEOAMME_TALKER_TP2_WER_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="videoamme-ci-50 (talker output WER)",
    repo_id=DATASETS["videoamme-ci-50"],
)


@dataclass
class _TalkerEvalArtifacts:
    summary: dict
    speed: dict
    per_sample: list
    audio_dir: str
    lang: str


@pytest.mark.benchmark
def test_thinker_tp2_actually_applied(
    qwen3_omni_fp8_tp2_server: ServerHandle,
) -> None:
    """Confirm the thinker stage actually came up at tp_size=2.
    Prevents silent fallback to TP=1
    """
    log_file = qwen3_omni_fp8_tp2_server.log_file
    checks = MetricCheckCollector("Thinker TP=2 server log checks")
    checks.check(
        log_file is not None and log_file.exists(),
        "TP=2 fixture did not capture a server log - check that the fixture "
        "passes log_file=... to ServerHandle",
    )
    if log_file is None or not log_file.exists():
        checks.assert_all()
        return
    text = log_file.read_text()
    checks.check(
        "tp_rank=0/2" in text,
        f"Thinker leader (rank 0) is not running at tp_size=2; "
        f"'tp_rank=0/2' missing from server log:\n{text[-2000:]}",
    )
    checks.check(
        "tp_rank=1/2" in text,
        f"Thinker follower (rank 1) did not come up; 'tp_rank=1/2' "
        f"missing from server log:\n{text[-2000:]}",
    )
    checks.assert_all()


@pytest.fixture(scope="module")
def talker_eval_artifacts(
    qwen3_omni_fp8_tp2_server: ServerHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> _TalkerEvalArtifacts:
    output_dir = str(tmp_path_factory.mktemp("videoamme_audio"))
    config = VideoEvalConfig(
        model="qwen3-omni",
        port=qwen3_omni_fp8_tp2_server.port,
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
    results = asyncio.run(run_videoamme_eval(config, compute_wer=False))
    return _TalkerEvalArtifacts(
        summary=results["summary"],
        speed=results["speed"],
        per_sample=results["per_sample"],
        audio_dir=str(Path(output_dir) / "audio"),
        lang=config.lang,
    )


@pytest.fixture(scope="module")
def wer_eval_artifacts(
    qwen3_omni_fp8_tp2_server: ServerHandle,
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> _TalkerEvalArtifacts:
    """Reuse saved benchmark audio for WER after freeing the talker server GPU."""
    stop_server(qwen3_omni_fp8_tp2_server.proc)
    wait_for_gpu_memory_release()
    return talker_eval_artifacts


@pytest.mark.benchmark
def test_videoamme_talker_tp2_accuracy_and_speed(
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> None:
    """Run Video-AMME with TP=2 thinker + Talker enabled."""
    summary = talker_eval_artifacts.summary
    print_videomme_accuracy_summary(
        summary,
        "qwen3-omni",
        title="Video-AMME Talker TP=2 Accuracy",
        dataset=VIDEOAMME_TALKER_TP2_DATASET_LABEL,
    )
    print_speed_summary(
        talker_eval_artifacts.speed,
        "qwen3-omni",
        CONCURRENCY,
        title="Video-AMME Talker TP=2 Speed",
        dataset=VIDEOAMME_TALKER_TP2_DATASET_LABEL,
    )

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    checks = MetricCheckCollector("Video-AMME Talker TP=2 accuracy and speed")
    checks.check(
        failed == 0,
        f"Video-AMME Talker TP=2 had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test",
    )
    accuracy = summary.get("accuracy")
    if accuracy is None:
        checks.fail("Video-AMME Talker TP=2 thinker-text accuracy missing from summary")
    else:
        checks.check(
            accuracy >= VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY,
            f"Video-AMME Talker TP=2 thinker-text accuracy {accuracy:.4f} "
            f"({accuracy * 100:.1f}%) < "
            f"threshold {VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY} "
            f"({VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY * 100:.0f}%)",
        )
    assert_speed_thresholds(
        talker_eval_artifacts.speed,
        VIDEOAMME_TALKER_TP2_THRESHOLDS,
        CONCURRENCY,
        collector=checks,
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_videoamme_talker_tp2_wer(
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
        dataset=VIDEOAMME_TALKER_TP2_WER_DATASET_LABEL,
    )
    persist_wer_in_benchmark_results(
        wer_eval_artifacts.audio_dir, wer, "videoamme_results.json"
    )
    checks = MetricCheckCollector("Video-AMME Talker TP=2 WER")
    assert_wer_partitioned(
        wer,
        max_wer_below_50_corpus=VIDEOAMME_TALKER_TP2_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=VIDEOAMME_TALKER_TP2_N_ABOVE_50_MAX,
        collector=checks,
    )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
