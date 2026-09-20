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
from tests.test_model.omni_ci_config import OmniCiModelPreset
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    router_worker_traffic_guard,
)
from tests.utils import (
    QWEN3_ASR_WER_CONCURRENCY,
    MetricCheckCollector,
    ServerHandle,
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
    omni_ci_model: OmniCiModelPreset,
    omni_ci_server: ServerHandle | ManagedRouterHandle,
) -> None:
    """Confirm the thinker stage actually came up at tp_size=2.
    Prevents silent fallback to TP=1
    """
    if omni_ci_model.name != "qwen3-omni":
        pytest.skip("MiniCPM-o uses DP2 rather than thinker TP2")
    log_file = omni_ci_server.log_file
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
    omni_ci_model: OmniCiModelPreset,
    omni_ci_server: ServerHandle | ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> _TalkerEvalArtifacts:
    output_dir = str(tmp_path_factory.mktemp("videoamme_audio"))
    config = VideoEvalConfig(
        model=omni_ci_model.name,
        port=omni_ci_server.port,
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
    if isinstance(omni_ci_server, ManagedRouterHandle):
        with router_worker_traffic_guard(
            omni_ci_server, label=f"{omni_ci_model.name} Video-AMME Talker"
        ) as router_guard:
            results = asyncio.run(run_videoamme_eval(config, compute_wer=False))
            router_guard.assert_served(
                min_total_requests=results["summary"].get("total_samples", 0)
            )
    else:
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
    omni_ci_server: ServerHandle | ManagedRouterHandle,
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> _TalkerEvalArtifacts:
    """Reuse saved benchmark audio for WER after freeing the talker server GPU."""
    if isinstance(omni_ci_server, ManagedRouterHandle):
        omni_ci_server.stop()
    else:
        stop_server(omni_ci_server.proc)
    wait_for_gpu_memory_release()
    return talker_eval_artifacts


@pytest.mark.benchmark
def test_videoamme_talker_tp2_accuracy_and_speed(
    omni_ci_model: OmniCiModelPreset,
    talker_eval_artifacts: _TalkerEvalArtifacts,
) -> None:
    """Run Video-AMME with the selected model and Talker enabled."""
    topology = "TP=2" if omni_ci_model.name == "qwen3-omni" else "DP=2"
    summary = talker_eval_artifacts.summary
    print_videomme_accuracy_summary(
        summary,
        omni_ci_model.name,
        title=f"Video-AMME Talker {topology} Accuracy",
        dataset=VIDEOAMME_TALKER_TP2_DATASET_LABEL,
    )
    print_speed_summary(
        talker_eval_artifacts.speed,
        omni_ci_model.name,
        CONCURRENCY,
        title=f"Video-AMME Talker {topology} Speed",
        dataset=VIDEOAMME_TALKER_TP2_DATASET_LABEL,
    )

    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    thresholds = omni_ci_model.thresholds["videoamme_talker"]
    checks = MetricCheckCollector(f"Video-AMME Talker {topology} accuracy and speed")
    checks.check(
        failed == 0,
        f"Video-AMME Talker {topology} had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test",
    )
    accuracy = summary.get("accuracy")
    if accuracy is None:
        checks.fail(
            f"Video-AMME Talker {topology} thinker-text accuracy missing from summary"
        )
    elif thresholds.calibrated:
        checks.check(
            accuracy >= thresholds.accuracy,
            f"Video-AMME Talker {topology} thinker-text accuracy {accuracy:.4f} "
            f"({accuracy * 100:.1f}%) < "
            f"threshold {thresholds.accuracy} "
            f"({thresholds.accuracy * 100:.0f}%)",
        )
    if thresholds.calibrated:
        assert_speed_thresholds(
            talker_eval_artifacts.speed,
            thresholds.speed,
            CONCURRENCY,
            collector=checks,
        )
    thresholds.require_calibrated(omni_ci_model.name, "videoamme_talker", checks)
    checks.assert_all()


@pytest.mark.benchmark
def test_videoamme_talker_tp2_wer(
    omni_ci_model: OmniCiModelPreset,
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
        omni_ci_model.name,
        dataset=VIDEOAMME_TALKER_TP2_WER_DATASET_LABEL,
    )
    persist_wer_in_benchmark_results(
        wer_eval_artifacts.audio_dir, wer, "videoamme_results.json"
    )
    thresholds = omni_ci_model.thresholds["videoamme_talker"]
    topology = "TP=2" if omni_ci_model.name == "qwen3-omni" else "DP=2"
    checks = MetricCheckCollector(f"Video-AMME Talker {topology} WER")
    assert_wer_partitioned(
        wer,
        max_wer_below_50_corpus=(
            thresholds.wer if thresholds.calibrated else float("inf")
        ),
        max_n_above_50=(
            thresholds.n_above_50 if thresholds.calibrated else float("inf")
        ),
        collector=checks,
    )
    thresholds.require_calibrated(omni_ci_model.name, "videoamme_talker", checks)
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
