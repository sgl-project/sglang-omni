# SPDX-License-Identifier: Apache-2.0
"""SeedTTS ASR correctness CI for Fun-ASR-Nano.

The test reuses the shared ASR SeedTTS harness against the SGLang Omni
Fun-ASR router. The English split gates WER and speed at the calibrated
concurrency. The Chinese split gates WER as well, since Fun-ASR-Nano targets
multilingual and CJK transcription and a gate on English alone would miss
regressions on its primary use case. Speed is gated on the English split
only, because a second speed gate on the same topology would add flake
surface without new signal.

Author:
Aaron Tian: https://github.com/db-ol
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_asr_seedtts import (
    FUN_ASR_MODEL_PATH,
    run_asr_seedtts_once,
)
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.wer import print_asr_speed_summary, print_asr_wer_summary
from benchmarks.tasks.asr import DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, apply_wer_slack

FUN_ASR_CI_MODEL_PATH = FUN_ASR_MODEL_PATH
FUN_ASR_CONCURRENCY = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
FUN_ASR_WARMUP_REQUESTS = FUN_ASR_CONCURRENCY * 2
SEEDTTS_FUN_ASR_EN_SAMPLES = 1088
SEEDTTS_FUN_ASR_ZH_SAMPLES = 2020
SEEDTTS_ASR_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="seedtts",
    repo_id=DATASETS["seedtts"],
)

# note (db-ol): references are the strict worst-of-5 from the official
# tune-ci-thresholds calibration run 20260719T2302Z_asr_fun_g25 on idle
# GPUs of the CI host (2x H100 80GB, DP=2 managed router, concurrency 32).
# EN WER was identical across all five repeats and speed CV stayed under
# 7 percent with no outliers. Two caveats the numbers alone do not show:
# heavily contended CI runs have measured 16 to 20 percent below the idle
# speed references, beyond the 10 percent slack, so a loaded runner leans
# on the flaky-pytest retry wrapper. The ZH per-sample bound keeps the more
# conservative worst of the two strict calibrations because that single
# utterance metric drifted between runs (0.75 in this run, 0.8333 in the
# previous strict worst-of-5 on the same topology).
FUN_ASR_EN_CORPUS_WER_MAX = 0.0172
FUN_ASR_EN_SAMPLE_WER_MAX = 0.2858
FUN_ASR_ZH_CORPUS_WER_MAX = 0.0136
FUN_ASR_ZH_SAMPLE_WER_MAX = 0.8334
FUN_ASR_THROUGHPUT_MIN = 64.27181519213258
FUN_ASR_LATENCY_MEAN_MAX_S = 0.49501860166257006
FUN_ASR_LATENCY_P95_MAX_S = 0.715
FUN_ASR_RTF_MEAN_MAX = 0.1076
FUN_ASR_RTF_P95_MAX = 0.1597

THRESHOLD_SLACK_HIGHER = 0.9
THRESHOLD_SLACK_LOWER = 1.1

FUN_ASR_EN_CORPUS_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_EN_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_EN_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_EN_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_ZH_CORPUS_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_ZH_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_ZH_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    FUN_ASR_ZH_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
FUN_ASR_THROUGHPUT_THRESHOLD = round(FUN_ASR_THROUGHPUT_MIN * THRESHOLD_SLACK_HIGHER, 3)
FUN_ASR_LATENCY_MEAN_THRESHOLD_S = round(
    FUN_ASR_LATENCY_MEAN_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
FUN_ASR_LATENCY_P95_THRESHOLD_S = round(
    FUN_ASR_LATENCY_P95_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
FUN_ASR_RTF_MEAN_THRESHOLD = round(FUN_ASR_RTF_MEAN_MAX * THRESHOLD_SLACK_LOWER, 4)
FUN_ASR_RTF_P95_THRESHOLD = round(FUN_ASR_RTF_P95_MAX * THRESHOLD_SLACK_LOWER, 4)
STARTUP_TIMEOUT = 600


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for ASR correctness CI")


@pytest.fixture(scope="module")
def seedtts_en_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_FUN_ASR_EN_SAMPLES,
        split="en",
    )


@pytest.fixture(scope="module")
def seedtts_zh_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_FUN_ASR_ZH_SAMPLES,
        split="zh",
    )


@pytest.fixture(scope="module")
def asr_router_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> ManagedRouterHandle:
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=FUN_ASR_CI_MODEL_PATH,
        model_name=FUN_ASR_CI_MODEL_PATH,
        worker_extra_args="",
        wait_timeout=STARTUP_TIMEOUT,
        log_prefix="fun_asr_router_logs",
    ) as router:
        yield router


def _format_high_wer_sample(sample: dict) -> str:
    return "\n".join(
        [
            f"sample_id={sample['id']}",
            f"ref_text={sample['ref_text']!r}",
            f"omni={sample['hyp_text']!r}",
            f"sample_wer={sample['wer']:.4f}",
            f"ref_norm={sample['ref_norm']!r}",
            f"omni_norm={sample['hyp_norm']!r}",
        ]
    )


def _collect_high_wer_samples(results: dict, threshold: float) -> list[str]:
    return [
        _format_high_wer_sample(sample)
        for sample in results["per_sample"]
        if sample["is_success"]
        and sample["wer"] is not None
        and sample["wer"] > threshold
    ]


@pytest.mark.benchmark
def test_fun_asr_matches_seedtts_reference_text_en(
    seedtts_en_samples: list[SampleInput],
    asr_router_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector("Fun-ASR EN correctness and speed")
    checks.check(
        len(seedtts_en_samples) == SEEDTTS_FUN_ASR_EN_SAMPLES,
        f"Expected {SEEDTTS_FUN_ASR_EN_SAMPLES} SeedTTS samples, "
        f"got {len(seedtts_en_samples)}",
    )
    if not seedtts_en_samples:
        checks.assert_all()

    with router_worker_traffic_guard(
        asr_router_server,
        label="Fun-ASR SeedTTS EN",
    ) as router_guard:
        results = asyncio.run(
            run_asr_seedtts_once(
                seedtts_en_samples,
                host="127.0.0.1",
                port=asr_router_server.port,
                model_path=FUN_ASR_CI_MODEL_PATH,
                lang="en",
                concurrency=FUN_ASR_CONCURRENCY,
                warmup=FUN_ASR_WARMUP_REQUESTS,
                disable_tqdm=False,
            )
        )
    summary = results["summary"]
    speed = results["speed"]

    high_wer_samples = _collect_high_wer_samples(
        results, FUN_ASR_EN_SAMPLE_WER_THRESHOLD
    )

    print_asr_wer_summary(
        summary, FUN_ASR_CI_MODEL_PATH, dataset=SEEDTTS_ASR_DATASET_LABEL
    )
    print_asr_speed_summary(
        speed, FUN_ASR_CI_MODEL_PATH, dataset=SEEDTTS_ASR_DATASET_LABEL
    )

    results_path = tmp_path_factory.getbasetemp() / "fun_asr_results.json"
    results_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "speed": speed,
                "router_ready_s": asr_router_server.router_ready_s,
            },
            indent=2,
        )
    )

    total_samples = summary["total_samples"]
    evaluated = summary["evaluated"]
    corpus_wer = summary["corpus_wer"]
    throughput_samples_per_s = speed["throughput_samples_per_s"]
    latency_mean_s = speed["latency_mean_s"]
    latency_p95_s = speed["latency_p95_s"]
    rtf_mean = speed["rtf_mean"]
    rtf_p95 = speed["rtf_p95"]

    checks.check(
        total_samples == SEEDTTS_FUN_ASR_EN_SAMPLES,
        f"Expected {SEEDTTS_FUN_ASR_EN_SAMPLES} scored samples, "
        f"got {total_samples}",
    )
    checks.check(
        evaluated == total_samples,
        f"Fun-ASR EN transcribed only {evaluated}/{total_samples} samples, "
        "failed requests are excluded from WER scoring",
    )
    checks.check(
        corpus_wer <= FUN_ASR_EN_CORPUS_WER_THRESHOLD,
        f"Fun-ASR EN corpus WER {corpus_wer:.4f} exceeds "
        f"{FUN_ASR_EN_CORPUS_WER_THRESHOLD:.4f}",
    )
    checks.check(
        not high_wer_samples,
        "Fun-ASR high-WER SeedTTS EN samples:\n" + "\n\n".join(high_wer_samples),
    )
    checks.check(
        throughput_samples_per_s >= FUN_ASR_THROUGHPUT_THRESHOLD,
        f"Fun-ASR throughput {throughput_samples_per_s:.3f} samples/s "
        f"is below {FUN_ASR_THROUGHPUT_THRESHOLD:.3f}",
    )
    checks.check(
        latency_mean_s <= FUN_ASR_LATENCY_MEAN_THRESHOLD_S,
        f"Fun-ASR mean latency {latency_mean_s:.3f}s exceeds "
        f"{FUN_ASR_LATENCY_MEAN_THRESHOLD_S:.3f}s",
    )
    checks.check(
        latency_p95_s <= FUN_ASR_LATENCY_P95_THRESHOLD_S,
        f"Fun-ASR p95 latency {latency_p95_s:.3f}s exceeds "
        f"{FUN_ASR_LATENCY_P95_THRESHOLD_S:.3f}s",
    )
    checks.check(
        rtf_mean <= FUN_ASR_RTF_MEAN_THRESHOLD,
        f"Fun-ASR mean RTF {rtf_mean:.4f} exceeds " f"{FUN_ASR_RTF_MEAN_THRESHOLD:.4f}",
    )
    checks.check(
        rtf_p95 <= FUN_ASR_RTF_P95_THRESHOLD,
        f"Fun-ASR p95 RTF {rtf_p95:.4f} exceeds " f"{FUN_ASR_RTF_P95_THRESHOLD:.4f}",
    )
    router_guard.assert_served(
        min_total_requests=len(seedtts_en_samples),
        min_worker_share=0.40,
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_fun_asr_matches_seedtts_reference_text_zh(
    seedtts_zh_samples: list[SampleInput],
    asr_router_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector("Fun-ASR ZH correctness")
    checks.check(
        len(seedtts_zh_samples) == SEEDTTS_FUN_ASR_ZH_SAMPLES,
        f"Expected {SEEDTTS_FUN_ASR_ZH_SAMPLES} SeedTTS samples, "
        f"got {len(seedtts_zh_samples)}",
    )
    if not seedtts_zh_samples:
        checks.assert_all()

    with router_worker_traffic_guard(
        asr_router_server,
        label="Fun-ASR SeedTTS ZH",
    ) as router_guard:
        results = asyncio.run(
            run_asr_seedtts_once(
                seedtts_zh_samples,
                host="127.0.0.1",
                port=asr_router_server.port,
                model_path=FUN_ASR_CI_MODEL_PATH,
                lang="zh",
                concurrency=FUN_ASR_CONCURRENCY,
                warmup=FUN_ASR_WARMUP_REQUESTS,
                disable_tqdm=False,
            )
        )
    summary = results["summary"]
    speed = results["speed"]

    high_wer_samples = _collect_high_wer_samples(
        results, FUN_ASR_ZH_SAMPLE_WER_THRESHOLD
    )

    print_asr_wer_summary(
        summary, FUN_ASR_CI_MODEL_PATH, dataset=SEEDTTS_ASR_DATASET_LABEL
    )
    print_asr_speed_summary(
        speed, FUN_ASR_CI_MODEL_PATH, dataset=SEEDTTS_ASR_DATASET_LABEL
    )

    results_path = tmp_path_factory.getbasetemp() / "fun_asr_zh_results.json"
    results_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "speed": speed,
            },
            indent=2,
        )
    )

    total_samples = summary["total_samples"]
    evaluated = summary["evaluated"]
    corpus_wer = summary["corpus_wer"]

    checks.check(
        total_samples == SEEDTTS_FUN_ASR_ZH_SAMPLES,
        f"Expected {SEEDTTS_FUN_ASR_ZH_SAMPLES} scored samples, "
        f"got {total_samples}",
    )
    checks.check(
        evaluated == total_samples,
        f"Fun-ASR ZH transcribed only {evaluated}/{total_samples} samples, "
        "failed requests are excluded from WER scoring",
    )
    checks.check(
        corpus_wer <= FUN_ASR_ZH_CORPUS_WER_THRESHOLD,
        f"Fun-ASR ZH corpus WER {corpus_wer:.4f} exceeds "
        f"{FUN_ASR_ZH_CORPUS_WER_THRESHOLD:.4f}",
    )
    checks.check(
        not high_wer_samples,
        "Fun-ASR high-WER SeedTTS ZH samples:\n" + "\n\n".join(high_wer_samples),
    )
    router_guard.assert_served(
        min_total_requests=len(seedtts_zh_samples),
        min_worker_share=0.40,
    )
    checks.assert_all()
