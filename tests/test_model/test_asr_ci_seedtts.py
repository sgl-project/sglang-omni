# SPDX-License-Identifier: Apache-2.0
"""SeedTTS ASR correctness CI for the selected ASR CI model preset.

The test reuses the shared ASR SeedTTS harness against the SGLang Omni ASR
router for the model selected via ASR_CI_MODEL (see
tests/test_model/asr_ci_config.py). The English split gates WER and speed at
the calibrated concurrency. The Chinese split gates WER as well, since the
CI ASR models target multilingual and CJK transcription and a gate on
English alone would miss regressions on a primary use case. Speed is gated
on the English split only, because a second speed gate on the same topology
would add flake surface without new signal.

Author:
Aaron Tian: https://github.com/db-ol
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_asr_seedtts import run_asr_seedtts_once
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.wer import print_asr_speed_summary, print_asr_wer_summary
from benchmarks.tasks.asr import DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
from tests.test_model.asr_ci_config import select_asr_ci_preset
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector

_MODEL_NAME, _PRESET = select_asr_ci_preset()
_THRESHOLDS = _PRESET.thresholds

ASR_CI_CONCURRENCY = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
ASR_CI_WARMUP_REQUESTS = ASR_CI_CONCURRENCY * 2
SEEDTTS_ASR_EN_SAMPLES = 1088
SEEDTTS_ASR_ZH_SAMPLES = 2020
SEEDTTS_ASR_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="seedtts",
    repo_id=DATASETS["seedtts"],
)
EN_RESULTS_BASENAME = "asr_seedtts_en_results.json"
ZH_RESULTS_BASENAME = "asr_seedtts_zh_results.json"
STARTUP_TIMEOUT = 600


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for ASR correctness CI")


@pytest.fixture(scope="module")
def seedtts_en_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_ASR_EN_SAMPLES,
        split="en",
    )


@pytest.fixture(scope="module")
def seedtts_zh_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_ASR_ZH_SAMPLES,
        split="zh",
    )


@pytest.fixture(scope="module")
def asr_router_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> ManagedRouterHandle:
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=_PRESET.resolved_model_path(),
        model_name=_PRESET.model_path,
        worker_extra_args="",
        wait_timeout=STARTUP_TIMEOUT,
        log_prefix="asr_seedtts_router_logs",
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


def _print_report_only_notice(split: str) -> None:
    print(
        f"[ASR CI] {_PRESET.display_name} {split} report-only: "
        "threshold assertions skipped pending calibration"
    )


@pytest.mark.benchmark
def test_asr_matches_seedtts_reference_text_en(
    seedtts_en_samples: list[SampleInput],
    asr_router_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector(f"{_PRESET.display_name} EN correctness and speed")
    checks.check(
        len(seedtts_en_samples) == SEEDTTS_ASR_EN_SAMPLES,
        f"Expected {SEEDTTS_ASR_EN_SAMPLES} SeedTTS samples, "
        f"got {len(seedtts_en_samples)}",
    )
    if not seedtts_en_samples:
        checks.assert_all()

    with router_worker_traffic_guard(
        asr_router_server,
        label=f"{_PRESET.display_name} SeedTTS EN",
    ) as router_guard:
        results = asyncio.run(
            run_asr_seedtts_once(
                seedtts_en_samples,
                host="127.0.0.1",
                port=asr_router_server.port,
                model_path=_PRESET.model_path,
                lang="en",
                concurrency=ASR_CI_CONCURRENCY,
                warmup=ASR_CI_WARMUP_REQUESTS,
                disable_tqdm=False,
            )
        )
    summary = results["summary"]
    speed = results["speed"]

    high_wer_samples = _collect_high_wer_samples(results, _THRESHOLDS.en_sample_wer_max)

    print_asr_wer_summary(
        summary, _PRESET.model_path, dataset=SEEDTTS_ASR_DATASET_LABEL
    )
    print_asr_speed_summary(
        speed, _PRESET.model_path, dataset=SEEDTTS_ASR_DATASET_LABEL
    )

    results_path = tmp_path_factory.getbasetemp() / EN_RESULTS_BASENAME
    results_path.write_text(
        json.dumps(
            {
                "model_path": _PRESET.model_path,
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
        total_samples == SEEDTTS_ASR_EN_SAMPLES,
        f"Expected {SEEDTTS_ASR_EN_SAMPLES} scored samples, " f"got {total_samples}",
    )
    checks.check(
        evaluated == total_samples,
        f"{_PRESET.display_name} EN transcribed only {evaluated}/{total_samples} "
        "samples, failed requests are excluded from WER scoring",
    )
    if _PRESET.gate_thresholds:
        checks.check(
            corpus_wer <= _THRESHOLDS.en_corpus_wer_max,
            f"{_PRESET.display_name} EN corpus WER {corpus_wer:.4f} exceeds "
            f"{_THRESHOLDS.en_corpus_wer_max:.4f}",
        )
        checks.check(
            not high_wer_samples,
            f"{_PRESET.display_name} high-WER SeedTTS EN samples:\n"
            + "\n\n".join(high_wer_samples),
        )
        checks.check(
            throughput_samples_per_s >= _THRESHOLDS.throughput_min,
            f"{_PRESET.display_name} throughput {throughput_samples_per_s:.3f} "
            f"samples/s is below {_THRESHOLDS.throughput_min:.3f}",
        )
        checks.check(
            latency_mean_s <= _THRESHOLDS.latency_mean_max_s,
            f"{_PRESET.display_name} mean latency {latency_mean_s:.3f}s exceeds "
            f"{_THRESHOLDS.latency_mean_max_s:.3f}s",
        )
        checks.check(
            latency_p95_s <= _THRESHOLDS.latency_p95_max_s,
            f"{_PRESET.display_name} p95 latency {latency_p95_s:.3f}s exceeds "
            f"{_THRESHOLDS.latency_p95_max_s:.3f}s",
        )
        checks.check(
            rtf_mean <= _THRESHOLDS.rtf_mean_max,
            f"{_PRESET.display_name} mean RTF {rtf_mean:.4f} exceeds "
            f"{_THRESHOLDS.rtf_mean_max:.4f}",
        )
        checks.check(
            rtf_p95 <= _THRESHOLDS.rtf_p95_max,
            f"{_PRESET.display_name} p95 RTF {rtf_p95:.4f} exceeds "
            f"{_THRESHOLDS.rtf_p95_max:.4f}",
        )
    else:
        _print_report_only_notice("EN")
    router_guard.assert_served(
        min_total_requests=len(seedtts_en_samples),
        min_worker_share=0.40,
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_asr_matches_seedtts_reference_text_zh(
    seedtts_zh_samples: list[SampleInput],
    asr_router_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector(f"{_PRESET.display_name} ZH correctness")
    checks.check(
        len(seedtts_zh_samples) == SEEDTTS_ASR_ZH_SAMPLES,
        f"Expected {SEEDTTS_ASR_ZH_SAMPLES} SeedTTS samples, "
        f"got {len(seedtts_zh_samples)}",
    )
    if not seedtts_zh_samples:
        checks.assert_all()

    with router_worker_traffic_guard(
        asr_router_server,
        label=f"{_PRESET.display_name} SeedTTS ZH",
    ) as router_guard:
        results = asyncio.run(
            run_asr_seedtts_once(
                seedtts_zh_samples,
                host="127.0.0.1",
                port=asr_router_server.port,
                model_path=_PRESET.model_path,
                lang="zh",
                concurrency=ASR_CI_CONCURRENCY,
                warmup=ASR_CI_WARMUP_REQUESTS,
                disable_tqdm=False,
            )
        )
    summary = results["summary"]
    speed = results["speed"]

    high_wer_samples = _collect_high_wer_samples(results, _THRESHOLDS.zh_sample_wer_max)

    print_asr_wer_summary(
        summary, _PRESET.model_path, dataset=SEEDTTS_ASR_DATASET_LABEL
    )
    print_asr_speed_summary(
        speed, _PRESET.model_path, dataset=SEEDTTS_ASR_DATASET_LABEL
    )

    results_path = tmp_path_factory.getbasetemp() / ZH_RESULTS_BASENAME
    results_path.write_text(
        json.dumps(
            {
                "model_path": _PRESET.model_path,
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
        total_samples == SEEDTTS_ASR_ZH_SAMPLES,
        f"Expected {SEEDTTS_ASR_ZH_SAMPLES} scored samples, " f"got {total_samples}",
    )
    checks.check(
        evaluated == total_samples,
        f"{_PRESET.display_name} ZH transcribed only {evaluated}/{total_samples} "
        "samples, failed requests are excluded from WER scoring",
    )
    if _PRESET.gate_thresholds:
        checks.check(
            corpus_wer <= _THRESHOLDS.zh_corpus_wer_max,
            f"{_PRESET.display_name} ZH corpus WER {corpus_wer:.4f} exceeds "
            f"{_THRESHOLDS.zh_corpus_wer_max:.4f}",
        )
        checks.check(
            not high_wer_samples,
            f"{_PRESET.display_name} high-WER SeedTTS ZH samples:\n"
            + "\n\n".join(high_wer_samples),
        )
    else:
        _print_report_only_notice("ZH")
    router_guard.assert_served(
        min_total_requests=len(seedtts_zh_samples),
        min_worker_share=0.40,
    )
    checks.assert_all()
