# SPDX-License-Identifier: Apache-2.0
"""SeedTTS ASR correctness CI for Qwen3-ASR.

The test uses the full English SeedTTS set as the speech corpus. It compares
normalized transcriptions from the SGLang Omni Qwen3-ASR router against the
dataset reference text.

Author:
chenyang zhao: https://github.com/zhaochenyang20
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_asr_seedtts import (
    QWEN3_ASR_MODEL_PATH,
    run_asr_seedtts_once,
)
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.wer import print_asr_speed_summary, print_asr_wer_summary
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
    router_worker_traffic_guard,
)
from tests.utils import QWEN3_ASR_WER_CONCURRENCY, MetricCheckCollector, apply_wer_slack

QWEN3_ASR_CI_MODEL_PATH = QWEN3_ASR_MODEL_PATH
QWEN3_ASR_CONCURRENCY = QWEN3_ASR_WER_CONCURRENCY
QWEN3_ASR_WARMUP_REQUESTS = QWEN3_ASR_CONCURRENCY * 2
SEEDTTS_ASR_CORRECTNESS_SAMPLES = 1088
SEEDTTS_ASR_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="seedtts",
    repo_id=DATASETS["seedtts"],
)

# P95 reference values calibrated by tune.py (worst-of-N).
SEEDTTS_ASR_CORPUS_WER_MAX = 0.0122
SEEDTTS_ASR_SAMPLE_WER_MAX = 0.1819
QWEN3_ASR_THROUGHPUT_MIN = 89.15367011738545
QWEN3_ASR_LATENCY_MEAN_MAX_S = 0.357
QWEN3_ASR_LATENCY_P95_MAX_S = 0.493352619552752
QWEN3_ASR_RTF_MEAN_MAX = 0.07720349691994066
QWEN3_ASR_RTF_P95_MAX = 0.1084

THRESHOLD_SLACK_HIGHER = 0.9
THRESHOLD_SLACK_LOWER = 1.1

SEEDTTS_ASR_CORPUS_WER_THRESHOLD = apply_wer_slack(
    SEEDTTS_ASR_CORPUS_WER_MAX, THRESHOLD_SLACK_LOWER
)
SEEDTTS_ASR_SAMPLE_WER_THRESHOLD = apply_wer_slack(
    SEEDTTS_ASR_SAMPLE_WER_MAX, THRESHOLD_SLACK_LOWER
)
QWEN3_ASR_THROUGHPUT_THRESHOLD = round(
    QWEN3_ASR_THROUGHPUT_MIN * THRESHOLD_SLACK_HIGHER, 3
)
QWEN3_ASR_LATENCY_MEAN_THRESHOLD_S = round(
    QWEN3_ASR_LATENCY_MEAN_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
QWEN3_ASR_LATENCY_P95_THRESHOLD_S = round(
    QWEN3_ASR_LATENCY_P95_MAX_S * THRESHOLD_SLACK_LOWER, 3
)
QWEN3_ASR_RTF_MEAN_THRESHOLD = round(QWEN3_ASR_RTF_MEAN_MAX * THRESHOLD_SLACK_LOWER, 4)
QWEN3_ASR_RTF_P95_THRESHOLD = round(QWEN3_ASR_RTF_P95_MAX * THRESHOLD_SLACK_LOWER, 4)
STARTUP_TIMEOUT = 600


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for ASR correctness CI")


@pytest.fixture(scope="module")
def seedtts_en_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_ASR_CORRECTNESS_SAMPLES,
        split="en",
    )


@pytest.fixture(scope="module")
def asr_router_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> ManagedRouterHandle:
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=QWEN3_ASR_CI_MODEL_PATH,
        model_name=QWEN3_ASR_CI_MODEL_PATH,
        worker_extra_args="",
        wait_timeout=STARTUP_TIMEOUT,
        log_prefix="qwen3_asr_router_logs",
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


@pytest.mark.benchmark
def test_asr_matches_seedtts_reference_text(
    seedtts_en_samples: list[SampleInput],
    asr_router_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector("Qwen3-ASR correctness and speed")
    checks.check(
        len(seedtts_en_samples) == SEEDTTS_ASR_CORRECTNESS_SAMPLES,
        f"Expected {SEEDTTS_ASR_CORRECTNESS_SAMPLES} SeedTTS samples, "
        f"got {len(seedtts_en_samples)}",
    )
    if not seedtts_en_samples:
        checks.assert_all()

    with router_worker_traffic_guard(
        asr_router_server,
        label="Qwen3-ASR SeedTTS",
    ) as router_guard:
        results = asyncio.run(
            run_asr_seedtts_once(
                seedtts_en_samples,
                host="127.0.0.1",
                port=asr_router_server.port,
                model_path=QWEN3_ASR_CI_MODEL_PATH,
                lang="en",
                concurrency=QWEN3_ASR_CONCURRENCY,
                warmup=QWEN3_ASR_WARMUP_REQUESTS,
                disable_tqdm=False,
            )
        )
    summary = results["summary"]
    speed = results["speed"]

    high_wer_samples = [
        _format_high_wer_sample(sample)
        for sample in results["per_sample"]
        if sample["is_success"]
        and sample["wer"] is not None
        and sample["wer"] > SEEDTTS_ASR_SAMPLE_WER_THRESHOLD
    ]

    print_asr_wer_summary(
        summary, QWEN3_ASR_CI_MODEL_PATH, dataset=SEEDTTS_ASR_DATASET_LABEL
    )
    print_asr_speed_summary(
        speed, QWEN3_ASR_CI_MODEL_PATH, dataset=SEEDTTS_ASR_DATASET_LABEL
    )

    results_path = tmp_path_factory.getbasetemp() / "qwen3_asr_results.json"
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

    corpus_wer = summary["corpus_wer"]
    throughput_samples_per_s = speed["throughput_samples_per_s"]
    latency_mean_s = speed["latency_mean_s"]
    latency_p95_s = speed["latency_p95_s"]
    rtf_mean = speed["rtf_mean"]
    rtf_p95 = speed["rtf_p95"]

    checks.check(
        corpus_wer <= SEEDTTS_ASR_CORPUS_WER_THRESHOLD,
        f"Qwen3-ASR corpus WER {corpus_wer:.4f} exceeds "
        f"{SEEDTTS_ASR_CORPUS_WER_THRESHOLD:.4f}",
    )
    checks.check(
        not high_wer_samples,
        "Qwen3-ASR high-WER SeedTTS samples:\n" + "\n\n".join(high_wer_samples),
    )
    checks.check(
        throughput_samples_per_s >= QWEN3_ASR_THROUGHPUT_THRESHOLD,
        f"Qwen3-ASR throughput {throughput_samples_per_s:.3f} samples/s "
        f"is below {QWEN3_ASR_THROUGHPUT_THRESHOLD:.3f}",
    )
    checks.check(
        latency_mean_s <= QWEN3_ASR_LATENCY_MEAN_THRESHOLD_S,
        f"Qwen3-ASR mean latency {latency_mean_s:.3f}s exceeds "
        f"{QWEN3_ASR_LATENCY_MEAN_THRESHOLD_S:.3f}s",
    )
    checks.check(
        latency_p95_s <= QWEN3_ASR_LATENCY_P95_THRESHOLD_S,
        f"Qwen3-ASR p95 latency {latency_p95_s:.3f}s exceeds "
        f"{QWEN3_ASR_LATENCY_P95_THRESHOLD_S:.3f}s",
    )
    checks.check(
        rtf_mean <= QWEN3_ASR_RTF_MEAN_THRESHOLD,
        f"Qwen3-ASR mean RTF {rtf_mean:.4f} exceeds "
        f"{QWEN3_ASR_RTF_MEAN_THRESHOLD:.4f}",
    )
    checks.check(
        rtf_p95 <= QWEN3_ASR_RTF_P95_THRESHOLD,
        f"Qwen3-ASR p95 RTF {rtf_p95:.4f} exceeds "
        f"{QWEN3_ASR_RTF_P95_THRESHOLD:.4f}",
    )
    router_guard.assert_served(
        min_total_requests=len(seedtts_en_samples),
        min_worker_share=0.40,
    )
    checks.assert_all()
