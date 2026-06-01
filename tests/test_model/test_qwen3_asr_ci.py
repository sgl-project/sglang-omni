# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR correctness CI for SGLang Omni.

The test uses the first 20 English SeedTTS samples as a lightweight speech
corpus. It compares normalized transcriptions from the SGLang Omni Qwen3-ASR
router against the dataset reference text.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import time

import pytest
import requests
from jiwer import process_words

from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.wer import print_asr_speed_summary, print_asr_wer_summary
from benchmarks.tasks.tts import (
    QWEN3_ASR_MAX_NEW_TOKENS,
    QWEN3_ASR_MODEL_PATH,
    normalize_text,
)
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, apply_wer_slack, disable_proxy

QWEN3_ASR_CI_MODEL_PATH = QWEN3_ASR_MODEL_PATH
QWEN3_ASR_CONCURRENCY = int(os.getenv("QWEN3_ASR_CI_CONCURRENCY", "32"))
QWEN3_ASR_WARMUP_REQUESTS = QWEN3_ASR_CONCURRENCY * 2
SEEDTTS_ASR_CORRECTNESS_SAMPLES = 20

# P95 reference values calibrated by tune.py (worst-of-N).
SEEDTTS_ASR_CORPUS_WER_MAX = 0.007
SEEDTTS_ASR_SAMPLE_WER_MAX = 0.0667
QWEN3_ASR_THROUGHPUT_MIN = 10.294983887949183
QWEN3_ASR_LATENCY_MEAN_MAX_S = 0.19333569328882733
QWEN3_ASR_LATENCY_P95_MAX_S = 0.555
QWEN3_ASR_RTF_MEAN_MAX = 0.0409
QWEN3_ASR_RTF_P95_MAX = 0.1421

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
REQUEST_TIMEOUT = 300


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Qwen3-ASR correctness CI")


@pytest.fixture(scope="module")
def seedtts_en_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_ASR_CORRECTNESS_SAMPLES,
        split="en",
    )


@pytest.fixture(scope="module")
def qwen3_asr_router_server(
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


def _transcribe_with_omni(port: int, sample: SampleInput) -> tuple[str, float, float]:
    with open(sample.ref_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    start = time.perf_counter()
    with disable_proxy():
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/audio/transcriptions",
            data={
                "model": QWEN3_ASR_CI_MODEL_PATH,
                "language": "en",
                "response_format": "json",
                "max_new_tokens": str(QWEN3_ASR_MAX_NEW_TOKENS),
            },
            files={
                "file": (
                    os.path.basename(sample.ref_audio),
                    audio_bytes,
                    "audio/wav",
                )
            },
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return (
        str(response.json()["text"]),
        time.perf_counter() - start,
        get_wav_duration(audio_bytes),
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _warm_up_asr_router(port: int, samples: list[SampleInput]) -> None:
    warmup_samples = samples[:QWEN3_ASR_WARMUP_REQUESTS]
    if not warmup_samples:
        return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=QWEN3_ASR_CONCURRENCY,
    ) as executor:
        futures = [
            executor.submit(_transcribe_with_omni, port, sample)
            for sample in warmup_samples
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print(
        "\n[Qwen3-ASR warmup] "
        f"requests={len(warmup_samples)} "
        f"concurrency={QWEN3_ASR_CONCURRENCY}"
    )


@pytest.mark.benchmark
def test_qwen3_asr_matches_seedtts_reference_text(
    seedtts_en_samples: list[SampleInput],
    qwen3_asr_router_server: ManagedRouterHandle,
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

    omni_outputs: dict[str, str] = {}
    latencies_s: list[float] = []
    audio_durations_s: list[float] = []
    with router_worker_traffic_guard(
        qwen3_asr_router_server,
        label="Qwen3-ASR SeedTTS",
    ) as router_guard:
        _warm_up_asr_router(qwen3_asr_router_server.port, seedtts_en_samples)
        wall_start_s = time.perf_counter()

        def _transcribe_sample(
            sample: SampleInput,
        ) -> tuple[str, str, float, float]:
            text, latency_s, audio_duration_s = _transcribe_with_omni(
                qwen3_asr_router_server.port,
                sample,
            )
            return sample.sample_id, text, latency_s, audio_duration_s

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=QWEN3_ASR_CONCURRENCY,
        ) as executor:
            futures = [
                executor.submit(_transcribe_sample, sample)
                for sample in seedtts_en_samples
            ]
            for future in concurrent.futures.as_completed(futures):
                sample_id, text, latency_s, audio_duration_s = future.result()
                omni_outputs[sample_id] = text
                latencies_s.append(latency_s)
                audio_durations_s.append(audio_duration_s)

        wall_clock_s = time.perf_counter() - wall_start_s

    sample_diffs: list[str] = []
    high_wer_samples: list[str] = []
    ref_norms: list[str] = []
    hyp_norms: list[str] = []
    sample_wers: list[float] = []
    for sample in seedtts_en_samples:
        omni_text = omni_outputs[sample.sample_id]
        ref_norm = normalize_text(sample.ref_text, "en")
        omni_norm = normalize_text(omni_text, "en")
        ref_norms.append(ref_norm)
        hyp_norms.append(omni_norm)
        sample_wer = process_words(ref_norm, omni_norm).wer
        sample_wers.append(sample_wer)
        if sample_wer > 0:
            diff = "\n".join(
                [
                    f"sample_id={sample.sample_id}",
                    f"ref_text={sample.ref_text!r}",
                    f"omni={omni_text!r}",
                    f"sample_wer={sample_wer:.4f}",
                    f"ref_norm={ref_norm!r}",
                    f"omni_norm={omni_norm!r}",
                ]
            )
            sample_diffs.append(diff)
            if sample_wer > SEEDTTS_ASR_SAMPLE_WER_THRESHOLD:
                high_wer_samples.append(diff)

    corpus_wer = process_words(ref_norms, hyp_norms).wer
    latency_mean_s = statistics.mean(latencies_s)
    latency_median_s = statistics.median(latencies_s)
    latency_p95_s = _percentile(latencies_s, 95)
    latency_p99_s = _percentile(latencies_s, 99)
    throughput_samples_per_s = len(latencies_s) / wall_clock_s
    rtfs = [
        latency_s / audio_duration_s
        for latency_s, audio_duration_s in zip(latencies_s, audio_durations_s)
        if audio_duration_s > 0
    ]
    rtf_mean = statistics.mean(rtfs)
    rtf_median = statistics.median(rtfs)
    rtf_p95 = _percentile(rtfs, 95)
    per_sample_wer_max = max(sample_wers, default=0.0)
    below_50_pairs = [
        (ref_norm, hyp_norm)
        for ref_norm, hyp_norm, sample_wer in zip(ref_norms, hyp_norms, sample_wers)
        if sample_wer <= 0.5
    ]
    wer_below_50_corpus = (
        process_words(
            [ref_norm for ref_norm, _ in below_50_pairs],
            [hyp_norm for _, hyp_norm in below_50_pairs],
        ).wer
        if below_50_pairs
        else 0.0
    )
    n_above_50 = sum(1 for sample_wer in sample_wers if sample_wer > 0.5)
    wer_summary = {
        "lang": "en",
        "total_samples": len(seedtts_en_samples),
        "evaluated": len(seedtts_en_samples),
        "skipped": 0,
        "corpus_wer": corpus_wer,
        "wer_corpus": corpus_wer,
        "wer_per_sample_mean": statistics.mean(sample_wers) if sample_wers else 0.0,
        "wer_per_sample_median": statistics.median(sample_wers) if sample_wers else 0.0,
        "wer_per_sample_std": (
            statistics.pstdev(sample_wers) if len(sample_wers) > 1 else 0.0
        ),
        "wer_per_sample_p95": _percentile(sample_wers, 95),
        "wer_per_sample_max": per_sample_wer_max,
        "wer_below_50_corpus": wer_below_50_corpus,
        "n_above_50_pct_wer": n_above_50,
        "pct_above_50_pct_wer": (
            n_above_50 / len(sample_wers) * 100 if sample_wers else 0.0
        ),
        "latency_mean_s": latency_mean_s,
        "latency_median_s": latency_median_s,
        "latency_p95_s": latency_p95_s,
        "rtf_mean": rtf_mean,
        "audio_duration_mean_s": (
            statistics.mean(audio_durations_s) if audio_durations_s else 0.0
        ),
    }
    speed_metrics = {
        "total_samples": len(seedtts_en_samples),
        "evaluated": len(seedtts_en_samples),
        "skipped": 0,
        "asr_model": QWEN3_ASR_CI_MODEL_PATH,
        "asr_concurrency": QWEN3_ASR_CONCURRENCY,
        "asr_latency_mean_s": latency_mean_s,
        "asr_latency_median_s": latency_median_s,
        "asr_latency_p95_s": latency_p95_s,
        "asr_latency_p99_s": latency_p99_s,
        "asr_total_time_s": wall_clock_s,
        "asr_latency_sum_s": sum(latencies_s),
        "asr_throughput_samples_per_s": throughput_samples_per_s,
        "asr_rtf_mean": rtf_mean,
        "asr_rtf_median": rtf_median,
        "asr_rtf_p95": rtf_p95,
        "asr_audio_processed_s": sum(audio_durations_s),
        # Backward-compatible calibration paths.
        "throughput_samples_per_s": throughput_samples_per_s,
        "latency_mean_s": latency_mean_s,
        "latency_p95_s": latency_p95_s,
        "rtf_mean": rtf_mean,
        "rtf_p95": rtf_p95,
    }

    print_asr_wer_summary(wer_summary, QWEN3_ASR_CI_MODEL_PATH)
    if sample_diffs:
        print("\n[ASR WER diagnostic diffs]\n" + "\n\n".join(sample_diffs))
    print_asr_speed_summary(speed_metrics, QWEN3_ASR_CI_MODEL_PATH)

    results = {"summary": wer_summary, "speed": speed_metrics}
    results_path = tmp_path_factory.getbasetemp() / "qwen3_asr_results.json"
    results_path.write_text(json.dumps(results, indent=2))
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
