# SPDX-License-Identifier: Apache-2.0
"""Whisper ASR correctness CI for SGLang Omni.

The test uses the first 20 English SeedTTS samples as a lightweight speech
corpus. It compares normalized transcriptions from the SGLang Omni Whisper
server against the dataset reference text.
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
from benchmarks.tasks.tts import normalize_text
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, disable_proxy

WHISPER_MODEL_PATH = "openai/whisper-large-v3"
WHISPER_ASR_WORKER_ARGS = "--stages.0.factory-args.max-running-requests 1"
WHISPER_ASR_CONCURRENCY = 2
SEEDTTS_ASR_CORRECTNESS_SAMPLES = 20
SEEDTTS_ASR_CORPUS_WER_MAX = 0.007
SEEDTTS_ASR_SAMPLE_WER_MAX = 0.0667
# H20 calibration on 2026-05-30, CUDA graph bs=1, concurrency=2 (DP=2 router):
# worst-of-5: throughput=10.153 samples/s, latency_mean=0.196s,
# latency_p95=0.570s, rtf_mean=0.0415, rtf_p95=0.1459.
WHISPER_ASR_THROUGHPUT_MIN = 10.153
WHISPER_ASR_LATENCY_MEAN_MAX_S = 0.196
WHISPER_ASR_LATENCY_P95_MAX_S = 0.570
WHISPER_ASR_RTF_MEAN_MAX = 0.0415
WHISPER_ASR_RTF_P95_MAX = 0.1459
STARTUP_TIMEOUT = 600
REQUEST_TIMEOUT = 300


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Whisper ASR correctness CI")


@pytest.fixture(scope="module")
def seedtts_en_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_ASR_CORRECTNESS_SAMPLES,
        split="en",
    )


@pytest.fixture(scope="module")
def whisper_asr_router_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> ManagedRouterHandle:
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=WHISPER_MODEL_PATH,
        model_name=WHISPER_MODEL_PATH,
        worker_extra_args=WHISPER_ASR_WORKER_ARGS,
        wait_timeout=STARTUP_TIMEOUT,
        log_prefix="whisper_asr_router_logs",
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
                "model": WHISPER_MODEL_PATH,
                "language": "en",
                "temperature": "0",
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


@pytest.mark.benchmark
def test_whisper_asr_matches_seedtts_reference_text(
    seedtts_en_samples: list[SampleInput],
    whisper_asr_router_server: ManagedRouterHandle,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector("Whisper ASR correctness and speed")
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
        whisper_asr_router_server,
        label="Whisper ASR SeedTTS",
    ) as router_guard:
        wall_start_s = time.perf_counter()

        def _transcribe_sample(
            sample: SampleInput,
        ) -> tuple[str, str, float, float]:
            text, latency_s, audio_duration_s = _transcribe_with_omni(
                whisper_asr_router_server.port,
                sample,
            )
            return sample.sample_id, text, latency_s, audio_duration_s

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=WHISPER_ASR_CONCURRENCY,
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
    for sample in seedtts_en_samples:
        omni_text = omni_outputs[sample.sample_id]
        ref_norm = normalize_text(sample.ref_text, "en")
        omni_norm = normalize_text(omni_text, "en")
        ref_norms.append(ref_norm)
        hyp_norms.append(omni_norm)
        sample_wer = process_words(ref_norm, omni_norm).wer
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
            if sample_wer > SEEDTTS_ASR_SAMPLE_WER_MAX:
                high_wer_samples.append(diff)

    if sample_diffs:
        print("\n[Whisper ASR correctness diffs]\n" + "\n\n".join(sample_diffs))

    corpus_wer = process_words(ref_norms, hyp_norms).wer
    latency_mean_s = statistics.mean(latencies_s)
    latency_p95_s = _percentile(latencies_s, 95)
    throughput_samples_per_s = len(latencies_s) / wall_clock_s
    rtfs = [
        latency_s / audio_duration_s
        for latency_s, audio_duration_s in zip(latencies_s, audio_durations_s)
        if audio_duration_s > 0
    ]
    rtf_mean = statistics.mean(rtfs)
    rtf_p95 = _percentile(rtfs, 95)
    print(
        "\n[Whisper ASR correctness] "
        f"samples={len(seedtts_en_samples)} "
        f"diff_samples={len(sample_diffs)} "
        f"corpus_wer={corpus_wer:.4f}"
    )
    print(
        "\n[Whisper ASR speed] "
        f"wall_clock_s={wall_clock_s:.3f} "
        f"concurrency={WHISPER_ASR_CONCURRENCY} "
        f"throughput_samples_per_s={throughput_samples_per_s:.3f} "
        f"latency_mean_s={latency_mean_s:.3f} "
        f"latency_p95_s={latency_p95_s:.3f} "
        f"rtf_mean={rtf_mean:.4f} "
        f"rtf_p95={rtf_p95:.4f}"
    )
    per_sample_wer_max = max(
        (
            process_words(
                normalize_text(sample.ref_text, "en"),
                normalize_text(omni_outputs[sample.sample_id], "en"),
            ).wer
            for sample in seedtts_en_samples
        ),
        default=0.0,
    )
    results_path = tmp_path_factory.getbasetemp() / "whisper_asr_results.json"
    results_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_samples": len(seedtts_en_samples),
                    "evaluated": len(seedtts_en_samples),
                    "corpus_wer": corpus_wer,
                    "wer_per_sample_max": per_sample_wer_max,
                },
                "speed": {
                    "throughput_samples_per_s": throughput_samples_per_s,
                    "latency_mean_s": latency_mean_s,
                    "latency_p95_s": latency_p95_s,
                    "rtf_mean": rtf_mean,
                    "rtf_p95": rtf_p95,
                },
            },
            indent=2,
        )
    )
    checks.check(
        corpus_wer <= SEEDTTS_ASR_CORPUS_WER_MAX,
        f"Whisper ASR corpus WER {corpus_wer:.4f} exceeds "
        f"{SEEDTTS_ASR_CORPUS_WER_MAX:.4f}",
    )
    checks.check(
        not high_wer_samples,
        "Whisper ASR high-WER SeedTTS samples:\n" + "\n\n".join(high_wer_samples),
    )
    checks.check(
        throughput_samples_per_s >= WHISPER_ASR_THROUGHPUT_MIN,
        f"Whisper ASR throughput {throughput_samples_per_s:.3f} samples/s "
        f"is below {WHISPER_ASR_THROUGHPUT_MIN:.3f}",
    )
    checks.check(
        latency_mean_s <= WHISPER_ASR_LATENCY_MEAN_MAX_S,
        f"Whisper ASR mean latency {latency_mean_s:.3f}s exceeds "
        f"{WHISPER_ASR_LATENCY_MEAN_MAX_S:.3f}s",
    )
    checks.check(
        latency_p95_s <= WHISPER_ASR_LATENCY_P95_MAX_S,
        f"Whisper ASR p95 latency {latency_p95_s:.3f}s exceeds "
        f"{WHISPER_ASR_LATENCY_P95_MAX_S:.3f}s",
    )
    checks.check(
        rtf_mean <= WHISPER_ASR_RTF_MEAN_MAX,
        f"Whisper ASR mean RTF {rtf_mean:.4f} exceeds "
        f"{WHISPER_ASR_RTF_MEAN_MAX:.4f}",
    )
    checks.check(
        rtf_p95 <= WHISPER_ASR_RTF_P95_MAX,
        f"Whisper ASR p95 RTF {rtf_p95:.4f} exceeds " f"{WHISPER_ASR_RTF_P95_MAX:.4f}",
    )
    router_guard.assert_served(
        min_total_requests=len(seedtts_en_samples),
        min_worker_share=0.40,
    )
    checks.assert_all()
