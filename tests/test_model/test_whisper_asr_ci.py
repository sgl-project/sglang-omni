# SPDX-License-Identifier: Apache-2.0
"""Whisper ASR correctness CI for SGLang Omni.

The test uses the first 20 English SeedTTS samples as a lightweight speech
corpus. It compares normalized transcriptions from the SGLang Omni Whisper
server against the dataset reference text.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import pytest
import requests
from jiwer import process_words

from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.tasks.tts import normalize_text
from tests.utils import (
    MetricCheckCollector,
    disable_proxy,
    no_proxy_env,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

WHISPER_MODEL_PATH = "openai/whisper-large-v3"
SEEDTTS_ASR_CORRECTNESS_SAMPLES = 20
SEEDTTS_ASR_CORPUS_WER_MAX = 0.01
SEEDTTS_ASR_SAMPLE_WER_MAX = 0.20
# H100 calibration on 2026-05-30 with CUDA graph bs=1:
# throughput=8.532 samples/s, latency_mean=0.117s, latency_p95=0.155s,
# rtf_mean=0.0243, rtf_p95=0.0300. Thresholds include initial CI jitter slack;
# retune these on H20 before treating them as final CI gates.
WHISPER_ASR_THROUGHPUT_MIN = 6.0
WHISPER_ASR_LATENCY_MEAN_MAX_S = 0.20
WHISPER_ASR_LATENCY_P95_MAX_S = 0.50
WHISPER_ASR_RTF_MEAN_MAX = 0.04
WHISPER_ASR_RTF_P95_MAX = 0.12
STARTUP_TIMEOUT = 600
REQUEST_TIMEOUT = 300
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _start_whisper_server(tmp_path_factory: pytest.TempPathFactory):
    from sglang_omni.utils import find_available_port

    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, prefix="whisper_asr_logs")
    env = no_proxy_env()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        WHISPER_MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
        "--stages.0.factory-args.max-running-requests",
        "1",
    ]
    proc = start_server_from_cmd(
        cmd,
        log_file,
        port,
        timeout=STARTUP_TIMEOUT,
        env=env,
    )
    return proc, port


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

    proc, port = _start_whisper_server(tmp_path_factory)
    omni_outputs: dict[str, str] = {}
    latencies_s: list[float] = []
    audio_durations_s: list[float] = []
    try:
        for sample in seedtts_en_samples:
            text, latency_s, audio_duration_s = _transcribe_with_omni(port, sample)
            omni_outputs[sample.sample_id] = text
            latencies_s.append(latency_s)
            audio_durations_s.append(audio_duration_s)
    finally:
        stop_server(proc)

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
    total_latency_s = sum(latencies_s)
    latency_mean_s = statistics.mean(latencies_s)
    latency_p95_s = _percentile(latencies_s, 95)
    throughput_samples_per_s = len(latencies_s) / total_latency_s
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
        f"total_latency_s={total_latency_s:.3f} "
        f"throughput_samples_per_s={throughput_samples_per_s:.3f} "
        f"latency_mean_s={latency_mean_s:.3f} "
        f"latency_p95_s={latency_p95_s:.3f} "
        f"rtf_mean={rtf_mean:.4f} "
        f"rtf_p95={rtf_p95:.4f}"
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
    checks.assert_all()
