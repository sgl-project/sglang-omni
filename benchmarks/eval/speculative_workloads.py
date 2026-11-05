# SPDX-License-Identifier: Apache-2.0
"""The workloads a speculative sweep can run, and how each client's results become load measurements."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from benchmarks.eval.benchmark_asr_transcribe_diarize import (
    RESULTS_FILE,
    SPEED_RESULTS_FILE,
)
from benchmarks.metrics.speculative import AccuracyMetric, LoadMeasurement, mean_or_none

HOST = "127.0.0.1"
TRANSCRIBE_DIARIZE_CLIENT = "benchmarks.eval.benchmark_asr_transcribe_diarize"


@dataclass(frozen=True, kw_only=True)
class AsrWorkload:
    """A client built on the benchmark_asr_seedtts sweep, which runs every concurrency and repeat in one call."""

    client: str
    client_arguments: tuple[str, ...]
    error_rate: Literal["WER", "CER"]


@dataclass(frozen=True, kw_only=True)
class TranscribeDiarizeWorkload:
    """A benchmark_asr_transcribe_diarize dataset; that client runs one concurrency per call against an existing server."""

    dataset: str


@dataclass(frozen=True, kw_only=True)
class TtsWorkload:
    """SeedTTS synthesis followed by transcription after the TTS server stops."""

    language: Literal["en", "zh"]
    error_rate: Literal["WER", "CER"]


WORKLOADS: dict[str, AsrWorkload | TranscribeDiarizeWorkload | TtsWorkload] = {
    "tts-seedtts-en": TtsWorkload(language="en", error_rate="WER"),
    "tts-seedtts-zh": TtsWorkload(language="zh", error_rate="CER"),
    "seedtts-en": AsrWorkload(
        client="benchmarks.eval.benchmark_asr_seedtts",
        client_arguments=("--lang", "en"),
        error_rate="WER",
    ),
    "seedtts-zh": AsrWorkload(
        client="benchmarks.eval.benchmark_asr_seedtts",
        client_arguments=("--lang", "zh"),
        error_rate="CER",
    ),
    "longlibriheavy-30": AsrWorkload(
        client="benchmarks.eval.benchmark_asr_longform",
        client_arguments=("--dataset", "longlibriheavy-30"),
        error_rate="WER",
    ),
    "longlibriheavy-60": AsrWorkload(
        client="benchmarks.eval.benchmark_asr_longform",
        client_arguments=("--dataset", "longlibriheavy-60"),
        error_rate="WER",
    ),
    "meanwhile": AsrWorkload(
        client="benchmarks.eval.benchmark_asr_longform",
        client_arguments=("--dataset", "meanwhile"),
        error_rate="WER",
    ),
    "pipecat-stt": AsrWorkload(
        client="benchmarks.eval.benchmark_asr_stt_benchmark",
        client_arguments=(),
        error_rate="WER",
    ),
    "movies800times": TranscribeDiarizeWorkload(dataset="movies800times"),
    "aishell4-long": TranscribeDiarizeWorkload(dataset="aishell4_long"),
    "googletime": TranscribeDiarizeWorkload(dataset="googletime"),
}


@dataclass(frozen=True, kw_only=True)
class LoadSettings:
    """What every leg runs, whatever the workload."""

    concurrencies: list[int]
    repeats: int
    warmup: bool
    max_samples: int | None
    client_arguments: list[str]


class MetricMean(BaseModel):
    mean: float | None


class MetricPeak(BaseModel):
    max: float


class AsrClientResources(BaseModel):
    gpu_memory_used_peak_mib: MetricPeak | None = None


class AsrClientResult(BaseModel):
    """One concurrency as the benchmark_asr_seedtts sweep writes it."""

    concurrency: int
    total: int
    failed: int
    throughput_samples_per_s: MetricMean
    latency_p95_s: MetricMean
    text_ttft_p95_s: MetricMean | None = None
    corpus_wer: MetricMean
    resources: AsrClientResources


class AsrClientPayload(BaseModel):
    results: list[AsrClientResult]


class TranscribeDiarizeSpeed(BaseModel):
    # note (wilsonzheng0327): the client writes no latency when every request failed.
    total_requests: int
    completed_requests: int
    failed_requests: int
    latency_p95_s: float | None = None
    text_ttft_p95_s: float | None = None


class TranscribeDiarizeRunConfig(BaseModel):
    wall_clock_s: float


class TranscribeDiarizeSpeedFile(BaseModel):
    config: TranscribeDiarizeRunConfig
    speed: TranscribeDiarizeSpeed


class DiarizationMetrics(BaseModel):
    cer: float | None = None
    speaker_timestamp_der: float | None = None


class TranscribeDiarizeResultsFile(BaseModel):
    diarization_metrics: DiarizationMetrics


def run_client(command: list[str], log_path: Path, *, check: bool) -> None:
    with open(log_path, "w") as client_log:
        subprocess.run(
            [sys.executable, "-m", *command],
            stdout=client_log,
            stderr=subprocess.STDOUT,
            check=check,
        )


def measure_asr(
    workload: AsrWorkload,
    settings: LoadSettings,
    *,
    port: int,
    model_path: str,
    leg_dir: Path,
) -> list[LoadMeasurement]:
    client_output = leg_dir / "client.json"
    run_client(
        [
            workload.client,
            "--host",
            HOST,
            "--port",
            str(port),
            "--model-path",
            model_path,
            "--output",
            str(client_output),
            "--concurrencies",
            ",".join(str(concurrency) for concurrency in settings.concurrencies),
            "--repeats",
            str(settings.repeats),
            *(["--warmup"] if settings.warmup else []),
            *(
                ["--max-samples", str(settings.max_samples)]
                if settings.max_samples is not None
                else []
            ),
            *workload.client_arguments,
            *settings.client_arguments,
        ],
        leg_dir / "client.log",
        check=True,
    )
    return [
        LoadMeasurement(
            concurrency=result.concurrency,
            total_requests=result.total,
            failed_requests=result.failed,
            throughput_requests_per_s=result.throughput_samples_per_s.mean,
            latency_p95_s=result.latency_p95_s.mean,
            first_output_latency_p95_s=(
                result.text_ttft_p95_s.mean
                if result.text_ttft_p95_s is not None
                else None
            ),
            accuracy=(
                {workload.error_rate: result.corpus_wer.mean}
                if result.corpus_wer.mean is not None
                else {}
            ),
            gpu_memory_peak_mib=(
                result.resources.gpu_memory_used_peak_mib.max
                if result.resources.gpu_memory_used_peak_mib is not None
                else None
            ),
        )
        for result in AsrClientPayload.model_validate_json(
            client_output.read_text()
        ).results
    ]


def measure_transcribe_diarize(
    workload: TranscribeDiarizeWorkload,
    settings: LoadSettings,
    *,
    port: int,
    model_path: str,
    leg_dir: Path,
) -> list[LoadMeasurement]:
    loads = []
    for concurrency in settings.concurrencies:
        speed_files = []
        diarization_metrics = []
        for repeat in range(1, settings.repeats + 1):
            run_dir = leg_dir / f"c{concurrency}_r{repeat}"
            # note (wilsonzheng0327): the client exits 1 after saving its files when any request failed, and the comparison counts failures itself.
            run_client(
                [
                    TRANSCRIBE_DIARIZE_CLIENT,
                    "--dataset",
                    workload.dataset,
                    "--use-existing-server",
                    "--host",
                    HOST,
                    "--port",
                    str(port),
                    "--model-path",
                    model_path,
                    "--concurrency",
                    str(concurrency),
                    # note (wilsonzheng0327): the benchmark runner's own default, one warmup request per concurrent slot.
                    "--warmup",
                    str(concurrency if settings.warmup else 0),
                    "--output-dir",
                    str(run_dir),
                    "--disable-tqdm",
                    *(
                        ["--max-samples", str(settings.max_samples)]
                        if settings.max_samples is not None
                        else []
                    ),
                    *settings.client_arguments,
                ],
                run_dir.with_suffix(".log"),
                check=False,
            )
            speed_files.append(
                TranscribeDiarizeSpeedFile.model_validate_json(
                    (run_dir / SPEED_RESULTS_FILE).read_text()
                )
            )
            diarization_metrics.append(
                TranscribeDiarizeResultsFile.model_validate_json(
                    (run_dir / RESULTS_FILE).read_text()
                ).diarization_metrics
            )
        accuracy: dict[AccuracyMetric, float | None] = {
            "CER": mean_or_none(
                [
                    metrics.cer
                    for metrics in diarization_metrics
                    if metrics.cer is not None
                ]
            ),
            "DER": mean_or_none(
                [
                    metrics.speaker_timestamp_der
                    for metrics in diarization_metrics
                    if metrics.speaker_timestamp_der is not None
                ]
            ),
        }
        loads.append(
            LoadMeasurement(
                concurrency=concurrency,
                total_requests=sum(run.speed.total_requests for run in speed_files),
                failed_requests=sum(run.speed.failed_requests for run in speed_files),
                # note (wilsonzheng0327): the client rounds throughput_qps to 3 decimals, which hides a 1% gap at the 0.03 requests/s of a 38-minute meeting.
                throughput_requests_per_s=mean_or_none(
                    [
                        run.speed.completed_requests / run.config.wall_clock_s
                        for run in speed_files
                        if run.config.wall_clock_s > 0
                    ]
                ),
                latency_p95_s=mean_or_none(
                    [
                        run.speed.latency_p95_s
                        for run in speed_files
                        if run.speed.latency_p95_s is not None
                    ]
                ),
                first_output_latency_p95_s=mean_or_none(
                    [
                        run.speed.text_ttft_p95_s
                        for run in speed_files
                        if run.speed.text_ttft_p95_s is not None
                    ]
                ),
                accuracy={
                    metric: value
                    for metric, value in accuracy.items()
                    if value is not None
                },
                gpu_memory_peak_mib=None,
            )
        )
    return loads
