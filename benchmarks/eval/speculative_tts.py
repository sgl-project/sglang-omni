# SPDX-License-Identifier: Apache-2.0
"""Generate and score TTS workloads for the shared speculative sweep."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.eval.speculative_workloads import (
    HOST,
    LoadSettings,
    TtsWorkload,
    run_client,
)
from benchmarks.metrics.speculative import LoadMeasurement, mean_or_none

TTS_CLIENT = "benchmarks.eval.benchmark_tts_seedtts"


class TtsSpeedSummary(BaseModel):
    total_requests: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    wall_clock_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    gpu_memory_peak_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    latency_p95_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    audio_ttfp_p95_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class TtsRequestResult(BaseModel):
    id: str
    is_success: bool
    audio_ttfp_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class TtsSpeedConfig(BaseModel):
    concurrency: int
    stream: bool


class TtsSpeedFile(BaseModel):
    config: TtsSpeedConfig
    summary: TtsSpeedSummary
    per_request: list[TtsRequestResult]


class TtsQualitySummary(BaseModel):
    total_samples: int = Field(ge=0)
    evaluated: int = Field(ge=0)
    skipped: int = Field(ge=0)
    wer_corpus: float = Field(ge=0, allow_inf_nan=False)


class TtsQualitySample(BaseModel):
    id: str
    is_success: bool
    is_transcribed: bool


class TtsQualityFile(BaseModel):
    summary: TtsQualitySummary
    per_sample: list[TtsQualitySample]


def generate_tts(
    workload: TtsWorkload,
    settings: LoadSettings,
    *,
    port: int,
    model_path: str,
    leg_dir: Path,
) -> None:
    """Save every concurrency and repeat on the running TTS server."""
    run_client(
        [
            TTS_CLIENT,
            "--generate-only",
            "--measure-gpu-memory",
            "--use-existing-server",
            "--host",
            HOST,
            "--port",
            str(port),
            "--model",
            model_path,
            "--meta",
            SEEDTTS_DATASET_ID,
            "--lang",
            workload.language,
            "--output-dir",
            str(leg_dir),
            "--concurrencies",
            ",".join(str(concurrency) for concurrency in settings.concurrencies),
            "--repeats",
            str(settings.repeats),
            # note (swsqy): the TTS client defaults to one warmup request per concurrent slot; its flag takes a count, not a boolean.
            *([] if settings.warmup else ["--warmup", "0"]),
            *(
                ["--max-samples", str(settings.max_samples)]
                if settings.max_samples is not None
                else []
            ),
            *settings.client_arguments,
        ],
        leg_dir / "generate.log",
        check=True,
    )


def measure_tts(
    workload: TtsWorkload,
    settings: LoadSettings,
    *,
    port: int,
    model_path: str,
    asr_model_path: str,
    asr_concurrency: int,
    leg_dir: Path,
) -> list[LoadMeasurement]:
    """Score saved audio on the running ASR server and read TTS timings."""
    loads = []
    for concurrency in settings.concurrencies:
        speed_summaries = []
        error_rates = []
        incomplete_reasons = []
        for repeat in range(1, settings.repeats + 1):
            point_name = (
                f"c{concurrency}"
                if settings.repeats == 1
                else f"c{concurrency}_r{repeat}"
            )
            point_dir = leg_dir / point_name
            speed = TtsSpeedFile.model_validate_json(
                (point_dir / "speed_results.json").read_text()
            )
            if speed.config.concurrency != concurrency:
                raise ValueError(
                    f"{point_dir}: unexpected concurrency in speed results"
                )
            else:
                pass
            run_client(
                [
                    TTS_CLIENT,
                    "--transcribe-only",
                    "--use-existing-server",
                    "--host",
                    HOST,
                    "--port",
                    str(port),
                    "--model",
                    model_path,
                    "--meta",
                    SEEDTTS_DATASET_ID,
                    "--lang",
                    workload.language,
                    "--output-dir",
                    str(point_dir),
                    "--concurrency",
                    str(concurrency),
                    "--asr-model-path",
                    asr_model_path,
                    "--asr-concurrency",
                    str(asr_concurrency),
                    *(
                        ["--max-samples", str(settings.max_samples)]
                        if settings.max_samples is not None
                        else []
                    ),
                    *settings.client_arguments,
                ],
                point_dir / "transcribe.log",
                check=True,
            )
            quality = TtsQualityFile.model_validate_json(
                (point_dir / "wer_results.json").read_text()
            )
            requests = {request.id: request for request in speed.per_request}
            samples = {sample.id: sample for sample in quality.per_sample}
            completed = sum(request.is_success for request in speed.per_request)
            evaluated = sum(sample.is_success for sample in quality.per_sample)
            if (
                len(requests) != len(speed.per_request)
                or len(samples) != len(quality.per_sample)
                or requests.keys() != samples.keys()
                or len(requests) != speed.summary.total_requests
                or completed != speed.summary.completed_requests
                or speed.summary.failed_requests != len(requests) - completed
                or quality.summary.total_samples != len(samples)
                or quality.summary.evaluated != evaluated
                or quality.summary.skipped != len(samples) - evaluated
                or any(
                    sample.is_success
                    and (
                        not sample.is_transcribed or not requests[sample.id].is_success
                    )
                    for sample in quality.per_sample
                )
            ):
                raise ValueError(f"{point_dir}: inconsistent TTS result coverage")
            else:
                pass
            transcription_failures = sum(
                request.is_success and not samples[request.id].is_transcribed
                for request in speed.per_request
            )
            if transcription_failures:
                incomplete_reasons.append(
                    f"{point_name}: {transcription_failures} generated samples were not transcribed"
                )
            else:
                pass
            if evaluated:
                error_rates.append(quality.summary.wer_corpus)
            else:
                incomplete_reasons.append(f"{point_name}: no scored samples")
            if (
                not speed.summary.wall_clock_s
                or speed.summary.latency_p95_s is None
                or speed.summary.total_requests == 0
            ):
                incomplete_reasons.append(
                    f"{point_name}: missing performance measurements"
                )
            else:
                pass
            if speed.config.stream and (
                speed.summary.audio_ttfp_p95_s is None
                or any(
                    request.is_success and request.audio_ttfp_s is None
                    for request in speed.per_request
                )
            ):
                incomplete_reasons.append(
                    f"{point_name}: missing streaming first-output latency"
                )
            else:
                pass
            speed_summaries.append(speed.summary)
        loads.append(
            LoadMeasurement(
                concurrency=concurrency,
                total_requests=sum(
                    summary.total_requests for summary in speed_summaries
                ),
                failed_requests=sum(
                    summary.failed_requests for summary in speed_summaries
                ),
                throughput_requests_per_s=mean_or_none(
                    [
                        summary.completed_requests / summary.wall_clock_s
                        for summary in speed_summaries
                        if summary.wall_clock_s
                    ]
                ),
                latency_p95_s=mean_or_none(
                    [
                        summary.latency_p95_s
                        for summary in speed_summaries
                        if summary.latency_p95_s is not None
                    ]
                ),
                first_output_latency_p95_s=mean_or_none(
                    [
                        summary.audio_ttfp_p95_s
                        for summary in speed_summaries
                        if summary.audio_ttfp_p95_s is not None
                    ]
                ),
                accuracy=(
                    {workload.error_rate: mean(error_rates)} if error_rates else {}
                ),
                gpu_memory_peak_mib=max(
                    (
                        summary.gpu_memory_peak_mib
                        for summary in speed_summaries
                        if summary.gpu_memory_peak_mib is not None
                    ),
                    default=None,
                ),
                incomplete_reasons=incomplete_reasons,
            )
        )
    return loads
