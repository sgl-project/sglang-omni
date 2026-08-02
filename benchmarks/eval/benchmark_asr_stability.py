# SPDX-License-Identifier: Apache-2.0
"""Functional and sustained-load validation for consumer ASR servers.

The harness exercises Qwen3-ASR, Fun-ASR-Nano, or Whisper through the public
OpenAI-compatible audio endpoints. It records functional, streaming, mixed-load,
chaos, health, resource, and GPU-memory-retention evidence in one JSON artifact.
Model revisions are references only: pass ``--model-revision`` to declare the
checkpoint that the running server actually loaded.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import soundfile as sf

from benchmarks.dataset.prepare import DATASETS, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.runtime_metrics import ResourceMonitor, collect_benchmark_provenance
from benchmarks.tasks.asr import (
    FUN_ASR_MODEL_PATH,
    OMNI_WHISPER_MODEL_PATH,
    PINNED_ASR_MODEL_REVISIONS,
    QWEN3_ASR_MODEL_PATH,
)
from benchmarks.tts_serving.http_contracts import (
    MAX_HTTP_RESPONSE_BYTES,
    ResponseBodyTooLarge,
    read_response_body,
)

SAMPLE_RATE = 16000
STREAM_ROUTE_HEADERS = {"x-sglang-omni-route-stream": "true"}
DEFAULT_AUDIO_BOUNDARY_MODELS = {FUN_ASR_MODEL_PATH}
PINNED_MODEL_REVISIONS = PINNED_ASR_MODEL_REVISIONS
MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_SSE_EVENTS = 100_000
LATENCY_RESERVOIR_SIZE = 8192


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    language: str
    audio_bytes: bytes
    duration_s: float


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _parse_concurrencies(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("--concurrencies must contain positive integers")
    try:
        concurrencies = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("--concurrencies must contain positive integers") from exc
    if any(concurrency <= 0 for concurrency in concurrencies):
        raise ValueError("--concurrencies must contain positive integers")
    return concurrencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--model-path",
        default=FUN_ASR_MODEL_PATH,
        help=(
            "Served ASR model id. Supported consumer profiles: "
            f"{QWEN3_ASR_MODEL_PATH}, {FUN_ASR_MODEL_PATH}, and "
            f"{OMNI_WHISPER_MODEL_PATH}."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Exact revision loaded by the server; never inferred.",
    )
    parser.add_argument("--meta", default=DATASETS["seedtts"])
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Exact dataset revision; canonical SeedTTS uses a pinned default.",
    )
    parser.add_argument("--duration-s", type=_positive_float, default=1800.0)
    parser.add_argument("--concurrencies", default="1,4,8,16")
    parser.add_argument("--samples-per-language", type=_positive_int, default=20)
    parser.add_argument("--request-timeout-s", type=_positive_float, default=60.0)
    parser.add_argument("--max-audio-duration-s", type=_positive_float, default=30.0)
    parser.add_argument(
        "--check-audio-boundary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Check just-under/just-over max audio duration. Defaults on for "
            "Fun-ASR; enable explicitly for a Whisper server with a matching "
            "over-limit rejection contract."
        ),
    )
    parser.add_argument(
        "--include-translation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Exercise /v1/audio/translations. Defaults off; enable explicitly "
            "for a Whisper server that exposes the translation route."
        ),
    )
    parser.add_argument(
        "--translation-source-language",
        default="zh",
        help="Required source-language hint used for translation checks.",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--mm-attention-backend", default=None)
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--torch-compile", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument(
        "--min-free-memory-mib", type=_nonnegative_float, default=2048.0
    )
    parser.add_argument(
        "--max-retained-memory-mib",
        type=_nonnegative_float,
        default=256.0,
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--monitor-interval-s", type=_positive_float, default=0.2)
    parser.add_argument("--chaos-interval-s", type=_positive_float, default=30.0)
    parser.add_argument("--cooldown-s", type=_nonnegative_float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--launch-command",
        default=os.environ.get("SGLANG_OMNI_BENCHMARK_LAUNCH_COMMAND"),
    )
    parser.add_argument("--output", default="asr_stability_results.json")
    return parser.parse_args()


def _resolve_dataset_revision(args: argparse.Namespace) -> str | None:
    if _is_local_source(args.meta):
        return None
    if args.dataset_revision is not None:
        return args.dataset_revision
    if args.meta == DATASETS["seedtts"]:
        return SEEDTTS_DATASET_REVISION
    return None


def _is_local_source(source: str) -> bool:
    return os.path.isfile(source) or source.endswith(".lst")


def _validate_args(args: argparse.Namespace) -> list[int]:
    concurrencies = (
        _parse_concurrencies(args.concurrencies)
        if isinstance(args.concurrencies, str)
        else list(args.concurrencies)
    )
    constraints = (
        (
            bool(concurrencies) and all(value > 0 for value in concurrencies),
            "--concurrencies must contain positive integers",
        ),
        (
            math.isfinite(args.duration_s) and args.duration_s > 0,
            "--duration-s must be finite and > 0",
        ),
        (
            args.samples_per_language > 0,
            "--samples-per-language must be > 0",
        ),
        (
            math.isfinite(args.request_timeout_s) and args.request_timeout_s > 0,
            "--request-timeout-s must be finite and > 0",
        ),
        (
            math.isfinite(args.max_audio_duration_s)
            and args.max_audio_duration_s >= 0.2,
            "--max-audio-duration-s must be finite and at least 0.2",
        ),
        (
            math.isfinite(args.monitor_interval_s) and args.monitor_interval_s > 0,
            "--monitor-interval-s must be finite and > 0",
        ),
        (
            math.isfinite(args.chaos_interval_s) and args.chaos_interval_s > 0,
            "--chaos-interval-s must be finite and > 0",
        ),
        (
            math.isfinite(args.cooldown_s) and args.cooldown_s >= 0,
            "--cooldown-s must be finite and nonnegative",
        ),
        (
            math.isfinite(args.min_free_memory_mib) and args.min_free_memory_mib >= 0,
            "--min-free-memory-mib must be finite and nonnegative",
        ),
        (
            math.isfinite(args.max_retained_memory_mib)
            and args.max_retained_memory_mib >= 0,
            "--max-retained-memory-mib must be finite and nonnegative",
        ),
        (args.gpu_index >= 0, "--gpu-index must be nonnegative"),
        (
            not _is_local_source(args.meta),
            "bilingual stability validation requires a dataset with distinct "
            "en and zh splits; one local meta.lst cannot represent both",
        ),
    )
    for valid, message in constraints:
        if not valid:
            raise ValueError(message)
    if args.include_translation is None:
        args.include_translation = False
    if args.include_translation and not str(args.translation_source_language).strip():
        raise ValueError(
            "--translation-source-language is required when translation " "is enabled"
        )
    if args.check_audio_boundary is None:
        args.check_audio_boundary = args.model_path in DEFAULT_AUDIO_BOUNDARY_MODELS
    return concurrencies


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    concurrencies = _validate_args(args)
    dataset_revision = _resolve_dataset_revision(args)
    samples = _load_prepared_samples(args, dataset_revision)
    evaluation_input_sha256 = _evaluation_input_sha256(samples)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=max(concurrencies) + 8)
    monitor = ResourceMonitor(
        gpu_index=args.gpu_index,
        interval_s=args.monitor_interval_s,
    ).start()
    memory_checkpoints: list[dict[str, Any]] = []
    resources: dict[str, Any]

    try:
        await _wait_for_resource_sample(monitor)
        memory_checkpoints.append(_memory_checkpoint("before_functional", monitor))
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        ) as session:
            functional = await _run_functional_checks(session, args, samples)
            memory_checkpoints.append(_memory_checkpoint("after_functional", monitor))
            stages, chaos = await _run_soak(
                session,
                args,
                samples,
                concurrencies=concurrencies,
                monitor=monitor,
            )
            memory_checkpoints.append(_memory_checkpoint("after_soak", monitor))
            await asyncio.sleep(args.cooldown_s)
            memory_checkpoints.append(_memory_checkpoint("after_cooldown", monitor))
            health_status = await _health_status(session, args)
    finally:
        resources = monitor.stop()

    memory_validation = _validate_memory_retention(
        memory_checkpoints,
        resources,
        min_free_memory_mib=args.min_free_memory_mib,
        max_retained_memory_mib=args.max_retained_memory_mib,
    )
    unexpected_errors = sum(stage["unexpected_errors"] for stage in stages)
    soak_exercised = all(
        stage["requests"] > 0
        and stage["chaos_events"] > 0
        and (not args.include_translation or stage["translations"] > 0)
        for stage in stages
    )
    passed = (
        all(check["passed"] for check in functional)
        and soak_exercised
        and unexpected_errors == 0
        and all(event["passed"] for event in chaos)
        and health_status == 200
        and memory_validation["passed"]
    )
    server_config = {
        "dtype": args.dtype,
        "quantization": args.quantization,
        "attention_backend": args.attention_backend,
        "mm_attention_backend": args.mm_attention_backend,
        "cuda_graph": args.cuda_graph,
        "torch_compile": args.torch_compile,
        "max_running_requests": args.max_running_requests,
        "mem_fraction_static": args.mem_fraction_static,
    }
    return {
        "schema_version": 2,
        "passed": passed,
        "provenance": collect_benchmark_provenance(
            model_id=args.model_path,
            model_revision=args.model_revision,
            dataset_id=args.meta,
            dataset_revision=dataset_revision,
            launch_command=args.launch_command,
            server_config=server_config,
            evaluation_input_sha256=evaluation_input_sha256,
        ),
        "config": {
            "host": args.host,
            "port": args.port,
            "duration_s": args.duration_s,
            "concurrencies": concurrencies,
            "samples_per_language": args.samples_per_language,
            "request_timeout_s": args.request_timeout_s,
            "max_audio_duration_s": args.max_audio_duration_s,
            "check_audio_boundary": args.check_audio_boundary,
            "include_translation": args.include_translation,
            "translation_source_language": (
                args.translation_source_language if args.include_translation else None
            ),
            "min_free_memory_mib": args.min_free_memory_mib,
            "max_retained_memory_mib": args.max_retained_memory_mib,
            "gpu_index": args.gpu_index,
            "monitor_interval_s": args.monitor_interval_s,
            "chaos_interval_s": args.chaos_interval_s,
            "cooldown_s": args.cooldown_s,
            "seed": args.seed,
            "model_path": args.model_path,
            "declared_model_revision": args.model_revision,
            "dataset_revision": dataset_revision,
            "declared_server": server_config,
        },
        "functional": functional,
        "soak_stages": stages,
        "chaos_events": chaos,
        "resources": resources,
        "memory_checkpoints": memory_checkpoints,
        "memory_validation": memory_validation,
        "final_health_status": health_status,
        "unexpected_errors": unexpected_errors,
    }


def _load_prepared_samples(
    args: argparse.Namespace,
    dataset_revision: str | None,
) -> list[PreparedSample]:
    samples_by_language: dict[str, list[PreparedSample]] = {}
    for language in ("en", "zh"):
        loaded = load_seedtts_samples(
            args.meta,
            max_samples=args.samples_per_language,
            split=language,
            revision=dataset_revision,
        )
        if not loaded:
            raise RuntimeError(
                f"No SeedTTS {language} samples loaded from {args.meta!r}"
            )
        samples_by_language[language] = [
            _prepare_sample(sample, language) for sample in loaded
        ]

    prepared: list[PreparedSample] = []
    sample_count = max(
        len(language_samples) for language_samples in samples_by_language.values()
    )
    for index in range(sample_count):
        for language in ("en", "zh"):
            language_samples = samples_by_language[language]
            if index < len(language_samples):
                prepared.append(language_samples[index])
    return prepared


def _prepare_sample(sample: SampleInput, language: str) -> PreparedSample:
    audio_bytes = Path(sample.ref_audio).read_bytes()
    return PreparedSample(
        sample_id=sample.sample_id,
        language=language,
        audio_bytes=audio_bytes,
        duration_s=_duration_s(audio_bytes),
    )


def _evaluation_input_sha256(samples: list[PreparedSample]) -> str:
    digest = hashlib.sha256(b"asr-stability-input-v1\0")
    for sample in samples:
        for value in (sample.sample_id, sample.language):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(len(sample.audio_bytes).to_bytes(8, "big"))
        digest.update(sample.audio_bytes)
    return digest.hexdigest()


async def _wait_for_resource_sample(monitor: ResourceMonitor) -> None:
    deadline = time.monotonic() + max(1.0, monitor.interval_s * 5)
    while not monitor.samples and monitor.error is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.01, remaining))


async def _run_functional_checks(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    samples: list[PreparedSample],
) -> list[dict[str, Any]]:
    en_sample = next(sample for sample in samples if sample.language == "en")
    zh_sample = next(sample for sample in samples if sample.language == "zh")
    checks: list[dict[str, Any]] = []

    en_result = await _post_transcription(session, args, en_sample)
    checks.append(_expect_success("basic_en", en_result))
    zh_result = await _post_transcription(session, args, zh_sample)
    checks.append(_expect_success("basic_zh", zh_result))
    if args.include_translation:
        translation = await _post_translation(session, args, zh_sample)
        checks.append(
            _expect_translation_success(
                "basic_zh_to_english",
                translation,
            )
        )

    stream_result = await _post_streaming_transcription(
        session,
        args,
        en_sample,
    )
    checks.append(
        {
            "name": "streaming_consistency",
            "passed": (
                stream_result["status"] == 200
                and stream_result["done"]
                and stream_result["text"] == en_result["text"]
            ),
            "stream_text": stream_result["text"],
            "non_stream_text": en_result["text"],
            **{key: value for key, value in stream_result.items() if key != "text"},
        }
    )

    cancellation = await _cancel_stream(session, args, en_sample)
    await asyncio.sleep(0.5)
    reconnect = await _post_streaming_transcription(session, args, en_sample)
    checks.append(
        {
            "name": "stream_cancel_and_reconnect",
            "passed": (
                cancellation["status"] == 200
                and cancellation["received_event"]
                and reconnect["status"] == 200
                and reconnect["done"]
            ),
            "cancel": cancellation,
            "reconnect_status": reconnect["status"],
            "reconnect_error": reconnect["error"],
        }
    )

    checks.append(
        _expect_status(
            "empty_audio",
            await _post_raw_audio(session, args, b"", "empty.wav", "en"),
            400,
        )
    )
    checks.append(
        _expect_status(
            "corrupt_audio",
            await _post_raw_audio(
                session,
                args,
                b"not-an-audio-file",
                "corrupt.wav",
                "en",
            ),
            400,
        )
    )

    if args.check_audio_boundary:
        near_limit = _resize_wav(
            en_sample.audio_bytes,
            args.max_audio_duration_s - 0.1,
        )
        checks.append(
            _expect_success(
                "near_limit_audio",
                await _post_raw_audio(
                    session,
                    args,
                    near_limit,
                    "near-limit.wav",
                    "en",
                ),
            )
        )
        over_limit = _resize_wav(
            en_sample.audio_bytes,
            args.max_audio_duration_s + 0.1,
        )
        checks.append(
            _expect_status(
                "over_limit_audio",
                await _post_raw_audio(
                    session,
                    args,
                    over_limit,
                    "over-limit.wav",
                    "en",
                ),
                400,
            )
        )
    return checks


async def _run_soak(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    samples: list[PreparedSample],
    *,
    concurrencies: list[int],
    monitor: ResourceMonitor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    randomizer = random.Random(args.seed)
    stage_duration_s = args.duration_s / len(concurrencies)
    stages: list[dict[str, Any]] = []
    chaos_events: list[dict[str, Any]] = []

    for concurrency in concurrencies:
        deadline = time.monotonic() + stage_duration_s
        counters: dict[str, int | float] = {
            "requests_issued": 0,
            "requests": 0,
            "successes": 0,
            "unexpected_errors": 0,
            "en": 0,
            "zh": 0,
            "zh_requests_issued": 0,
            "translations": 0,
            "chaos_requests": 0,
            "chaos_events": 0,
            "successful_audio_s": 0.0,
        }
        latency_count = 0
        latency_total_s = 0.0
        latency_reservoir: list[float] = []
        latency_randomizer = random.Random(args.seed ^ concurrency)
        request_slots = asyncio.Semaphore(concurrency)
        active_requests = 0
        max_in_flight_observed = 0

        def record_latency(value: float) -> None:
            nonlocal latency_count, latency_total_s
            latency_count += 1
            latency_total_s += value
            if len(latency_reservoir) < LATENCY_RESERVOIR_SIZE:
                latency_reservoir.append(value)
                return
            replacement = latency_randomizer.randrange(latency_count)
            if replacement < LATENCY_RESERVOIR_SIZE:
                latency_reservoir[replacement] = value

        async def limited_request(
            call: Callable[..., Awaitable[dict[str, Any]]],
            *call_args: Any,
        ) -> dict[str, Any]:
            nonlocal active_requests, max_in_flight_observed
            async with request_slots:
                active_requests += 1
                max_in_flight_observed = max(
                    max_in_flight_observed,
                    active_requests,
                )
                try:
                    return await call(*call_args)
                finally:
                    active_requests -= 1

        async def worker(worker_id: int) -> None:
            index = worker_id
            while time.monotonic() < deadline:
                sample = samples[index % len(samples)]
                index += concurrency
                counters["requests_issued"] += 1
                translate = False
                if sample.language == "zh":
                    zh_sequence = int(counters["zh_requests_issued"])
                    counters["zh_requests_issued"] += 1
                    translate = args.include_translation and zh_sequence % 4 == 0
                if translate:
                    result = await limited_request(
                        _post_translation,
                        session,
                        args,
                        sample,
                    )
                    counters["translations"] += 1
                else:
                    result = await limited_request(
                        _post_transcription,
                        session,
                        args,
                        sample,
                    )
                counters["requests"] += 1
                counters[sample.language] += 1
                record_latency(result["latency_s"])
                successful_text = bool(result["text"])
                if translate:
                    successful_text = _looks_like_english_translation(result["text"])
                if result["status"] == 200 and successful_text:
                    counters["successes"] += 1
                    counters["successful_audio_s"] += sample.duration_s
                else:
                    counters["unexpected_errors"] += 1

        async def chaos_worker() -> None:
            event_number = 0
            while True:
                if event_number == 0:
                    await asyncio.sleep(0)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    await asyncio.sleep(min(args.chaos_interval_s, remaining))
                    if time.monotonic() >= deadline:
                        return
                sample = randomizer.choice(samples)
                if event_number % 2 == 0:
                    counters["chaos_requests"] += 1
                    result = await limited_request(
                        _post_raw_audio,
                        session,
                        args,
                        b"invalid",
                        "intentional-corrupt.wav",
                        sample.language,
                    )
                    chaos_events.append(
                        {
                            "stage_concurrency": concurrency,
                            "kind": "malformed",
                            "status": result["status"],
                            "passed": result["status"] == 400,
                            "error": result["error"],
                        }
                    )
                else:
                    counters["chaos_requests"] += 1
                    cancel = await limited_request(
                        _cancel_stream,
                        session,
                        args,
                        sample,
                    )
                    counters["chaos_requests"] += 1
                    reconnect = await limited_request(
                        _post_streaming_transcription,
                        session,
                        args,
                        sample,
                    )
                    chaos_events.append(
                        {
                            "stage_concurrency": concurrency,
                            "kind": "cancel_reconnect",
                            "status": cancel["status"],
                            "passed": (
                                cancel["status"] == 200
                                and cancel["received_event"]
                                and reconnect["status"] == 200
                                and reconnect["done"]
                            ),
                            "cancel_error": cancel["error"],
                            "reconnect_error": reconnect["error"],
                        }
                    )
                counters["chaos_events"] += 1
                event_number += 1

        started = time.monotonic()
        tasks = [
            asyncio.create_task(worker(worker_id)) for worker_id in range(concurrency)
        ]
        tasks.append(asyncio.create_task(chaos_worker()))
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        elapsed_s = time.monotonic() - started
        zh_requests = int(counters["zh"])
        stages.append(
            {
                "concurrency": concurrency,
                "max_in_flight_observed": max_in_flight_observed,
                "duration_s": elapsed_s,
                **counters,
                "total_http_requests": (
                    int(counters["requests"]) + int(counters["chaos_requests"])
                ),
                "translation_target_ratio": (0.25 if args.include_translation else 0.0),
                "translation_observed_ratio": (
                    float(counters["translations"]) / zh_requests
                    if zh_requests
                    else None
                ),
                "throughput_requests_per_s": (
                    float(counters["successes"]) / max(elapsed_s, 1e-9)
                ),
                "audio_seconds_per_s": (
                    float(counters["successful_audio_s"]) / max(elapsed_s, 1e-9)
                ),
                "latency_mean_s": (
                    latency_total_s / latency_count if latency_count else None
                ),
                "latency_p95_s": _percentile(latency_reservoir, 95),
                "latency_samples_seen": latency_count,
                "latency_reservoir_size": len(latency_reservoir),
                "latency_percentile_method": "bounded_reservoir",
                "memory": _memory_checkpoint(
                    f"after_concurrency_{concurrency}", monitor
                ),
            }
        )
    return stages, chaos_events


async def _post_transcription(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    sample: PreparedSample,
) -> dict[str, Any]:
    return await _post_raw_audio(
        session,
        args,
        sample.audio_bytes,
        f"{sample.sample_id}.wav",
        sample.language,
    )


async def _post_translation(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    sample: PreparedSample,
) -> dict[str, Any]:
    return await _post_raw_audio(
        session,
        args,
        sample.audio_bytes,
        f"{sample.sample_id}.wav",
        str(args.translation_source_language).strip(),
        endpoint="translations",
    )


async def _post_raw_audio(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    audio_bytes: bytes,
    filename: str,
    language: str | None,
    *,
    endpoint: str = "transcriptions",
) -> dict[str, Any]:
    form = aiohttp.FormData()
    form.add_field("model", args.model_path)
    if language is not None:
        form.add_field("language", language)
    form.add_field("response_format", "json")
    form.add_field(
        "file",
        audio_bytes,
        filename=filename,
        content_type="audio/wav",
    )
    started = time.perf_counter()
    try:
        async with session.post(_url(args, endpoint), data=form) as response:
            try:
                body_bytes = await read_response_body(response)
            except ResponseBodyTooLarge as exc:
                return {
                    "status": response.status,
                    "text": "",
                    "body": "",
                    "latency_s": time.perf_counter() - started,
                    "error": str(exc),
                }
            body = body_bytes.decode("utf-8", errors="replace")
            text = ""
            error = None
            if response.status == 200:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    error = "response body is not valid JSON"
                else:
                    if not isinstance(payload, dict):
                        error = "response JSON must be an object"
                    elif isinstance(payload.get("text"), str):
                        text = payload["text"]
                    else:
                        error = "response JSON has no string text field"
            return {
                "status": response.status,
                "text": text,
                "body": body[:500],
                "latency_s": time.perf_counter() - started,
                "error": error,
            }
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return _transport_error_result(started, exc)


async def _post_streaming_transcription(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    sample: PreparedSample,
) -> dict[str, Any]:
    event_count = 0
    delta_event_count = 0
    first_event_latency_s: float | None = None
    seen_done_event = False
    seen_done_sentinel = False
    final_text = ""
    status = 0
    bytes_read = 0
    started = time.perf_counter()
    error: str | None = None
    try:
        async with session.post(
            _url(args),
            data=_stream_form(args, sample),
            headers=STREAM_ROUTE_HEADERS,
        ) as response:
            status = response.status
            async for raw_line in response.content:
                bytes_read = _checked_stream_bytes(bytes_read, raw_line)
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    seen_done_sentinel = True
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                event_count += 1
                if event_count > MAX_SSE_EVENTS:
                    raise ValueError(
                        "SSE event count exceeded benchmark cap " f"({MAX_SSE_EVENTS})"
                    )
                if first_event_latency_s is None:
                    first_event_latency_s = time.perf_counter() - started
                event_type = event.get("type")
                if event_type == "transcript.text.delta":
                    delta_event_count += 1
                elif event_type == "transcript.text.done":
                    seen_done_event = True
                    if isinstance(event.get("text"), str):
                        final_text = event["text"]
    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ResponseBodyTooLarge,
        ValueError,
    ) as exc:
        error = f"{type(exc).__name__}: {exc}"

    return {
        "status": status,
        "events": event_count,
        "delta_events": delta_event_count,
        "response_bytes": bytes_read,
        "first_event_latency_s": first_event_latency_s,
        "done": seen_done_event and seen_done_sentinel,
        "text": final_text,
        "error": error,
    }


async def _read_first_stream_delta(
    response: aiohttp.ClientResponse,
) -> tuple[bool, int]:
    bytes_read = 0
    async for raw_line in response.content:
        bytes_read = _checked_stream_bytes(bytes_read, raw_line)
        line = raw_line.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return False, bytes_read
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "transcript.text.done":
            return False, bytes_read
        if (
            event_type == "transcript.text.delta"
            and isinstance(event.get("delta"), str)
            and event["delta"]
        ):
            return True, bytes_read
    return False, bytes_read


async def _cancel_stream(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    sample: PreparedSample,
) -> dict[str, Any]:
    response: aiohttp.ClientResponse | None = None
    status = 0
    received_event = False
    bytes_read = 0
    error: str | None = None
    try:
        response = await session.post(
            _url(args),
            data=_stream_form(args, sample),
            headers=STREAM_ROUTE_HEADERS,
        )
        status = response.status
        received_event, bytes_read = await asyncio.wait_for(
            _read_first_stream_delta(response),
            timeout=min(10.0, args.request_timeout_s),
        )
    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ResponseBodyTooLarge,
        ValueError,
    ) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            response.close()
    return {
        "status": status,
        "received_event": received_event,
        "response_bytes": bytes_read,
        "error": error,
    }


def _stream_form(
    args: argparse.Namespace,
    sample: PreparedSample,
) -> aiohttp.FormData:
    form = aiohttp.FormData()
    form.add_field("model", args.model_path)
    form.add_field("language", sample.language)
    form.add_field("response_format", "json")
    form.add_field("stream", "true")
    form.add_field(
        "file",
        sample.audio_bytes,
        filename=f"{sample.sample_id}.wav",
        content_type="audio/wav",
    )
    return form


async def _health_status(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
) -> int:
    try:
        async with session.get(f"http://{args.host}:{args.port}/health") as response:
            await read_response_body(response)
            return response.status
    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ResponseBodyTooLarge,
    ):
        return 0


def _checked_stream_bytes(total: int, line: bytes) -> int:
    if len(line) > MAX_SSE_LINE_BYTES:
        raise ResponseBodyTooLarge(
            bytes_read=len(line),
            max_bytes=MAX_SSE_LINE_BYTES,
        )
    next_total = total + len(line)
    if next_total > MAX_HTTP_RESPONSE_BYTES:
        raise ResponseBodyTooLarge(
            bytes_read=next_total,
            max_bytes=MAX_HTTP_RESPONSE_BYTES,
        )
    return next_total


def _transport_error_result(
    started: float,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "status": 0,
        "text": "",
        "body": "",
        "latency_s": time.perf_counter() - started,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _looks_like_english_translation(text: str) -> bool:
    latin_letters = sum("a" <= character.lower() <= "z" for character in text)
    cjk_characters = sum(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in text
    )
    return latin_letters > cjk_characters


def _expect_translation_success(
    name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    text = result["text"] if isinstance(result.get("text"), str) else ""
    return {
        "name": name,
        "passed": (result["status"] == 200 and _looks_like_english_translation(text)),
        **result,
    }


def _expect_success(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": result["status"] == 200 and bool(result["text"]),
        **result,
    }


def _expect_status(
    name: str,
    result: dict[str, Any],
    expected_status: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": result["status"] == expected_status,
        "expected_status": expected_status,
        **result,
    }


def _resize_wav(audio_bytes: bytes, duration_s: float) -> bytes:
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        raise ValueError("cannot resize empty audio")
    target_samples = round(duration_s * SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        old_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_length = round(len(audio) * SAMPLE_RATE / sample_rate)
        new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        audio = np.interp(
            new_positions,
            old_positions,
            audio,
        ).astype(np.float32)
    repeats = max(1, (target_samples + len(audio) - 1) // len(audio))
    resized = np.tile(audio, repeats)[:target_samples]
    buffer = io.BytesIO()
    sf.write(buffer, resized, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _duration_s(audio_bytes: bytes) -> float:
    info = sf.info(io.BytesIO(audio_bytes))
    return info.frames / float(info.samplerate)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round((percentile / 100.0) * (len(ordered) - 1))),
    )
    return ordered[index]


def _memory_checkpoint(
    label: str,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    if not monitor.samples:
        return {
            "label": label,
            "monotonic_s": time.monotonic(),
            "sample_elapsed_s": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_free_mib": None,
            "gpu_process_memory_mib": None,
            "gpu_process_rss_mib": None,
            "error": monitor.error or "resource monitor has no samples",
        }
    sample = monitor.samples[-1]
    return {
        "label": label,
        "monotonic_s": time.monotonic(),
        "sample_elapsed_s": sample.elapsed_s,
        "gpu_memory_used_mib": sample.gpu_memory_used_mib,
        "gpu_memory_free_mib": sample.gpu_memory_free_mib,
        "gpu_process_memory_mib": sample.gpu_process_memory_mib,
        "gpu_process_rss_mib": sample.gpu_process_rss_mib,
        "error": monitor.error,
    }


def _validate_memory_retention(
    checkpoints: list[dict[str, Any]],
    resources: dict[str, Any],
    *,
    min_free_memory_mib: float,
    max_retained_memory_mib: float,
) -> dict[str, Any]:
    checkpoint_memory = {
        checkpoint["label"]: {
            "used_mib": checkpoint.get("gpu_memory_used_mib"),
            "free_mib": checkpoint.get("gpu_memory_free_mib"),
            "process_mib": checkpoint.get("gpu_process_memory_mib"),
            "process_rss_mib": checkpoint.get("gpu_process_rss_mib"),
            "error": checkpoint.get("error"),
        }
        for checkpoint in checkpoints
    }
    before = checkpoint_memory.get("before_functional")
    cooldown = checkpoint_memory.get("after_cooldown")
    free_summary = resources.get("gpu_memory_free_mib")
    minimum_free_raw = (
        free_summary.get("min") if isinstance(free_summary, dict) else None
    )
    before_used_raw = before.get("used_mib") if before is not None else None
    cooldown_used_raw = cooldown.get("used_mib") if cooldown is not None else None
    telemetry_error = (
        resources.get("error") is not None
        or before is None
        or before.get("error") is not None
        or cooldown is None
        or cooldown.get("error") is not None
    )
    required_values = (
        minimum_free_raw,
        before_used_raw,
        cooldown_used_raw,
    )
    if telemetry_error or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in required_values
    ):
        return {
            "passed": False,
            "error": "required GPU memory samples are unavailable",
            "checkpoints": checkpoint_memory,
        }
    minimum_free, before_used, cooldown_used = (
        float(value) for value in required_values
    )
    if not all(
        math.isfinite(value) for value in (minimum_free, before_used, cooldown_used)
    ):
        return {
            "passed": False,
            "error": "required GPU memory samples are unavailable",
            "checkpoints": checkpoint_memory,
        }

    retained_mib = cooldown_used - before_used
    failures: list[str] = []
    if minimum_free < min_free_memory_mib:
        failures.append("minimum free GPU memory fell below the required floor")
    if retained_mib > max_retained_memory_mib:
        failures.append("GPU memory retained after cooldown exceeded the limit")
    return {
        "passed": not failures,
        "error": "; ".join(failures) if failures else None,
        "minimum_free_mib": minimum_free,
        "required_free_mib": min_free_memory_mib,
        "retained_after_cooldown_mib": retained_mib,
        "maximum_retained_mib": max_retained_memory_mib,
        "checkpoints": checkpoint_memory,
    }


def _url(args: argparse.Namespace, endpoint: str = "transcriptions") -> str:
    return f"http://{args.host}:{args.port}/v1/audio/{endpoint}"


def main() -> None:
    args = parse_args()
    result = asyncio.run(main_async(args))
    output = Path(os.path.abspath(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": result["passed"], "output": str(output)}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
