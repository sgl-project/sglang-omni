# SPDX-License-Identifier: Apache-2.0
"""Functional and sustained-load validation for an ASR transcription server."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import soundfile as sf

from benchmarks.dataset.prepare import DATASETS, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.runtime_metrics import ResourceMonitor, collect_benchmark_provenance

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    language: str
    audio_bytes: bytes
    duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--meta", default=DATASETS["seedtts"])
    parser.add_argument("--dataset-revision", default=SEEDTTS_DATASET_REVISION)
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--concurrencies", default="1,4,8,16")
    parser.add_argument("--samples-per-language", type=int, default=20)
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--monitor-interval-s", type=float, default=0.2)
    parser.add_argument("--launch-command", default=None)
    parser.add_argument("--output", default="asr_stability_results.json")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be > 0")
    concurrencies = [
        int(value) for value in args.concurrencies.split(",") if value.strip()
    ]
    if not concurrencies or any(value < 1 for value in concurrencies):
        raise ValueError("--concurrencies must contain positive integers")

    samples = _load_prepared_samples(args)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=max(concurrencies) + 8)
    resource_monitor = ResourceMonitor(
        gpu_index=args.gpu_index,
        interval_s=args.monitor_interval_s,
    ).start()
    memory_checkpoints: list[dict[str, Any]] = [
        _memory_checkpoint("before_functional", args.gpu_index)
    ]

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        functional = await _run_functional_checks(session, args, samples)
        memory_checkpoints.append(
            _memory_checkpoint("after_functional", args.gpu_index)
        )
        stages, chaos = await _run_soak(
            session,
            args,
            samples,
            concurrencies=concurrencies,
        )
        memory_checkpoints.append(_memory_checkpoint("after_soak", args.gpu_index))
        await asyncio.sleep(5)
        memory_checkpoints.append(_memory_checkpoint("after_cooldown", args.gpu_index))
        health_status = await _health_status(session, args)

    resources = resource_monitor.stop()
    unexpected_errors = sum(stage["unexpected_errors"] for stage in stages)
    passed = (
        all(check["passed"] for check in functional)
        and unexpected_errors == 0
        and all(event["passed"] for event in chaos)
        and health_status == 200
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "provenance": collect_benchmark_provenance(
            model_id=args.model_path,
            model_revision=args.model_revision,
            dataset_id=args.meta,
            dataset_revision=args.dataset_revision,
            launch_command=args.launch_command,
            server_config={
                "dtype": "bfloat16",
                "attention_backend": "flashinfer",
                "mm_attention_backend": "triton_attn",
                "cuda_graph": True,
                "torch_compile": False,
                "max_running_requests": 16,
                "mem_fraction_static": 0.65,
            },
        ),
        "config": {
            "host": args.host,
            "port": args.port,
            "duration_s": args.duration_s,
            "concurrencies": concurrencies,
            "samples_per_language": args.samples_per_language,
            "seed": 123,
        },
        "functional": functional,
        "soak_stages": stages,
        "chaos_events": chaos,
        "resources": resources,
        "memory_checkpoints": memory_checkpoints,
        "final_health_status": health_status,
        "unexpected_errors": unexpected_errors,
    }


def _load_prepared_samples(args: argparse.Namespace) -> list[PreparedSample]:
    prepared: list[PreparedSample] = []
    for language in ("en", "zh"):
        loaded = load_seedtts_samples(
            args.meta,
            max_samples=args.samples_per_language,
            split=language,
            revision=args.dataset_revision,
        )
        prepared.extend(_prepare_sample(sample, language) for sample in loaded)
    if not prepared:
        raise RuntimeError("No SeedTTS samples were loaded")
    return prepared


def _prepare_sample(sample: SampleInput, language: str) -> PreparedSample:
    audio_bytes = Path(sample.ref_audio).read_bytes()
    return PreparedSample(
        sample_id=sample.sample_id,
        language=language,
        audio_bytes=audio_bytes,
        duration_s=_duration_s(audio_bytes),
    )


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

    stream_result = await _post_streaming_transcription(session, args, en_sample)
    checks.append(
        {
            "name": "streaming_consistency",
            "passed": (
                stream_result["status"] == 200
                and stream_result["done"]
                and stream_result["text"] == en_result["text"]
            ),
            "status": stream_result["status"],
            "stream_text": stream_result["text"],
            "non_stream_text": en_result["text"],
            "events": stream_result["events"],
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
                and reconnect["done"]
            ),
            "cancel": cancellation,
            "reconnect_status": reconnect["status"],
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

    below_30s = _resize_wav(en_sample.audio_bytes, 29.9)
    below_30s_result = await _post_raw_audio(
        session,
        args,
        below_30s,
        "below-30s.wav",
        "en",
    )
    checks.append(_expect_success("below_30s_audio", below_30s_result))

    beyond_30s = _resize_wav(en_sample.audio_bytes, 30.1)
    beyond_30s_result = await _post_raw_audio(
        session,
        args,
        beyond_30s,
        "beyond-30s.wav",
        "en",
    )
    checks.append(_expect_success("beyond_30s_audio", beyond_30s_result))
    return checks


async def _run_soak(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    samples: list[PreparedSample],
    *,
    concurrencies: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    randomizer = random.Random(123)
    stage_duration_s = args.duration_s / len(concurrencies)
    stages: list[dict[str, Any]] = []
    chaos_events: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        deadline = time.monotonic() + stage_duration_s
        counters = {
            "requests": 0,
            "successes": 0,
            "unexpected_errors": 0,
            "en": 0,
            "zh": 0,
        }
        latencies: list[float] = []

        async def worker(worker_id: int) -> None:
            index = worker_id
            while time.monotonic() < deadline:
                sample = samples[index % len(samples)]
                index += concurrency
                result = await _post_transcription(session, args, sample)
                counters["requests"] += 1
                counters[sample.language] += 1
                latencies.append(result["latency_s"])
                if result["status"] == 200 and result["text"]:
                    counters["successes"] += 1
                else:
                    counters["unexpected_errors"] += 1

        async def chaos_worker() -> None:
            event_number = 0
            while time.monotonic() < deadline:
                await asyncio.sleep(min(30.0, max(0.0, deadline - time.monotonic())))
                if time.monotonic() >= deadline:
                    break
                sample = randomizer.choice(samples)
                if event_number % 2 == 0:
                    result = await _post_raw_audio(
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
                        }
                    )
                else:
                    result = await _cancel_stream(session, args, sample)
                    reconnect = await _post_streaming_transcription(
                        session, args, sample
                    )
                    chaos_events.append(
                        {
                            "stage_concurrency": concurrency,
                            "kind": "cancel_reconnect",
                            "status": result["status"],
                            "passed": (
                                result["status"] == 200
                                and result["received_event"]
                                and reconnect["done"]
                            ),
                        }
                    )
                event_number += 1

        started = time.monotonic()
        await asyncio.gather(
            *(worker(worker_id) for worker_id in range(concurrency)),
            chaos_worker(),
        )
        stages.append(
            {
                "concurrency": concurrency,
                "duration_s": time.monotonic() - started,
                **counters,
                "throughput_requests_per_s": counters["successes"]
                / max(time.monotonic() - started, 1e-9),
                "latency_mean_s": statistics.mean(latencies) if latencies else None,
                "latency_p95_s": _percentile(latencies, 95),
                "memory": _memory_checkpoint(
                    f"after_concurrency_{concurrency}",
                    args.gpu_index,
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


async def _post_raw_audio(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    audio_bytes: bytes,
    filename: str,
    language: str,
) -> dict[str, Any]:
    form = aiohttp.FormData()
    form.add_field("model", args.model_path)
    form.add_field("language", language)
    form.add_field("response_format", "json")
    form.add_field(
        "file",
        audio_bytes,
        filename=filename,
        content_type="audio/wav",
    )
    started = time.perf_counter()
    async with session.post(_url(args), data=form) as response:
        body = await response.text()
        text = ""
        if response.status == 200:
            try:
                text = str(json.loads(body).get("text", ""))
            except json.JSONDecodeError:
                pass
        return {
            "status": response.status,
            "text": text,
            "body": body[:500],
            "latency_s": time.perf_counter() - started,
        }


async def _post_streaming_transcription(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    sample: PreparedSample,
) -> dict[str, Any]:
    form = _stream_form(args, sample)
    events: list[dict[str, Any]] = []
    async with session.post(_url(args), data=form) as response:
        async for raw_line in response.content:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    done_events = [
        event for event in events if event.get("type") == "transcript.text.done"
    ]
    return {
        "status": response.status,
        "events": len(events),
        "done": bool(done_events),
        "text": str(done_events[-1].get("text", "")) if done_events else "",
    }


async def _cancel_stream(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    sample: PreparedSample,
) -> dict[str, Any]:
    response = await session.post(_url(args), data=_stream_form(args, sample))
    first_line = ""
    try:
        first_line = (
            await asyncio.wait_for(response.content.readline(), timeout=10)
        ).decode(errors="replace")
    finally:
        status = response.status
        response.close()
    return {
        "status": status,
        "received_event": first_line.startswith("data:"),
    }


def _stream_form(args: argparse.Namespace, sample: PreparedSample) -> aiohttp.FormData:
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
    async with session.get(f"http://{args.host}:{args.port}/health") as response:
        await response.read()
        return response.status


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
    target_samples = round(duration_s * SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        old_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_length = round(len(audio) * SAMPLE_RATE / sample_rate)
        new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        audio = np.interp(new_positions, old_positions, audio).astype(np.float32)
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


def _memory_checkpoint(label: str, gpu_index: int) -> dict[str, Any]:
    output = _command(
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used,memory.free,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    return {
        "label": label,
        "monotonic_s": time.monotonic(),
        "nvidia_smi_csv": output,
    }


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}/v1/audio/transcriptions"


def main() -> None:
    args = parse_args()
    result = asyncio.run(main_async(args))
    output = Path(os.path.abspath(args.output))
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "output": str(output)}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
