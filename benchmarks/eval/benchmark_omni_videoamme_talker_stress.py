# SPDX-License-Identifier: Apache-2.0
# note (luojiaxuan):
"""Sustained-load Video-AMME Talker stress benchmark for Qwen3-Omni.

This benchmark keeps the same task shape as Video-AMME Talker
(video + spoken question -> text + speech), but drives a longer load profile
than the correctness CI. It repeats the Video-AMME subset across one or more
concurrency levels, records per-request arrival/start/finish timing, and emits
time-bucket summaries that make latency drift and client-side admission lag
visible.

Run from the repo root after launching a speech-mode Qwen3-Omni server::

    python -m benchmarks.eval.benchmark_omni_videoamme_talker_stress \\
        --base-url http://localhost:8000 \\
        --repo-id zhaochenyang20/Video_AMME_ci \\
        --max-samples 50 \\
        --request-count 150 \\
        --concurrency-levels 8,16,32 \\
        --request-rate 1.0 \\
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408

Results are written to
``<output-dir>/videoamme_talker_stress_results.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import logging
import math
import os
import struct
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.utils import (
    get_wav_duration,
    save_json_results,
    wait_for_service,
)
from benchmarks.dataset.videomme import (
    DEFAULT_VIDEOAMME_REPO_ID,
    VideoAMMESample,
    load_videoamme_samples,
)
from benchmarks.metrics.performance import compute_speed_metrics, print_speed_summary
from benchmarks.metrics.video import (
    compute_videomme_metrics,
    print_videomme_accuracy_summary,
)
from benchmarks.tasks.video_understanding import (
    VIDEOAMME_REQUEST_TEXT,
    build_videomme_result_records,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY_LEVELS = [8, 16, 32]
DEFAULT_REQUEST_COUNT = 150
DEFAULT_BUCKET_SECONDS = 60.0
PROFILE_ENDPOINT_TIMEOUT_S = 30
DEFAULT_ASR_TRANSCRIBE_CONCURRENCY = int(
    os.getenv("QWEN3_ASR_CONCURRENCY", os.getenv("SEEDTTS_ASR_CONCURRENCY", "32"))
)


@dataclass
class StressRequestRecord:
    stage_id: str
    index: int
    request_id: str
    source_sample_id: str
    scheduled_offset_s: float
    start_offset_s: float
    finish_offset_s: float
    scheduled_lag_s: float
    client_queue_wait_s: float
    result: RequestResult


def _build_base_url(args: argparse.Namespace) -> str:
    return args.base_url or f"http://{args.host}:{args.port}"


def _parse_positive_int_list(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _parse_request_rate(raw: str) -> float:
    if raw.lower() in {"inf", "infinity"}:
        return float("inf")
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive or inf")
    return value


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


def _repeat_samples(
    samples: list[VideoAMMESample],
    *,
    request_count: int,
    stage_id: str,
) -> list[VideoAMMESample]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if not samples:
        raise ValueError("at least one Video-AMME sample is required")

    repeated: list[VideoAMMESample] = []
    for index in range(request_count):
        sample = samples[index % len(samples)]
        sample_id = f"{stage_id}_{index:05d}_{_safe_id(sample.sample_id)}"
        repeated.append(replace(sample, sample_id=sample_id, question_id=sample_id))
    return repeated


def _repeat_source_sample_ids(
    samples: list[VideoAMMESample],
    *,
    request_count: int,
) -> list[str]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if not samples:
        raise ValueError("at least one Video-AMME sample is required")
    return [samples[index % len(samples)].sample_id for index in range(request_count)]


def _scheduled_offsets(request_count: int, request_rate: float) -> list[float]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if math.isinf(request_rate):
        return [0.0] * request_count
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    interval_s = 1.0 / request_rate
    return [index * interval_s for index in range(request_count)]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def _drift(values: list[float]) -> dict[str, float | None]:
    if len(values) < 2:
        return {"first_half_mean": None, "last_half_mean": None, "delta": None}
    midpoint = max(1, len(values) // 2)
    first = values[:midpoint]
    last = values[midpoint:]
    if not last:
        last = values[-midpoint:]
    first_mean = _mean(first)
    last_mean = _mean(last)
    delta = None
    if first_mean is not None and last_mean is not None:
        delta = round(last_mean - first_mean, 4)
    return {
        "first_half_mean": first_mean,
        "last_half_mean": last_mean,
        "delta": delta,
    }


def _bucket_records(
    records: list[StressRequestRecord],
    *,
    bucket_seconds: float,
) -> list[dict[str, Any]]:
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")

    buckets: dict[int, list[StressRequestRecord]] = {}
    for record in records:
        bucket_index = int(record.finish_offset_s // bucket_seconds)
        buckets.setdefault(bucket_index, []).append(record)

    summaries: list[dict[str, Any]] = []
    for bucket_index in sorted(buckets):
        bucket = buckets[bucket_index]
        successes = [record for record in bucket if record.result.is_success]
        latencies = [record.result.latency_s for record in successes]
        rtfs = [
            record.result.rtf
            for record in successes
            if record.result.rtf > 0 and math.isfinite(record.result.rtf)
        ]
        audio_returned = sum(
            1 for record in successes if record.result.audio_duration_s > 0
        )
        summaries.append(
            {
                "bucket_index": bucket_index,
                "start_s": round(bucket_index * bucket_seconds, 3),
                "end_s": round((bucket_index + 1) * bucket_seconds, 3),
                "total_requests": len(bucket),
                "completed_requests": len(successes),
                "failed_requests": len(bucket) - len(successes),
                "throughput_qps": round(len(successes) / bucket_seconds, 4),
                "audio_returned": audio_returned,
                "latency_s": _summary(latencies),
                "rtf": _summary(rtfs),
                "scheduled_lag_s": _summary(
                    [record.scheduled_lag_s for record in bucket]
                ),
                "client_queue_wait_s": _summary(
                    [record.client_queue_wait_s for record in bucket]
                ),
            }
        )
    return summaries


def _stress_summary(
    records: list[StressRequestRecord],
    *,
    bucket_seconds: float,
) -> dict[str, Any]:
    successes = [record for record in records if record.result.is_success]
    latencies = [record.result.latency_s for record in successes]
    rtfs = [
        record.result.rtf
        for record in successes
        if record.result.rtf > 0 and math.isfinite(record.result.rtf)
    ]
    audio_returned = sum(
        1 for record in successes if record.result.audio_duration_s > 0
    )
    scheduled_lags = [record.scheduled_lag_s for record in records]
    queue_waits = [record.client_queue_wait_s for record in records]
    finish_offsets = [record.finish_offset_s for record in records]

    return {
        "observed_duration_s": round(max(finish_offsets), 4) if finish_offsets else 0,
        "total_requests": len(records),
        "completed_requests": len(successes),
        "failed_requests": len(records) - len(successes),
        "audio_returned": audio_returned,
        "audio_return_rate": (
            round(audio_returned / len(successes), 4) if successes else None
        ),
        "latency_s": _summary(latencies),
        "rtf": _summary(rtfs),
        "scheduled_lag_s": _summary(scheduled_lags),
        "client_queue_wait_s": _summary(queue_waits),
        "latency_mean_drift_s": _drift(latencies),
        "rtf_mean_drift": _drift(rtfs),
        "time_buckets": _bucket_records(records, bucket_seconds=bucket_seconds),
    }


def _record_to_json(record: StressRequestRecord) -> dict[str, Any]:
    data = asdict(record)
    data["result"] = asdict(record.result)
    return data


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
) -> dict[str, Any] | None:
    try:
        async with session.get(url, timeout=PROFILE_ENDPOINT_TIMEOUT_S) as response:
            response.raise_for_status()
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("Failed to GET %s: %s", url, exc)
        return None


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        async with session.post(
            url,
            json=payload,
            timeout=PROFILE_ENDPOINT_TIMEOUT_S,
        ) as response:
            body = await response.json()
            response.raise_for_status()
            return body
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.warning("Failed to POST %s: %s", url, exc)
        return None


async def _start_request_profile(
    session: aiohttp.ClientSession,
    base_url: str,
    *,
    run_id: str,
    event_dir: str,
) -> dict[str, Any] | None:
    return await _post_json(
        session,
        f"{base_url}/start_request_profile",
        {"run_id": run_id, "event_dir": event_dir},
    )


async def _stop_request_profile(
    session: aiohttp.ClientSession,
    base_url: str,
    *,
    run_id: str,
) -> dict[str, Any] | None:
    return await _post_json(
        session,
        f"{base_url}/stop_request_profile",
        {"run_id": run_id},
    )


def _make_talker_send_fn(
    *,
    model_name: str,
    api_url: str,
    max_tokens: int,
    temperature: float,
    video_fps: float | None,
    video_max_frames: int | None,
    video_min_pixels: int | None,
    video_max_pixels: int | None,
    video_total_pixels: int | None,
    audio_output_dir: str | None,
):
    async def send_fn(
        session: aiohttp.ClientSession,
        sample: VideoAMMESample,
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text=VIDEOAMME_REQUEST_TEXT[:60],
        )
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": VIDEOAMME_REQUEST_TEXT}],
            "videos": [sample.video_path],
            "audios": [sample.audio_path],
            "modalities": ["text", "audio"],
            "audio": {"format": "wav"},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if video_fps is not None:
            payload["video_fps"] = video_fps
        if video_max_frames is not None:
            payload["video_max_frames"] = video_max_frames
        if video_min_pixels is not None:
            payload["video_min_pixels"] = video_min_pixels
        if video_max_pixels is not None:
            payload["video_max_pixels"] = video_max_pixels
        if video_total_pixels is not None:
            payload["video_total_pixels"] = video_total_pixels

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, json=payload) as response:
                body = await response.json()
                if response.status >= 400:
                    result.error = body.get("detail", str(body))
                    return result

            message = body.get("choices", [{}])[0].get("message", {})
            result.text = message.get("content", "") or ""
            audio_obj = message.get("audio")
            if not isinstance(audio_obj, dict):
                result.error = "No audio in response"
                return result
            audio_b64 = audio_obj.get("data", "")
            if not audio_b64:
                result.error = "Empty audio data in response"
                return result

            try:
                wav_bytes = base64.b64decode(audio_b64, validate=True)
                result.audio_duration_s = round(get_wav_duration(wav_bytes), 4)
            except (binascii.Error, ValueError, struct.error) as exc:
                result.error = f"Invalid audio data: {exc}"
                return result

            if audio_output_dir:
                os.makedirs(audio_output_dir, exist_ok=True)
                wav_path = os.path.join(audio_output_dir, f"{sample.sample_id}.wav")
                with open(wav_path, "wb") as file_obj:
                    file_obj.write(wav_bytes)
                result.wav_path = wav_path

            usage = body.get("usage", {})
            result.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            result.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            result.is_success = True
            result.engine_time_s = time.perf_counter() - start_time
            if result.audio_duration_s > 0:
                result.rtf = result.engine_time_s / result.audio_duration_s
            if result.completion_tokens > 0 and result.engine_time_s > 0:
                result.tok_per_s = result.completion_tokens / result.engine_time_s
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        return result

    return send_fn


async def _run_load_stage(
    *,
    stage_id: str,
    samples: list[VideoAMMESample],
    source_sample_ids: list[str] | None = None,
    send_fn,
    max_concurrency: int,
    request_rate: float,
    timeout_s: int,
    disable_tqdm: bool,
) -> tuple[list[RequestResult], list[StressRequestRecord], float]:
    if source_sample_ids is not None and len(source_sample_ids) != len(samples):
        raise ValueError("source_sample_ids must match samples length")

    semaphore = asyncio.Semaphore(max_concurrency)
    offsets = _scheduled_offsets(len(samples), request_rate)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    pbar = tqdm(total=len(samples), disable=disable_tqdm)
    stage_start = time.perf_counter()

    async def run_one(index: int, sample: VideoAMMESample) -> StressRequestRecord:
        scheduled_offset = offsets[index]
        scheduled_at = stage_start + scheduled_offset
        await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
        arrived_at = time.perf_counter()
        scheduled_lag_s = max(0.0, arrived_at - scheduled_at)

        async with semaphore:
            started_at = time.perf_counter()
            result = await send_fn(session, sample)
        finished_at = time.perf_counter()
        pbar.update(1)
        return StressRequestRecord(
            stage_id=stage_id,
            index=index,
            request_id=result.request_id,
            source_sample_id=(
                source_sample_ids[index]
                if source_sample_ids is not None
                else sample.sample_id
            ),
            scheduled_offset_s=round(scheduled_offset, 4),
            start_offset_s=round(started_at - stage_start, 4),
            finish_offset_s=round(finished_at - stage_start, 4),
            scheduled_lag_s=round(scheduled_lag_s, 4),
            client_queue_wait_s=round(started_at - arrived_at, 4),
            result=result,
        )

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                asyncio.create_task(run_one(index, sample))
                for index, sample in enumerate(samples)
            ]
            records = list(await asyncio.gather(*tasks))
    finally:
        pbar.close()

    wall_clock_s = time.perf_counter() - stage_start
    records.sort(key=lambda record: record.index)
    return [record.result for record in records], records, wall_clock_s


async def run_videoamme_talker_stress(args: argparse.Namespace) -> dict[str, Any]:
    base_url = _build_base_url(args)
    api_url = f"{base_url}/v1/chat/completions"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_samples = load_videoamme_samples(
        repo_id=args.repo_id,
        split=args.split,
        max_samples=args.max_samples,
    )
    if not base_samples:
        raise ValueError("No Video-AMME samples loaded")

    run_id = args.profile_run_id or f"videoamme-talker-stress-{int(time.time())}"
    event_dir = str(Path(args.profile_event_dir or (output_dir / "events")).resolve())
    save_audio = args.save_audio or args.compute_wer

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        health_before = await _get_json(session, f"{base_url}/health")
        profile_info = None
        if not args.no_profile:
            profile_info = await _start_request_profile(
                session,
                base_url,
                run_id=run_id,
                event_dir=event_dir,
            )

    stages: list[dict[str, Any]] = []
    try:
        for concurrency in args.concurrency_levels:
            stage_id = f"c{concurrency}"
            stage_samples = _repeat_samples(
                base_samples,
                request_count=args.request_count,
                stage_id=stage_id,
            )
            source_sample_ids = _repeat_source_sample_ids(
                base_samples,
                request_count=args.request_count,
            )
            audio_output_dir = (
                str(output_dir / "audio" / stage_id) if save_audio else None
            )
            send_fn = _make_talker_send_fn(
                model_name=args.model,
                api_url=api_url,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                video_fps=args.video_fps,
                video_max_frames=args.video_max_frames,
                video_min_pixels=args.video_min_pixels,
                video_max_pixels=args.video_max_pixels,
                video_total_pixels=args.video_total_pixels,
                audio_output_dir=audio_output_dir,
            )
            request_results, records, wall_clock_s = await _run_load_stage(
                stage_id=stage_id,
                samples=stage_samples,
                source_sample_ids=source_sample_ids,
                send_fn=send_fn,
                max_concurrency=concurrency,
                request_rate=args.request_rate,
                timeout_s=args.timeout_s,
                disable_tqdm=args.disable_tqdm,
            )
            per_sample = build_videomme_result_records(stage_samples, request_results)
            accuracy = compute_videomme_metrics(per_sample)
            speed = compute_speed_metrics(request_results, wall_clock_s=wall_clock_s)
            stress = _stress_summary(records, bucket_seconds=args.bucket_seconds)
            wer_results = None
            if args.compute_wer:
                from benchmarks.metrics.wer import print_wer_summary
                from benchmarks.tasks.tts import compute_text_audio_consistency

                wer_results = compute_text_audio_consistency(
                    request_results,
                    args.lang,
                    args.asr_device,
                    asr_concurrency=args.asr_concurrency,
                )

            print_videomme_accuracy_summary(
                accuracy,
                args.model,
                title=f"Video-AMME Talker Stress Accuracy {stage_id}",
            )
            print_speed_summary(
                speed,
                args.model,
                concurrency,
                title=f"Video-AMME Talker Stress Speed {stage_id}",
            )
            if wer_results:
                print_wer_summary(wer_results["summary"], args.model)
            logger.info(
                "stage=%s latency_drift=%s queue_wait_p95=%s audio_return_rate=%s",
                stage_id,
                stress["latency_mean_drift_s"]["delta"],
                stress["client_queue_wait_s"]["p95"],
                stress["audio_return_rate"],
            )

            stages.append(
                {
                    "stage_id": stage_id,
                    "concurrency": concurrency,
                    "request_count": len(stage_samples),
                    "request_rate": (
                        "inf" if math.isinf(args.request_rate) else args.request_rate
                    ),
                    "wall_clock_s": round(wall_clock_s, 4),
                    "accuracy": accuracy,
                    "speed": speed,
                    "stress": stress,
                    "wer": wer_results,
                    "per_sample": per_sample,
                    "requests": [_record_to_json(record) for record in records],
                }
            )
    finally:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            profile_after = None
            if not args.no_profile:
                profile_after = await _stop_request_profile(
                    session,
                    base_url,
                    run_id=run_id,
                )
            health_after = await _get_json(session, f"{base_url}/health")

    output = {
        "config": {
            "base_url": base_url,
            "model": args.model,
            "repo_id": args.repo_id,
            "split": args.split,
            "base_sample_count": len(base_samples),
            "request_count": args.request_count,
            "concurrency_levels": args.concurrency_levels,
            "request_rate": (
                "inf" if math.isinf(args.request_rate) else args.request_rate
            ),
            "bucket_seconds": args.bucket_seconds,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "video_fps": args.video_fps,
            "video_max_frames": args.video_max_frames,
            "video_min_pixels": args.video_min_pixels,
            "video_max_pixels": args.video_max_pixels,
            "video_total_pixels": args.video_total_pixels,
            "save_audio": save_audio,
            "compute_wer": args.compute_wer,
            "asr_device": args.asr_device,
            "asr_concurrency": args.asr_concurrency,
            "lang": args.lang,
            "profile_run_id": run_id,
            "profile_event_dir": event_dir if not args.no_profile else None,
        },
        "profile": profile_info,
        "profile_stop": profile_after,
        "health_before": health_before,
        "health_after": health_after,
        "stages": stages,
    }
    save_json_results(
        output,
        str(output_dir),
        "videoamme_talker_stress_results.json",
    )
    return output


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Stress Video-AMME Talker with sustained text+audio output load."
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="qwen3-omni")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_VIDEOAMME_REPO_ID)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--request-count", type=int, default=DEFAULT_REQUEST_COUNT)
    parser.add_argument(
        "--concurrency-levels",
        type=_parse_positive_int_list,
        default=DEFAULT_CONCURRENCY_LEVELS,
        help="Comma-separated concurrency levels, e.g. 8,16,32.",
    )
    parser.add_argument(
        "--request-rate",
        type=_parse_request_rate,
        default=float("inf"),
        help="Offered requests per second for each stage, or inf.",
    )
    parser.add_argument(
        "--bucket-seconds",
        type=float,
        default=DEFAULT_BUCKET_SECONDS,
        help="Time bucket size for latency/backlog summaries.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--video-fps", type=float, default=2)
    parser.add_argument("--video-max-frames", type=int, default=128)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=401408)
    parser.add_argument("--video-total-pixels", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results/videoamme_stress")
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--compute-wer", action="store_true")
    parser.add_argument("--asr-device", type=str, default="cuda:0")
    parser.add_argument(
        "--asr-concurrency",
        type=int,
        default=DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    )
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--profile-run-id", type=str, default=None)
    parser.add_argument("--profile-event-dir", type=str, default=None)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    if args.request_count <= 0:
        raise ValueError("--request-count must be positive")
    if args.bucket_seconds <= 0:
        raise ValueError("--bucket-seconds must be positive")

    base_url = _build_base_url(args)
    wait_for_service(base_url)
    asyncio.run(run_videoamme_talker_stress(args))


if __name__ == "__main__":
    main()
