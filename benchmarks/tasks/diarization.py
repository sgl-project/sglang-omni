# SPDX-License-Identifier: Apache-2.0
"""Standalone diarization requests and speaker metrics for audio benchmarks."""

from __future__ import annotations

import asyncio
import math
import mimetypes
import time
from dataclasses import asdict
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.transcribe_diarize_metrics import (
    _parse_timestamped_speaker_segments,
    timestamp_der_segments,
)


def validate_diarization(payload, *, duration: float) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("Expected a diarization response object")
    returned_duration = payload.get("duration")
    if (
        type(returned_duration) not in (int, float)
        or not math.isfinite(returned_duration)
        or abs(returned_duration - duration) > 0.02
    ):
        raise ValueError("Diarization duration does not match the input audio")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Diarization response is missing speaker segments")
    for item in segments:
        if not isinstance(item, dict):
            raise ValueError("Invalid diarization segment")
        start, end, speaker = item.get("start"), item.get("end"), item.get("speaker")
        if (
            type(start) not in (int, float)
            or type(end) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end <= returned_duration + 1e-6
            or not isinstance(speaker, str)
            or not speaker
        ):
            raise ValueError("Invalid diarization segment bounds or speaker")
    return segments


def _pcm16(path: Path) -> bytes:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor)
    return np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes()


async def _live_request(session, api_url, pcm, result, started):
    segments = []
    url = api_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    async with session.ws_connect(url + "/stream", receive_timeout=90) as socket:

        async def receive_until(expected):
            while True:
                event = await socket.receive_json()
                if event.get("type") == "error":
                    raise ValueError(event.get("message", "Diarization stream failed"))
                if event.get("type") == "diarization.update":
                    if result.diarization_first_update_s is None:
                        result.diarization_first_update_s = (
                            time.perf_counter() - started
                        )
                    segments.extend(event["segments"])
                if event.get("type") == expected:
                    return event

        await receive_until("session.ready")
        # Flow-controlled 100 ms packets, sent without real-time pacing.
        for offset in range(0, len(pcm), 3200):
            sent = time.perf_counter()
            await socket.send_bytes(pcm[offset : offset + 3200])
            await receive_until("audio.ack")
            result.diarization_ack_latency_s.append(time.perf_counter() - sent)
        await socket.send_json({"type": "audio.end"})
        done = await receive_until("diarization.done")
        return {"duration": done["duration"], "segments": segments}


def make_diarization_send_fn(model_name: str, api_url: str, *, stream: bool = False):
    async def send(session: aiohttp.ClientSession, sample) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        path = Path(getattr(sample, "audio_path", None) or sample.ref_audio)
        try:
            result.audio_duration_s = sf.info(path).duration
            audio = _pcm16(path) if stream else path.read_bytes()
        except (OSError, RuntimeError, ValueError) as exc:
            result.error = str(exc)
            return result
        started = time.perf_counter()
        try:
            if stream:
                async with asyncio.timeout(session.timeout.total or 1800):
                    payload = await _live_request(
                        session, api_url, audio, result, started
                    )
            else:
                form = aiohttp.FormData()
                form.add_field("model", model_name)
                form.add_field("response_format", "json")
                form.add_field(
                    "file",
                    audio,
                    filename=path.name,
                    content_type=mimetypes.guess_type(path.name)[0]
                    or "application/octet-stream",
                )
                async with session.post(api_url, data=form) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"HTTP {response.status}: {await response.text()}"
                        )
                    payload = await response.json()
            result.diarization_segments = validate_diarization(
                payload, duration=result.audio_duration_s
            )
            result.is_success = True
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - started
        if result.is_success and result.audio_duration_s > 0:
            result.rtf = result.latency_s / result.audio_duration_s
        return result

    return send


async def run_diarization(
    samples,
    *,
    host: str,
    port: int,
    model_path: str,
    concurrency: int,
    warmup: int = 0,
    disable_tqdm: bool = True,
    stream: bool = False,
    request_timeout_s: int = 1800,
):
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=concurrency,
            warmup=warmup,
            disable_tqdm=disable_tqdm,
            timeout_s=request_timeout_s,
        )
    )
    outputs = await runner.run(
        samples,
        make_diarization_send_fn(
            model_path,
            f"http://{host}:{port}/v1/audio/diarizations",
            stream=stream,
        ),
    )
    return outputs, runner.wall_clock_s


def build_diarization_evaluation(
    samples, outputs, wall_clock_s, *, collar: float = 0.0
):
    if not math.isfinite(collar) or collar < 0:
        raise ValueError("DER collar must be finite and nonnegative")
    by_id = {item.request_id: item for item in outputs}
    if len(by_id) != len(outputs):
        raise ValueError("Duplicate diarization result identities")
    sample_ids = {sample.sample_id for sample in samples}
    if len(sample_ids) != len(samples) or set(by_id) - sample_ids:
        raise ValueError("Diarization sample and result identities do not match")
    aligned_outputs = []
    per_sample = []
    total = false_alarm = missed = confusion = 0.0
    count_correct = count_error = quality_evaluated = invalid_references = 0
    for sample in samples:
        result = by_id.get(sample.sample_id)
        if result is None:
            result = RequestResult(request_id=sample.sample_id, error="Missing result")
        aligned_outputs.append(result)
        record = asdict(result)
        record["id"] = sample.sample_id
        reference_text = getattr(sample, "expected_text", None)
        if reference_text is not None:
            reference = _parse_timestamped_speaker_segments(reference_text)
            if reference_text.strip() and not reference:
                invalid_references += 1
                record["quality_error"] = "Reference has no speaker timestamps"
            else:
                # Failed requests remain in the quality denominator as all misses.
                prediction = (
                    [
                        (item["start"], item["end"], item["speaker"])
                        for item in (result.diarization_segments or [])
                    ]
                    if result.is_success
                    else []
                )
                detail = timestamp_der_segments(reference, prediction, collar=collar)
                reference_count = len({item[2] for item in reference})
                predicted_count = len({item[2] for item in prediction})
                absolute_error = abs(reference_count - predicted_count)
                quality_evaluated += 1
                count_correct += absolute_error == 0
                count_error += absolute_error
                total += detail["total"]
                false_alarm += detail["false_alarm"]
                missed += detail["missed_detection"]
                confusion += detail["confusion"]
                record.update(
                    diarization_metrics=detail,
                    reference_speaker_count=reference_count,
                    predicted_speaker_count=predicted_count,
                )
        per_sample.append(record)
    metrics = {
        "der": (false_alarm + missed + confusion) / total if total else None,
        "reference_speaker_seconds": total,
        "false_alarm_seconds": false_alarm,
        "missed_speaker_seconds": missed,
        "confusion_seconds": confusion,
        "speaker_count_accuracy": (
            count_correct / quality_evaluated if quality_evaluated else None
        ),
        "speaker_count_mae": (
            count_error / quality_evaluated if quality_evaluated else None
        ),
        "quality_evaluated": quality_evaluated,
        "invalid_references": invalid_references,
        "collar_s": collar,
    }
    successes = sum(item.is_success for item in aligned_outputs)
    speed = compute_speed_metrics(aligned_outputs, wall_clock_s=wall_clock_s)
    speed.update(
        rtfx=speed.get("audio_throughput_s_per_s", 0.0),
        throughput_samples_per_s=speed.get("throughput_qps", 0.0),
    )
    for name in (
        "latency_mean_s",
        "latency_median_s",
        "latency_p95_s",
        "latency_p99_s",
        "rtf_mean",
        "rtf_p95",
    ):
        speed.setdefault(name, None)
    first_updates = [
        item.diarization_first_update_s
        for item in outputs
        if item.is_success and item.diarization_first_update_s is not None
    ]
    ack_latencies = [
        latency
        for item in outputs
        if item.is_success
        for latency in item.diarization_ack_latency_s
    ]
    for label, values in (
        ("diarization_first_update", first_updates),
        ("diarization_ack_latency", ack_latencies),
    ):
        if values:
            speed[f"{label}_mean_s"] = float(np.mean(values))
            speed[f"{label}_p95_s"] = float(np.percentile(values, 95))
    percent = dict(metrics)
    for name in ("der", "speaker_count_accuracy"):
        if percent[name] is not None:
            percent[name] *= 100
    return {
        "task": "diarize",
        "summary": {
            "total_samples": len(samples),
            "evaluated": successes,
            "skipped": len(samples) - successes,
            "corpus_wer": None,
            "wer_per_sample_max": None,
        },
        "speed": speed,
        "diarization_metrics": metrics,
        "diarization_metrics_percent": percent,
        "per_sample": per_sample,
    }
